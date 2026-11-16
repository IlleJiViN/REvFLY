"""Thread-safe HTTP API server and optional workbench for REvFLY."""

from __future__ import annotations

import json
import math
import mimetypes
import queue
import re
import threading
import urllib.parse
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import numpy as np

from .. import __version__
from ..assets import resolve_asset_root
from ..bio.malecns_pools import premotor_unavailability
from ..contracts import (
    AssetValidationError,
    CONTROL_STEPS,
    CONTROL_TIMESTEP_S,
    DEFAULT_FEEDBACK_KD,
    DEFAULT_FEEDBACK_KP,
    InitialState,
    LF_ACTUATOR_NAMES,
    LF_JOINT_GROUPS,
    LF_JOINT_NAMES,
    LF_QPOS_ADDRESSES,
    STATE_STEPS,
    SessionSpec,
    require_int,
    require_real_float,
    validate_matrix,
    validate_time_grid,
    validate_vector,
)
from ..evidence import write_json_atomic
from ..goals import validate_multijoint_goal
from ..rate.model import RATE_ARCHITECTURES
from .core import DEFAULT_THRESHOLD_DEG, EngineV2, EngineV2Config
from .multijoint import (
    MULTIJOINT_GOAL_FAMILIES,
    MultiJointEngine,
    MultiJointEngineConfig,
    evaluate_bio_realization,
)
from .openapi import build_openapi_v2_spec

MAX_BODY_BYTES = 1_048_576
MAX_QUEUED_JOBS = 16
MAX_RETAINED_JOBS = 128
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
_JOB_ID = re.compile(r"^[0-9a-f]{32}$")
_ALLOWED_REQUEST_KEYS = frozenset(
    {
        "goal_scale",
        "target_rad",
        "time_s",
        "mode",
        "threshold_deg",
        "correction_steps",
        "correction_lr_hz",
        "correction_optimizer",
        "feedback_kp",
        "feedback_kd",
        "initial_tibia_delta_deg",
        "render",
        "check_directions",
    }
)
_MULTIJOINT_REQUEST_KEYS = frozenset({
    "scope", "target_rad", "time_s", "goal_scale", "goal_family", "joint_names",
    "mode", "threshold_deg", "correction_steps", "feedback_kp", "feedback_kd",
    "initial_qpos_delta_deg", "render", "bio_mode", "bio_seed",
    "bio_physics_vjp_steps", "bio_population_scale",
})
_MULTIJOINT_MODES = ("fast", "auto", "refine", "closed_loop")
_BIO_ONLY_MODES = (
    "deterministic_isi",
    "poisson_refractory",
    "direct_lif",
    "premotor_connectome",
)
_MULTIJOINT_BIO_MODES = ("none", *_BIO_ONLY_MODES)
_BIO_REQUEST_KEYS = frozenset({
    "scope", "bio_mode", "source_job_id", "goal_family", "goal_scale",
    "command_pool_rates_hz", "target_excitation", "target_rad",
    "tau_bridge_ms", "population_scale", "premotor_opt_steps",
    "physics_vjp_steps", "seed",
})
_RATE_STEPS = CONTROL_STEPS * 2
_MULTIJOINT_DEFAULTS = {
    "goal_scale": 0.975,
    "family": "kinematic_teacher_scaled",
    "mode": "auto",
    "threshold_deg": DEFAULT_THRESHOLD_DEG,
    "correction_steps": 4,
    "feedback_kp": 8.0,
    "feedback_kd": 0.04,
    "bio_mode": "none",
}
_ZIP_SUFFIXES = frozenset({".json", ".npz", ".mp4", ".webm", ".png"})
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_UI_ASSETS = {
    "/": ("apps/templates/rate_control.html", "text/html; charset=utf-8"),
    "/index.html": ("apps/templates/rate_control.html", "text/html; charset=utf-8"),
    "/static/rate_control.js": ("static/rate_control.js", "text/javascript; charset=utf-8"),
    "/static/rate_control.css": ("static/rate_control.css", "text/css; charset=utf-8"),
}


class RequestError(ValueError):
    """Structured HTTP 4xx/5xx validation error."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strict_json_loads(raw_text: str) -> Any:
    def _reject_constant(token: str) -> Any:
        raise RequestError(f"Invalid non-finite JSON token: {token}")

    def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        obj: dict[str, Any] = {}
        for key, value in pairs:
            if key in obj:
                raise RequestError(f"Duplicate JSON field: {key!r}")
            obj[key] = value
        return obj

    return json.loads(raw_text, parse_constant=_reject_constant, object_pairs_hook=_reject_duplicate_keys)


def _normalize_route_path(path: str) -> str:
    """Normalize supported /v1/* and legacy /api/rate/* aliases onto /v2/* endpoints."""
    if path in ("/v1/health", "/v2/health"):
        return "/health"
    if path == "/api/rate/capabilities":
        return "/v2/capabilities"
    if path in ("/api/rate/goal", "/v1/goals", "/v2/goals"):
        return "/v2/goal"
    if path == "/api/rate/jobs":
        return "/v2/jobs"
    m_api = re.fullmatch(r"/api/rate/jobs/([0-9a-f]{32})(?:/(download|files/(.+)))?", path)
    if m_api:
        job_id = m_api.group(1)
        if m_api.group(2) == "download":
            return f"/v2/jobs/{job_id}/download"
        if m_api.group(3) is not None:
            return f"/v2/jobs/{job_id}/artifacts/{m_api.group(3)}"
        return f"/v2/jobs/{job_id}"
    m_files = re.fullmatch(r"/(v1|v2)/jobs/([0-9a-f]{32})/files/(.+)", path)
    if m_files:
        return f"/{m_files.group(1)}/jobs/{m_files.group(2)}/artifacts/{m_files.group(3)}"
    return path


def _json_copy(value: Any) -> Any:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    return json.loads(encoded)


def _public_error(exc: Exception) -> str:
    message = re.sub(r"[\x00-\x1f\x7f]+", " ", str(exc))[:400]
    message = re.sub(
        r"(?i)(bearer\s+|(?:api[_-]?key|token|password)\s*[=:]\s*)[^\s,;]+",
        r"\1[redacted]",
        message,
    )
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _safe_relative(value: Any) -> str | None:
    if type(value) is not str or not value or "\\" in value:
        return None
    posix = PurePosixPath(value)
    if posix.is_absolute() or any(part in ("", ".", "..") for part in posix.parts):
        return None
    return posix.as_posix()


def _host_without_port(host_header: str) -> str:
    host = host_header.strip()
    if host.startswith("["):
        end = host.find("]")
        return host[1:end].lower() if end != -1 else host.lower()
    if ":" in host:
        return host.rsplit(":", 1)[0].lower()
    return host.lower()


def _find_ui_root(asset_root: str | Path | None) -> Path | None:
    checkout = Path(__file__).resolve().parents[3]
    if asset_root is None:
        try:
            asset_root = resolve_asset_root()
        except AssetValidationError:
            pass
    for candidate in (asset_root, checkout, Path.cwd()):
        if candidate is not None:
            root = Path(candidate).expanduser().resolve()
            if all((root / name).is_file() for name, _ in _UI_ASSETS.values()):
                return root
    return None


def _selected_joint_names(value: Any) -> list[str]:
    if value is None:
        return list(LF_JOINT_NAMES)
    if type(value) is not list or not value or any(
        type(name) is not str or name not in LF_JOINT_NAMES for name in value
    ) or len(set(value)) != len(value):
        raise RequestError("joint_names must contain unique supported left-foreleg joint names")
    return value


@dataclass
class _Job:
    job_id: str
    output: Path
    arguments: dict[str, Any]
    scope: str = "tibia"
    status: str = "queued"
    progress: dict[str, Any] = field(default_factory=lambda: {"stage": "queued"})
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = field(default_factory=_utc_now)
    started_at: str | None = None
    finished_at: str | None = None

    def public(self) -> dict[str, Any]:
        return _json_copy(
            {
                "job_id": self.job_id,
                "status": self.status,
                "progress": self.progress,
                "result": self.result,
                "error": self.error,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }
        )


class RateControlV2App:
    """Thread-safe job queue and HTTP application wrapping EngineV2."""

    def __init__(
        self,
        *,
        ui_enabled: bool = False,
        controller: EngineV2 | None = None,
        controller_factory: Callable[[], EngineV2] | None = None,
        config: EngineV2Config | None = None,
        multijoint_controller: MultiJointEngine | None = None,
        multijoint_controller_factory: Callable[[], MultiJointEngine] | None = None,
        multijoint_config: MultiJointEngineConfig | None = None,
        output_base: str | Path | None = None,
        asset_root: str | Path | None = None,
    ) -> None:
        if type(ui_enabled) is not bool:
            raise ValueError("ui_enabled must be boolean")
        self.ui_enabled = ui_enabled
        self._config = config or EngineV2Config(asset_root=asset_root)
        self._controller_instance = controller
        self._controller_factory = controller_factory
        self._controller_lock = threading.Lock()
        cfg_src = (
            controller.config
            if (config is None and controller is not None and hasattr(controller, "config"))
            else self._config
        )
        self._multijoint_config = multijoint_config or MultiJointEngineConfig(
            backend=cfg_src.backend,
            asset_root=cfg_src.asset_root,
            runtime_root=cfg_src.runtime_root,
            physics_python=cfg_src.physics_python,
            timeout_s=cfg_src.timeout_s,
        )
        self._multijoint_instance: MultiJointEngine | None = multijoint_controller
        self._multijoint_controller_factory = multijoint_controller_factory
        self._multijoint_status: dict[str, Any] | None = None
        self._output_base = Path(output_base).expanduser().resolve() if output_base else None
        self._ui_root = _find_ui_root(asset_root or self._config.asset_root) if ui_enabled else None
        self._lock = threading.RLock()
        self._jobs: dict[str, _Job] = {}
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=MAX_QUEUED_JOBS)
        self._closed = threading.Event()
        self._worker = threading.Thread(
            target=self._work,
            name="rate-v2-physical-worker",
            daemon=True,
        )
        self._worker.start()

    def controller(self) -> EngineV2:
        with self._controller_lock:
            if self._controller_instance is None:
                if self._controller_factory is not None:
                    self._controller_instance = self._controller_factory()
                else:
                    self._controller_instance = EngineV2(self._config)
            return self._controller_instance

    def _resolved_output_base(self) -> Path:
        if self._output_base is not None:
            return self._output_base
        try:
            return (self.controller().catalog.runtime_root / "engine_jobs").resolve()
        except Exception:
            return (Path.cwd() / "results/v2_runtime/engine_jobs").resolve()

    def capabilities(self) -> dict[str, Any]:
        tibia = self._single_joint_capabilities()
        multijoint = self.multijoint_capabilities()
        tibia_bio_modes = [mode for mode in _BIO_ONLY_MODES if mode != "premotor_connectome"]
        tibia_premotor_error = self._tibia_premotor_unavailability(tibia)
        if tibia_premotor_error is None:
            tibia_bio_modes.append("premotor_connectome")
        lf15_bio_modes = [mode for mode in multijoint["bio_modes"] if mode != "none"]
        return _json_copy({
            **tibia,
            "multijoint": multijoint,
            "bio_interface": {
                "scopes": ["tibia", "lf15"],
                "modes": [
                    mode for mode in _BIO_ONLY_MODES
                    if mode in tibia_bio_modes or mode in lf15_bio_modes
                ],
                "modes_by_scope": {"tibia": tibia_bio_modes, "lf15": lf15_bio_modes},
                "unavailable_modes_by_scope": {
                    "tibia": {"premotor_connectome": tibia_premotor_error} if tibia_premotor_error else {},
                    "lf15": multijoint["unavailable_bio_modes"],
                },
                "rate_architectures": sorted(RATE_ARCHITECTURES),
            },
        })

    def _single_joint_capabilities(self) -> dict[str, Any]:
        value = self.controller().capabilities()
        if type(value) is not dict:
            raise ValueError("Controller capabilities must be a JSON object")
        return value

    def multijoint_capabilities(self) -> dict[str, Any]:
        with self._controller_lock:
            if self._multijoint_status is None:
                self._multijoint_status = self._initialize_multijoint()
            return _json_copy(self._multijoint_status)

    def _initialize_multijoint(self) -> dict[str, Any]:
        config = self._multijoint_config
        status = {
            "ready": False,
            "error": None,
            "scope": "lf15",
            "backend": config.backend,
            "profile": config.profile,
            "model_id": None,
            "checkpoint_sha256": None,
            "joint_names": list(LF_JOINT_NAMES),
            "joint_groups": {key: list(value) for key, value in LF_JOINT_GROUPS.items()},
            "actuator_names": list(LF_ACTUATOR_NAMES),
            "controlled_muscles": len(LF_ACTUATOR_NAMES),
            "teacher_assisted_muscles": 0,
            "modes": [],
            "bio_modes": [],
            "unavailable_bio_modes": {},
            "goal_families": [],
            "defaults": _MULTIJOINT_DEFAULTS,
        }
        if config.backend not in ("persistent_cached", "persistent_fresh", "fake"):
            return {**status, "error": "Left-foreleg multi-joint control requires a persistent backend"}
        try:
            controller = (
                self._multijoint_instance
                if self._multijoint_instance is not None
                else (
                    self._multijoint_controller_factory()
                    if self._multijoint_controller_factory is not None
                    else MultiJointEngine(config)
                )
            )
            self._multijoint_instance = controller
            native = controller.capabilities()
        except Exception as exc:
            return {**status, "error": _public_error(exc)}
        if not native["ready"]:
            return {**status, "error": "Multi-joint checkpoint was not found"}
        return {
            **status,
            "ready": True,
            "model_id": native["model_id"],
            "checkpoint_sha256": native["checkpoint"]["sha256"],
            "checkpoint_sha256_verified": native["checkpoint"]["expected_sha256_verified"],
            "architecture": native["checkpoint"].get("architecture"),
            "available_architectures": sorted(RATE_ARCHITECTURES),
            "is_certified_frozen_physics": native["is_certified_frozen_physics"],
            "modes": native["modes"],
            "bio_modes": [mode for mode in native["bio_modes"] if mode in _MULTIJOINT_BIO_MODES],
            "unavailable_bio_modes": native["unavailable_bio_modes"],
            "goal_families": native["goal_families"],
        }

    def _multijoint_controller(self) -> MultiJointEngine:
        status = self.multijoint_capabilities()
        if not status["ready"]:
            raise RequestError(status["error"], 503)
        assert self._multijoint_instance is not None
        return self._multijoint_instance

    def _require_premotor(self, scope: str) -> None:
        if scope == "lf15":
            status = self.multijoint_capabilities()
            reason = status["error"] or status["unavailable_bio_modes"].get("premotor_connectome")
        else:
            reason = self._tibia_premotor_unavailability(self._single_joint_capabilities())
        if reason is not None:
            raise RequestError(f"premotor_connectome unavailable: {reason}", 503)

    def _tibia_premotor_unavailability(self, capabilities: dict[str, Any]) -> str | None:
        if not capabilities.get("ready"):
            return "Tibia rate controller is not ready"
        return premotor_unavailability(self.controller().catalog.asset_root)

    def assets(self) -> dict[str, Any]:
        return _json_copy(self.controller().verify_assets())

    def ui_asset(self, route: str) -> tuple[Path, str]:
        if self._ui_root is None:
            raise RequestError("Control workbench files were not found in the checkout", 404)
        name, mime = _UI_ASSETS[route]
        path = self._ui_root / name
        if path.resolve() != path or not path.is_file():
            raise RequestError("Control workbench file not found", 404)
        return path, mime

    def goal(self, scale: Any, initial_delta_deg: Any = 0.0) -> dict[str, Any]:
        try:
            scale_val = require_real_float(scale, "scale", positive=True)
            delta_val = require_real_float(initial_delta_deg, "initial_delta_deg")
            goal = self.controller().goal_from_scale(
                scale_val, initial_tibia_delta_deg=delta_val
            )
        except Exception as exc:
            raise RequestError(str(exc)) from None
        validate_vector(goal.get("target_rad"), "controller target_rad", length=STATE_STEPS)
        validate_time_grid(goal.get("time_s"), "controller time_s")
        return _json_copy(goal)

    def multijoint_goal(
        self,
        scale: Any,
        joint_names: Any = None,
        *,
        family: str = "kinematic_teacher_scaled",
        initial_delta_deg: Any = None,
    ) -> dict[str, Any]:
        if family not in MULTIJOINT_GOAL_FAMILIES:
            raise RequestError("Unsupported left-foreleg goal family")
        selected = _selected_joint_names(joint_names)
        kinematic = family == "kinematic_teacher_scaled"
        if not kinematic and selected != list(LF_JOINT_NAMES):
            raise RequestError("Physics-generated motion families require a goal for all seven joints")
        try:
            scale_val = require_real_float(scale, "scale", positive=True)
            delta_arr = (
                validate_vector(initial_delta_deg, "initial_delta_deg", length=len(LF_JOINT_NAMES))
                if initial_delta_deg is not None
                else None
            )
        except ValueError as exc:
            raise RequestError(str(exc)) from None
        controller = self._multijoint_controller()
        reachable = None
        if kinematic:
            reference = controller.reference_target_rad()
            factors = np.asarray([scale_val if name in selected else 1.0 for name in LF_JOINT_NAMES])
            with np.errstate(over="ignore", invalid="ignore"):
                target = reference[0] + factors * (reference - reference[0])
                if delta_arr is not None:
                    target = target + np.radians(delta_arr)[None, :]
            time_s = (np.arange(STATE_STEPS) * CONTROL_TIMESTEP_S).tolist()
        else:
            try:
                goal = controller.goal_from_family(
                    family=family,
                    scale=scale_val,
                    initial_delta_deg=delta_arr,
                )
            except ValueError as exc:
                raise RequestError(str(exc)) from None
            target = np.asarray(goal["target_rad"])
            time_s = goal["time_s"]
            reachable = goal["reachable_in_profile"]
        with np.errstate(over="ignore", invalid="ignore"):
            target_deg = np.degrees(target)
        if not np.isfinite(target_deg).all():
            raise RequestError("scale produced a non-finite target")
        return {
            "scope": "lf15",
            "family": family,
            "reachable_in_profile": reachable,
            "goal_scale": scale_val,
            "joint_names": list(LF_JOINT_NAMES),
            "selected_joint_names": selected,
            "initial_joint_deg": target_deg[0].tolist(),
            "target_rad": target.tolist(),
            "target_deg": target_deg.tolist(),
            "time_s": time_s,
        }

    def _validated_arguments(self, payload: Any) -> dict[str, Any]:
        if type(payload) is not dict or (set(payload) - _ALLOWED_REQUEST_KEYS):
            raise RequestError("Request must be an object with only supported rate-control fields")
        if ("target_rad" in payload) == ("goal_scale" in payload):
            raise RequestError("Provide exactly one of target_rad or goal_scale")
        try:
            mode = payload.get("mode", "auto")
            if type(mode) is not str or mode not in {"fast", "auto", "refine", "closed_loop"}:
                raise RequestError("mode must be fast, auto, refine, or closed_loop")
            threshold = require_real_float(
                payload.get("threshold_deg", DEFAULT_THRESHOLD_DEG),
                "threshold_deg",
                positive=True,
            )
            steps = require_int(
                payload.get("correction_steps", 3),
                "correction_steps",
                minimum=0,
                maximum=20,
            )
            correction_lr = require_real_float(
                payload.get("correction_lr_hz", 0.05),
                "correction_lr_hz",
                positive=True,
            )
            correction_optimizer = payload.get("correction_optimizer", "lbfgs_linesearch")
            if correction_optimizer not in ("lbfgs_linesearch", "adam"):
                raise RequestError(
                    "correction_optimizer must be 'lbfgs_linesearch' or 'adam'"
                )
            feedback_kp = require_real_float(
                payload.get("feedback_kp", DEFAULT_FEEDBACK_KP),
                "feedback_kp",
                nonnegative=True,
            )
            feedback_kd = require_real_float(
                payload.get("feedback_kd", DEFAULT_FEEDBACK_KD),
                "feedback_kd",
                nonnegative=True,
            )
            delta = require_real_float(
                payload.get("initial_tibia_delta_deg", 0.0),
                "initial_tibia_delta_deg",
            )
            render = payload.get("render", self.ui_enabled)
            if type(render) is not bool:
                raise RequestError("render must be a boolean")
            check_directions = payload.get("check_directions", False)
            if type(check_directions) is not bool:
                raise RequestError("check_directions must be a boolean")

            if "goal_scale" in payload:
                if "time_s" in payload:
                    raise RequestError("time_s cannot override a registered goal_scale trajectory")
                goal = self.goal(payload["goal_scale"], delta)
                target = goal["target_rad"]
                times = goal["time_s"]
            else:
                target = validate_vector(payload["target_rad"], "target_rad", length=STATE_STEPS).tolist()
                times = validate_time_grid(
                    payload.get("time_s", [i * 0.002 for i in range(STATE_STEPS)]),
                    "time_s",
                ).tolist()
        except RequestError:
            raise
        except Exception as exc:
            raise RequestError(str(exc)) from None

        caps = self._single_joint_capabilities()
        if caps.get("ready") is False:
            raise RequestError("Rate controller is not ready", 503)
        if mode == "closed_loop" and "closed_loop" not in caps.get("modes", []):
            raise RequestError("closed_loop mode is not supported by the active backend", 400)

        return {
            "target_rad": target,
            "mode": mode,
            "threshold_deg": threshold,
            "correction_steps": steps,
            "correction_lr_hz": correction_lr,
            "correction_optimizer": correction_optimizer,
            "feedback_kp": feedback_kp,
            "feedback_kd": feedback_kd,
            "initial_tibia_delta_deg": delta,
            "time_s": times,
            "render": render,
            "check_directions": check_directions,
        }

    def _validated_multijoint_arguments(
        self,
        payload: dict[str, Any],
        *,
        default_family: str | None = None,
    ) -> dict[str, Any]:
        if set(payload) - _MULTIJOINT_REQUEST_KEYS:
            raise RequestError("Request contains unsupported multi-joint fields")
        if ("target_rad" in payload) == ("goal_scale" in payload):
            raise RequestError("Provide exactly one of target_rad or goal_scale")
        if "target_rad" in payload and "goal_family" in payload:
            raise RequestError("goal_family cannot be combined with explicit target_rad")
        selected = _selected_joint_names(payload.get("joint_names"))
        defaults = _MULTIJOINT_DEFAULTS
        mode = payload.get("mode", defaults["mode"])
        if mode not in _MULTIJOINT_MODES:
            raise RequestError("mode must be fast, auto, refine, or closed_loop")
        bio_mode = payload.get("bio_mode", defaults["bio_mode"])
        if bio_mode not in _MULTIJOINT_BIO_MODES:
            raise RequestError("Unsupported multi-joint bio_mode")
        if bio_mode == "premotor_connectome":
            self._require_premotor("lf15")
        render = payload.get("render", self.ui_enabled)
        if type(render) is not bool:
            raise RequestError("render must be a boolean")
        try:
            delta = payload.get("initial_qpos_delta_deg", [0.0] * len(LF_JOINT_NAMES))
            if type(delta) is not list or len(delta) != len(LF_JOINT_NAMES):
                raise RequestError("initial_qpos_delta_deg must contain seven values")
            arguments = {
                "joint_names": selected,
                "mode": mode,
                "threshold_deg": require_real_float(
                    payload.get("threshold_deg", defaults["threshold_deg"]), "threshold_deg", positive=True,
                ),
                "correction_steps": require_int(
                    payload.get("correction_steps", defaults["correction_steps"]),
                    "correction_steps", minimum=0, maximum=20,
                ),
                "feedback_kp": require_real_float(
                    payload.get("feedback_kp", defaults["feedback_kp"]), "feedback_kp", nonnegative=True,
                ),
                "feedback_kd": require_real_float(
                    payload.get("feedback_kd", defaults["feedback_kd"]), "feedback_kd", nonnegative=True,
                ),
                "initial_qpos_delta_deg": [require_real_float(item, "initial_qpos_delta_deg") for item in delta],
                "render": render,
                "bio_mode": bio_mode,
                "bio_seed": require_int(payload.get("bio_seed", 0), "bio_seed", minimum=0, maximum=2**32 - 1),
                "bio_physics_vjp_steps": require_int(
                    payload.get("bio_physics_vjp_steps", 0), "bio_physics_vjp_steps", minimum=0, maximum=100,
                ),
                "bio_population_scale": require_int(
                    payload.get("bio_population_scale", 1), "bio_population_scale", minimum=1, maximum=128,
                ),
            }
            if "goal_scale" in payload:
                if "time_s" in payload:
                    raise RequestError("time_s cannot override a registered goal_scale trajectory")
                family = payload.get("goal_family", default_family or defaults["family"])
                if type(family) is not str or family not in MULTIJOINT_GOAL_FAMILIES:
                    raise RequestError("Unsupported left-foreleg goal_family")
                target = self.multijoint_goal(
                    payload["goal_scale"], selected, family=family,
                )["target_rad"]
            else:
                raw = payload["target_rad"]
                if type(raw) is not list or any(
                    type(row) is not list or any(type(item) not in (int, float) for item in row)
                    for row in raw
                ):
                    raise RequestError("target_rad must be a matrix of finite numbers")
                target_arr, _ = validate_multijoint_goal(
                    raw, n_joints=len(LF_JOINT_NAMES), time_s=payload.get("time_s"),
                )
                target = target_arr.tolist()
        except RequestError:
            raise
        except (ValueError, TypeError, OverflowError) as exc:
            raise RequestError(str(exc)) from None
        self._multijoint_controller()
        return {**arguments, "target_rad": target}

    def _validated_bio_arguments(self, payload: Any) -> dict[str, Any]:
        if type(payload) is not dict or (set(payload) - _BIO_REQUEST_KEYS):
            raise RequestError("Request must be an object with only supported bio fields")
        bio_scope = payload.get("scope", "lf15")
        if bio_scope not in ("tibia", "lf15"):
            raise RequestError("scope must be tibia or lf15")
        bio_mode = payload.get("bio_mode", "deterministic_isi")
        if type(bio_mode) is not str or bio_mode not in _BIO_ONLY_MODES:
            raise RequestError("Unsupported bio_mode")
        if bio_mode == "premotor_connectome":
            self._require_premotor(bio_scope)
        n_pools = 15 if bio_scope == "lf15" else 2

        has_source_job = "source_job_id" in payload
        has_arrays = "command_pool_rates_hz" in payload
        has_goal = "goal_scale" in payload
        if int(has_source_job) + int(has_arrays) + int(has_goal) != 1:
            raise RequestError(
                "Provide exactly one of source_job_id, command_pool_rates_hz, or goal_scale"
            )

        tau_from_source: float | None = None
        wf_list: list[list[float]] | None = None
        exc_list: list[list[float]] | None = None
        target_list: Any = None
        if has_source_job:
            src_id = payload["source_job_id"]
            if type(src_id) is not str or not _JOB_ID.fullmatch(src_id):
                raise RequestError("Invalid source_job_id")
            with self._lock:
                src_job = self._jobs.get(src_id)
                if src_job is None:
                    raise RequestError("Unknown source_job_id", 404)
                if src_job.status != "completed":
                    raise RequestError("source_job_id has not completed yet", 409)
                declared = (src_job.result or {}).get("artifacts") or {}
                rel = _safe_relative(declared.get("physical_traces") or declared.get("traces"))
                if rel is None:
                    raise RequestError("source_job_id has no physical trace artifact", 400)
                trace_path = (src_job.output / rel).resolve()
                if not trace_path.is_relative_to(src_job.output.resolve()) or not trace_path.is_file():
                    raise RequestError("source_job_id trace artifact is missing", 404)
            with np.load(trace_path, allow_pickle=False) as tf:
                if "command_pool_rates_hz" not in tf or "target_rad" not in tf:
                    raise RequestError("source_job_id trace lacks required rate fields", 400)
                wf_arr = np.asarray(tf["command_pool_rates_hz"], dtype=np.float64)
                exc_key = "feedforward_excitation" if "feedforward_excitation" in tf else "excitation"
                if exc_key not in tf:
                    raise RequestError("source_job_id trace lacks excitation field", 400)
                exc_arr = np.asarray(tf[exc_key], dtype=np.float64)
                targ_arr = np.asarray(tf["target_rad"], dtype=np.float64)
                if "bridge_tau_ms" in tf:
                    tau_from_source = float(np.asarray(tf["bridge_tau_ms"]).reshape(()))
            if bio_scope == "tibia" and wf_arr.shape == (_RATE_STEPS, 15):
                wf_arr = wf_arr[:, (13, 14)]
                exc_arr = exc_arr[:, (13, 14)]
                if targ_arr.ndim == 2:
                    targ_arr = targ_arr[:, 6]
            if bio_scope == "lf15":
                if wf_arr.shape != (_RATE_STEPS, 15) or exc_arr.shape != (CONTROL_STEPS, 15) or targ_arr.shape != (STATE_STEPS, len(LF_JOINT_NAMES)):
                    raise RequestError("source_job_id shapes are incompatible with lf15 bio scope")
            else:
                if wf_arr.shape != (_RATE_STEPS, 2) or exc_arr.shape != (CONTROL_STEPS, 2) or targ_arr.shape != (STATE_STEPS,):
                    raise RequestError("source_job_id shapes are incompatible with tibia bio scope")
            wf_list = wf_arr.tolist()
            exc_list = exc_arr.tolist()
            target_list = targ_arr.tolist()
        elif has_arrays:
            if "target_excitation" not in payload or "target_rad" not in payload:
                raise RequestError("command_pool_rates_hz requires target_excitation and target_rad")
            try:
                wf_arr = validate_matrix(
                    payload["command_pool_rates_hz"], "command_pool_rates_hz",
                    rows=_RATE_STEPS, cols=n_pools, bounds=(0.0, 1000.0),
                )
                exc_arr = validate_matrix(
                    payload["target_excitation"], "target_excitation",
                    rows=CONTROL_STEPS, cols=n_pools, bounds=(0.0, 1.0),
                )
                if bio_scope == "lf15":
                    targ_arr = validate_matrix(
                        payload["target_rad"], "target_rad",
                        rows=STATE_STEPS, cols=len(LF_JOINT_NAMES),
                    )
                else:
                    targ_arr = validate_vector(payload["target_rad"], "target_rad", length=STATE_STEPS)
            except Exception as exc:
                raise RequestError(str(exc)) from None
            wf_list = wf_arr.tolist()
            exc_list = exc_arr.tolist()
            target_list = targ_arr.tolist()
        else:
            if bio_scope != "lf15":
                raise RequestError("tibia bio jobs require source_job_id or explicit rate arrays")
            family = payload.get("goal_family", "swing_scaled")
            if type(family) is not str or family not in MULTIJOINT_GOAL_FAMILIES:
                raise RequestError("Unsupported left-foreleg goal_family")
            target_list = self.multijoint_goal(payload["goal_scale"], family=family)["target_rad"]

        try:
            default_tau = tau_from_source if tau_from_source is not None else (20.0 if bio_scope == "tibia" else 2.0)
            tau_ms = require_real_float(
                payload.get("tau_bridge_ms", default_tau), "tau_bridge_ms", positive=True,
            )
            pop_scale = require_int(payload.get("population_scale", 1), "population_scale", minimum=1, maximum=128)
            premotor_steps = require_int(
                payload.get("premotor_opt_steps", 40), "premotor_opt_steps", minimum=0, maximum=400,
            )
            vjp_steps = require_int(payload.get("physics_vjp_steps", 0), "physics_vjp_steps", minimum=0, maximum=100)
            seed = require_int(payload.get("seed", 0), "seed", minimum=0, maximum=2**32 - 1)
        except RequestError:
            raise
        except Exception as exc:
            raise RequestError(str(exc)) from None

        if bio_scope == "lf15":
            self._multijoint_controller()
        else:
            caps = self._single_joint_capabilities()
            if caps.get("ready") is False:
                raise RequestError("Rate controller is not ready", 503)

        return {
            "bio_scope": bio_scope,
            "bio_mode": bio_mode,
            "command_pool_rates_hz": wf_list,
            "target_excitation": exc_list,
            "target_rad": target_list,
            "tau_bridge_ms": tau_ms,
            "population_scale": pop_scale,
            "premotor_opt_steps": premotor_steps,
            "physics_vjp_steps": vjp_steps,
            "seed": seed,
        }

    def _enqueue_job(self, scope: str, arguments: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._closed.is_set():
                raise RequestError("Rate-control service is shutting down", 503)
            if self._queue.full():
                raise RequestError("The physical job queue is full", 429)
            if len(self._jobs) >= MAX_RETAINED_JOBS:
                finished = [
                    key for key, job in self._jobs.items() if job.status in {"completed", "failed"}
                ]
                for key in finished[: max(1, len(self._jobs) - MAX_RETAINED_JOBS + 1)]:
                    del self._jobs[key]
            if len(self._jobs) >= MAX_RETAINED_JOBS:
                raise RequestError("Too many active rate-control jobs", 429)
            job_id = uuid.uuid4().hex
            day = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d")
            output_dir = self._resolved_output_base() / f"{day}_{job_id}"
            job = _Job(job_id, output_dir, arguments, scope)
            self._jobs[job_id] = job
            self._queue.put_nowait(job_id)
            return {"job_id": job_id, "status": "queued"}

    def submit(self, payload: Any) -> dict[str, Any]:
        scope = payload.get("scope", "tibia") if type(payload) is dict else "tibia"
        if scope == "lf15":
            arguments = self._validated_multijoint_arguments(payload)
        elif scope == "tibia":
            if type(payload) is dict and "scope" in payload:
                payload = {key: value for key, value in payload.items() if key != "scope"}
            arguments = self._validated_arguments(payload)
        else:
            raise RequestError("scope must be tibia or lf15")
        return self._enqueue_job(scope, arguments)

    def submit_multijoint(self, payload: Any) -> dict[str, Any]:
        if type(payload) is not dict:
            raise RequestError("Request must be an object with only supported multi-joint fields")
        if payload.get("scope", "lf15") != "lf15":
            raise RequestError("scope must be lf15 on /v2/multijoint/jobs")
        arguments = self._validated_multijoint_arguments(payload, default_family="swing_scaled")
        return self._enqueue_job("lf15", arguments)

    def submit_bio(self, payload: Any) -> dict[str, Any]:
        arguments = self._validated_bio_arguments(payload)
        return self._enqueue_job("bio", arguments)

    def job(self, job_id: str) -> dict[str, Any]:
        if not _JOB_ID.fullmatch(job_id):
            raise RequestError("Unknown job", 404)
        with self._lock:
            if job_id not in self._jobs:
                raise RequestError("Unknown job", 404)
            return self._jobs[job_id].public()

    def artifact_paths(self, job_id: str) -> dict[str, Path]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise RequestError("Unknown job", 404)
            declared = (job.result or {}).get("artifacts", {})
            if type(declared) is not dict:
                return {}
            found: dict[str, Path] = {}
            out_resolved = job.output.resolve()
            for value in declared.values():
                relative = _safe_relative(value)
                if relative is None:
                    continue
                resolved = (job.output / relative).resolve()
                if not resolved.is_relative_to(out_resolved) or not resolved.is_file():
                    continue
                if resolved.stat().st_size > MAX_ARTIFACT_BYTES:
                    continue
                found[relative] = resolved
            return found

    def artifact(self, job_id: str, relative: str) -> Path:
        safe_rel = _safe_relative(relative)
        if safe_rel is None:
            raise RequestError("Artifact is not available for this job", 404)
        found = self.artifact_paths(job_id)
        if safe_rel not in found:
            raise RequestError("Artifact is not available for this job", 404)
        return found[safe_rel]

    def _run_multijoint(
        self,
        job: _Job,
        target: Any,
        args: dict[str, Any],
        progress_callback: Callable[..., None],
    ) -> dict[str, Any]:
        controller = self._multijoint_controller()
        selected = args.pop("joint_names")
        result = controller.run(target, output=job.output, progress_callback=progress_callback, **args)
        artifacts = dict(result["artifacts"])
        artifacts["physical_traces"] = artifacts["traces"]
        render = dict(result.get("render") or {})
        for absolute_path_field in ("mp4_path", "webm_path", "preview_png_path"):
            render.pop(absolute_path_field, None)
        render["rendered"] = "video_mp4" in artifacts or "video_webm" in artifacts
        with np.load(job.output / artifacts["physical_traces"], allow_pickle=False) as traces:
            target_deg = np.degrees(traces["target_rad"])
            actual_deg = np.degrees(traces["qpos"][:, LF_QPOS_ADDRESSES])
            rates = traces["command_pool_rates_hz"]
            time_ms = traces["time_s"] * 1000.0
        max_error = np.max(np.abs(actual_deg - target_deg), axis=0)
        result.update(
            scope_id="lf15",
            selected_joint_names=selected,
            joint_names=list(LF_JOINT_NAMES),
            actuator_names=list(LF_ACTUATOR_NAMES),
            initial_qpos_delta_deg=args["initial_qpos_delta_deg"],
            max_abs_joint_error_deg=float(max_error.max()),
            per_joint_max_abs_error_deg=dict(zip(LF_JOINT_NAMES, max_error.tolist())),
            artifacts=artifacts,
            render=render,
        )
        if args["mode"] == "closed_loop":
            result.setdefault("feedback", {"kp": args["feedback_kp"], "kd": args["feedback_kd"]})
        write_json_atomic(job.output / artifacts["manifest"], result)
        result["trace"] = {
            "time_ms": time_ms.tolist(),
            "target_deg": target_deg.tolist(),
            "actual_deg": actual_deg.tolist(),
            "rate_time_ms": list(range(len(rates))),
            "rates_hz": rates.tolist(),
        }
        return result

    def _run_bio(
        self,
        job: _Job,
        args: dict[str, Any],
        progress_callback: Callable[..., None],
    ) -> dict[str, Any]:
        progress_callback({"stage": "bio_realize", "bio_mode": args["bio_mode"], "scope": args["bio_scope"]})
        target = np.asarray(args["target_rad"], dtype=np.float64)
        bio_scope = args["bio_scope"]
        if bio_scope == "lf15":
            ctrl = self._multijoint_controller()
            if args["command_pool_rates_hz"] is None:
                _, wf, exc = ctrl.predict_nominal(target)
            else:
                wf = np.asarray(args["command_pool_rates_hz"], dtype=np.float64)
                exc = np.asarray(args["target_excitation"], dtype=np.float64)
            return ctrl.bio_realize(
                wf,
                exc,
                target,
                bio_mode=args["bio_mode"],
                scope="lf15",
                seed=args["seed"],
                tau_bridge_ms=args["tau_bridge_ms"],
                population_scale=args["population_scale"],
                premotor_opt_steps=args["premotor_opt_steps"],
                physics_vjp_steps=args["physics_vjp_steps"],
                output=job.output,
            )
        tibia_ctrl = self.controller()
        hashes = tibia_ctrl.catalog.three_hash_record(tibia_ctrl.config.profile or "adjacent_static_parent_v1")
        profile = tibia_ctrl.config.profile or "adjacent_static_parent_v1"
        spec = SessionSpec(
            profile=profile,
            asset_manifest_sha256=tibia_ctrl.catalog.manifest_sha256,
            teacher_sha256=tibia_ctrl.catalog.teacher_sha256,
            raw_xml_sha256=tibia_ctrl.catalog.raw_xml_sha256,
            profile_rule_sha256=hashes["profile_rule_sha256"],
            cache_mode=tibia_ctrl.backend.describe().cache_mode,
        )
        sess = tibia_ctrl.backend.open_session(spec)
        try:
            tibia_ctrl.backend.reset(sess, InitialState(0.0), seed=int(args["seed"]))
            return evaluate_bio_realization(
                backend=tibia_ctrl.backend,
                session=sess,
                catalog=tibia_ctrl.catalog,
                scope="tibia",
                command_rates_hz=np.asarray(args["command_pool_rates_hz"], dtype=np.float64),
                target_excitation=np.asarray(args["target_excitation"], dtype=np.float64),
                target_rad=target,
                bio_mode=args["bio_mode"],
                seed=int(args["seed"]),
                tau_bridge_ms=float(args["tau_bridge_ms"]),
                population_scale=int(args["population_scale"]),
                premotor_opt_steps=int(args["premotor_opt_steps"]),
                physics_vjp_steps=int(args["physics_vjp_steps"]),
                output_dir=job.output,
            )
        finally:
            tibia_ctrl.backend.close_session(sess)

    def _work(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                if job_id is None:
                    return
                with self._lock:
                    job = self._jobs[job_id]
                    if self._closed.is_set():
                        job.status = "failed"
                        job.error = "Service stopped before execution"
                        job.finished_at = _utc_now()
                        continue
                    job.status = "running"
                    job.started_at = _utc_now()
                    job.progress = {"stage": "starting"}

                def progress_callback(event: Any = None, **kwargs: Any) -> None:
                    payload = {"stage": event} if isinstance(event, str) else dict(event or {})
                    payload.update(kwargs)
                    try:
                        payload = _json_copy(payload)
                    except (TypeError, ValueError):
                        payload = {"stage": "running", "message": "Non-JSON progress update"}
                    with self._lock:
                        job.progress = payload

                try:
                    args = dict(job.arguments)
                    if job.scope == "bio":
                        result = self._run_bio(job, args, progress_callback)
                    elif job.scope == "lf15":
                        target = args.pop("target_rad")
                        result = self._run_multijoint(job, target, args, progress_callback)
                    else:
                        target = args.pop("target_rad")
                        result = self.controller().run(
                            target,
                            output=job.output,
                            progress_callback=progress_callback,
                            **args,
                        )
                    if type(result) is not dict:
                        raise ValueError("EngineV2 returned non-object result")
                    result = _json_copy(result)
                    with self._lock:
                        job.result = result
                        job.status = "completed"
                        job.progress = {**job.progress, "stage": "completed"}
                        job.finished_at = _utc_now()
                except Exception as exc:
                    with self._lock:
                        job.status = "failed"
                        job.error = _public_error(exc)
                        job.progress = {**job.progress, "stage": "failed"}
                        job.finished_at = _utc_now()
            finally:
                self._queue.task_done()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        with self._lock:
            while True:
                try:
                    job_id = self._queue.get_nowait()
                except queue.Empty:
                    break
                if job_id is not None and job_id in self._jobs:
                    job = self._jobs[job_id]
                    job.status = "failed"
                    job.error = "Service stopped before execution"
                    job.finished_at = _utc_now()
                self._queue.task_done()
            self._queue.put_nowait(None)
        self._worker.join(timeout=0.5)
        with self._controller_lock:
            if self._controller_instance is not None:
                self._controller_instance.close()
            if self._multijoint_instance is not None:
                self._multijoint_instance.close()


class RateControlV2HTTPServer(ThreadingHTTPServer):
    """Multi-threaded local HTTP server hosting RateControlV2App."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], *, app: RateControlV2App) -> None:
        self.app = app
        super().__init__(address, RateControlV2Handler)

    def server_close(self) -> None:
        self.app.close()
        super().server_close()


class RateControlV2Handler(BaseHTTPRequestHandler):
    """Strict security-hardened HTTP request handler supporting /v2/* and /v1/* routes."""

    server_version = "RateControlV2"
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, *args: Any) -> None:
        pass

    @property
    def app(self) -> RateControlV2App:
        return self.server.app  # type: ignore[attr-defined]

    def _validate_host_origin(self) -> None:
        host_header = self.headers.get("Host", "")
        if host_header:
            host = _host_without_port(host_header)
            bound_host = str(self.server.server_address[0]).lower()
            allowed_hosts = set(_LOCAL_HOSTS)
            if bound_host and bound_host != "0.0.0.0":
                allowed_hosts.add(bound_host)
            if host not in allowed_hosts:
                raise RequestError("Invalid Host header", 403)
        if self.command == "POST":
            origin = self.headers.get("Origin")
            if origin is not None:
                parsed = urllib.parse.urlparse(origin)
                if _host_without_port(parsed.netloc) not in _LOCAL_HOSTS:
                    raise RequestError("Cross-origin POST rejected", 403)

    def _headers(self, status: int, mime: str, size: int, **headers: Any) -> None:
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'",
        )
        for key, value in headers.items():
            self.send_header(key.replace("_", "-"), str(value))
        self.end_headers()

    def _json(self, payload: Any, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(raw))
        if self.command != "HEAD":
            self.wfile.write(raw)

    def _file(self, path: Path, *, mime: str | None = None) -> None:
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        extra: dict[str, Any] = {"Accept_Ranges": "bytes"}
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
            if not match or not any(match.groups()) or size == 0:
                self._headers(416, "application/octet-stream", 0, Content_Range=f"bytes */{size}")
                return
            if match[1]:
                start = int(match[1])
                end = min(int(match[2]), size - 1) if match[2] else size - 1
            else:
                suffix = int(match[2])
                start = max(0, size - suffix) if suffix > 0 else size
            if start >= size or end < start:
                self._headers(416, "application/octet-stream", 0, Content_Range=f"bytes */{size}")
                return
            status = 206
            extra["Content_Range"] = f"bytes {start}-{end}/{size}"
        mime = mime or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        length = max(0, end - start + 1)
        self._headers(status, mime, length, **extra)
        if self.command != "HEAD":
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = length
                while remaining:
                    chunk = stream.read(min(65_536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    def _download(self, job_id: str) -> None:
        found = self.app.artifact_paths(job_id)
        selected = [
            (name, path)
            for name, path in sorted(found.items())
            if path.suffix.lower() in _ZIP_SUFFIXES
            and not any(
                part == "iterations" or part.startswith("iteration_")
                for part in PurePosixPath(name).parts
            )
        ]
        if not selected:
            raise RequestError("No downloadable artifacts available for this job", 404)
        import io

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for rel, path in selected:
                zf.write(path, arcname=rel)
        raw = buf.getvalue()
        self._headers(
            200,
            "application/zip",
            len(raw),
            Content_Disposition=f'attachment; filename="sensory_translator_v2_{job_id}.zip"',
        )
        if self.command != "HEAD":
            self.wfile.write(raw)

    def _read_json_body(self) -> Any:
        length_hdr = self.headers.get("Content-Length")
        if length_hdr is None:
            raise RequestError("Missing Content-Length header", 411)
        try:
            length = int(length_hdr)
        except ValueError:
            raise RequestError("Invalid Content-Length header", 400) from None
        if length < 0 or length > MAX_BODY_BYTES:
            raise RequestError(f"Request body exceeds limit ({MAX_BODY_BYTES} bytes)", 413)
        raw_bytes = self.rfile.read(length)
        if len(raw_bytes) != length:
            raise RequestError("Incomplete request body", 400)
        try:
            return _strict_json_loads(raw_bytes.decode("utf-8"))
        except RequestError:
            raise
        except Exception as exc:
            raise RequestError(f"Malformed JSON request: {exc}", 400) from None

    def _dispatch_get(self) -> None:
        self._validate_host_origin()
        parsed = urllib.parse.urlsplit(self.path)
        path = _normalize_route_path(parsed.path)
        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=False)

        if self.app.ui_enabled and path in _UI_ASSETS:
            asset, mime = self.app.ui_asset(path)
            self._file(asset, mime=mime)
            return
        if path == "/":
            self._json(
                {
                    "engine": "sensory-engine-v2",
                    "api_version": "v2",
                    "version": __version__,
                    "ui_enabled": self.app.ui_enabled,
                    "controller_initialized": self.app._controller_instance is not None,
                    "links": {
                        "health": "/health",
                        "capabilities": "/v2/capabilities",
                        "goal": "/v2/goal",
                        "jobs": "/v2/jobs",
                        "multijoint_goal": "/v2/multijoint/goal",
                        "multijoint_jobs": "/v2/multijoint/jobs",
                        "bio_jobs": "/v2/bio/jobs",
                        "openapi": "/openapi.json",
                    },
                }
            )
            return
        if path == "/health":
            self._json(
                {
                    "ok": True,
                    "status": "ok",
                    "version": "v2",
                    "engine": "sensory-engine-v2",
                    "controller_initialized": self.app._controller_instance is not None,
                }
            )
            return
        if path == "/openapi.json":
            host = self.headers.get("Host") or f"127.0.0.1:{self.server.server_address[1]}"
            self._json(build_openapi_v2_spec(base_url=f"http://{host}"))
            return
        if path in ("/v2/capabilities", "/v1/capabilities"):
            self._json(self.app.capabilities())
            return
        if path == "/v2/assets":
            self._json(self.app.assets())
            return
        if path in ("/v2/goal", "/v1/goal"):
            scale_vals = query.get("scale")
            if not scale_vals or len(scale_vals) != 1:
                raise RequestError("Provide single numeric query parameter scale", 400)
            delta_vals = query.get("initial_delta_deg", ["0"])
            if len(delta_vals) != 1:
                raise RequestError("Provide at most one initial_delta_deg parameter", 400)
            try:
                scale_num = float(scale_vals[0])
                delta_num = float(delta_vals[0])
            except ValueError:
                raise RequestError("scale and initial_delta_deg must be numeric", 400) from None
            scope = query.get("scope", ["tibia"])
            if scope == ["lf15"]:
                joints = query.get("joints")
                if joints is not None and len(joints) != 1:
                    raise RequestError("Provide at most one joints parameter")
                families = query.get("family", [_MULTIJOINT_DEFAULTS["family"]])
                if len(families) != 1:
                    raise RequestError("Provide at most one family parameter")
                self._json(self.app.multijoint_goal(
                    scale_num, joints[0].split(",") if joints else None, family=families[0],
                ))
            elif scope == ["tibia"]:
                self._json(self.app.goal(scale_num, delta_num))
            else:
                raise RequestError("scope must be tibia or lf15")
            return
        if path == "/v2/multijoint/goal":
            scale_vals = query.get("scale", ["0.92"])
            if len(scale_vals) != 1:
                raise RequestError("Provide single numeric query parameter scale", 400)
            families = query.get("family", ["swing_scaled"])
            if len(families) != 1:
                raise RequestError("Provide at most one family parameter")
            joints = query.get("joints")
            if joints is not None and len(joints) != 1:
                raise RequestError("Provide at most one joints parameter")
            init_deltas = query.get("initial_delta_deg")
            if init_deltas is not None and len(init_deltas) != 1:
                raise RequestError("Provide at most one initial_delta_deg parameter")
            try:
                scale_num = float(scale_vals[0])
                delta_list = (
                    [float(item) for item in init_deltas[0].split(",")]
                    if init_deltas is not None
                    else None
                )
            except ValueError:
                raise RequestError("scale and initial_delta_deg must be numeric", 400) from None
            self._json(
                self.app.multijoint_goal(
                    scale_num,
                    joints[0].split(",") if joints else None,
                    family=families[0],
                    initial_delta_deg=delta_list,
                )
            )
            return

        m_download = re.fullmatch(r"/(?:v1|v2)/jobs/([0-9a-f]{32})/download", path)
        if m_download:
            self._download(m_download.group(1))
            return

        m_art = re.fullmatch(r"/(?:v1|v2)/jobs/([0-9a-f]{32})/artifacts/(.+)", path)
        if m_art:
            rel = urllib.parse.unquote(m_art.group(2))
            self._file(self.app.artifact(m_art.group(1), rel))
            return

        m_job = re.fullmatch(r"/(?:v1|v2)/jobs/([0-9a-f]{32})", path)
        if m_job:
            self._json(self.app.job(m_job.group(1)))
            return

        raise RequestError("Endpoint not found", 404)

    def do_GET(self) -> None:
        try:
            self._dispatch_get()
        except RequestError as exc:
            self._json({"error": str(exc)}, status=exc.status)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            self._json({"error": _public_error(exc)}, status=500)

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        try:
            self._validate_host_origin()
            path = _normalize_route_path(urllib.parse.urlsplit(self.path).path)
            if path in ("/v2/jobs", "/v1/jobs"):
                payload = self._read_json_body()
                accepted = self.app.submit(payload)
                self._json(accepted, status=202)
                return
            if path == "/v2/multijoint/jobs":
                payload = self._read_json_body()
                accepted = self.app.submit_multijoint(payload)
                self._json(accepted, status=202)
                return
            if path == "/v2/bio/jobs":
                payload = self._read_json_body()
                accepted = self.app.submit_bio(payload)
                self._json(accepted, status=202)
                return
            raise RequestError("Endpoint not found", 404)
        except RequestError as exc:
            self._json({"error": str(exc)}, status=exc.status)
        except Exception as exc:
            self._json({"error": _public_error(exc)}, status=500)
