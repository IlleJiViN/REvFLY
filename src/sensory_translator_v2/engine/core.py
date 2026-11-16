"""Lazy, thread-safe Engine v2 rate-control orchestrator."""

from __future__ import annotations

import hashlib
import math
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ..assets import AssetCatalog, ModelEntry, load_asset_catalog, verify_asset_catalog
from ..backends.base import FakeBackend
from ..backends.v1_subprocess import V1SubprocessBackend
from ..contracts import (
    Backend,
    BackendUnavailableError,
    CONTROL_STEPS,
    CONTROL_TIMESTEP_S,
    ClosedLoopCapableBackend,
    ClosedLoopRequest,
    DEFAULT_DIRECTION_EPS,
    DEFAULT_DIRECTION_SEED,
    DEFAULT_FEEDBACK_KD,
    DEFAULT_FEEDBACK_KP,
    DEFAULT_JACOBIAN_EPS,
    EXT_TO_FLEX_RATIO,
    Evaluation,
    EvaluationRequest,
    InitialState,
    RenderCapableBackend,
    RequestValidationError,
    STATE_STEPS,
    Session,
    SessionSpec,
    TARGET_ACTUATOR_COLUMNS,
    TARGET_ACTUATOR_NAMES,
    TARGET_JOINT,
    TARGET_QPOS_ADDRESS,
    TRACE_KEYS,
    require_int,
    require_profile,
    require_real_float,
)
from ..evidence import utc_now_iso, write_json_atomic, write_npz_atomic, write_run_artifacts
from ..goals import encode_joint_condition, goal_from_scale, validate_joint_goal
from ..rate import (
    ConditionalRateFFN,
    DEFAULT_MAX_RATE_HZ,
    MotorRateProtocol,
    direct_pool_rate_waveform,
    load_rate_checkpoint,
    rate_to_excitation,
)
from ..rate.optim import lbfgs_direction as _lbfgs_direction

DEFAULT_THRESHOLD_DEG = math.degrees(0.1)  # 5.729577951308233 deg


def _load_teacher_arrays(teacher_path: Path) -> dict[str, np.ndarray]:
    with np.load(teacher_path, allow_pickle=False) as archive:
        required = ("qpos", "qvel")
        for key in required:
            if key not in archive.files:
                raise RequestValidationError(f"Teacher NPZ missing required key {key!r}")
        return {
            key: np.asarray(archive[key], dtype=np.float64)
            for key in archive.files
            if np.issubdtype(archive[key].dtype, np.number)
        }


def _persist_partial_evidence(
    out_dir: Path,
    *,
    profile: str,
    model_id: str,
    mode: str,
    initializer: str,
    target_arr: np.ndarray,
    delta_rad: float,
    history: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    exc: Exception,
    elapsed_s: float,
) -> str:
    """Persist structured partial failure evidence without claiming physics certification."""
    artifacts: dict[str, str] = {"partial_evidence": "partial_evidence.json"}
    best_partial_summary: dict[str, Any] | None = None
    if candidates:
        best_partial = min(candidates, key=lambda item: item["mse_rad2"])
        ev: Evaluation = best_partial["eval"]
        traces_npz = {key: ev.traces[key] for key in TRACE_KEYS}
        traces_npz["target_rad"] = target_arr
        traces_npz["actual_rad"] = ev.joint_rad
        traces_npz["command_pool_rates_hz"] = best_partial["waveform_hz"]
        traces_npz["rate_knots_hz"] = best_partial["knots_hz"]
        traces_npz["excitation"] = best_partial["controls"]
        write_npz_atomic(out_dir / "partial_traces.npz", traces_npz)
        artifacts["partial_traces_npz"] = "partial_traces.npz"
        best_partial_summary = {
            "step": int(best_partial["step"]),
            "mse_rad2": float(best_partial["mse_rad2"]),
            "rmse_deg": float(best_partial["rmse_deg"]),
            "nominal_rmse_deg": float(candidates[0]["rmse_deg"]),
            "rate_knots_hz": best_partial["knots_hz"].tolist(),
        }

    payload = {
        "schema_version": "sensory_translator_v2_partial_evidence_v1",
        "status": "failed",
        "partial": True,
        "created_utc": utc_now_iso(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "profile": profile,
        "model_id": model_id,
        "mode": mode,
        "initializer": initializer,
        "initial_tibia_delta_rad": float(delta_rad),
        "target_rad": target_arr.tolist(),
        "elapsed_s": float(elapsed_s),
        "candidates_evaluated": len(candidates),
        "is_certified_frozen_physics": False,
        "behavior_simulation_evaluated": False,
        "independent_replay_verified": False,
        "history": history,
        "best_partial_candidate": best_partial_summary,
        "artifacts": artifacts,
    }
    write_json_atomic(out_dir / "partial_evidence.json", payload)
    return "partial_evidence.json"


def _mse_and_rmse_deg(actual_rad: np.ndarray, target_rad: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(actual_rad, dtype=np.float64) - np.asarray(target_rad, dtype=np.float64)
    mse = float(np.mean(diff ** 2))
    rmse_deg = float(np.degrees(np.sqrt(mse)))
    return mse, rmse_deg


@dataclass(frozen=True)
class EngineV2Config:
    """Configuration for EngineV2 with explicit paths, backend selection, and profile."""

    backend: str = "persistent_cached"
    profile: str | None = None
    model_id: str | None = None
    checkpoint_path: Path | None = None
    asset_root: str | Path | None = None
    runtime_root: str | Path | None = None
    physics_python: str | Path | None = None
    scripts_root: str | Path | None = None
    timeout_s: float = 90.0
    direction_seed: int = DEFAULT_DIRECTION_SEED
    jacobian_eps: float = DEFAULT_JACOBIAN_EPS
    direction_eps: float = DEFAULT_DIRECTION_EPS
    verify_assets_on_start: bool = True
    backend_factory: Callable[[AssetCatalog, str], Backend] | None = None

    def __post_init__(self) -> None:
        allowed_backends = ("v1_subprocess", "persistent_fresh", "persistent_cached", "fake")
        if self.backend not in allowed_backends and self.backend_factory is None:
            raise RequestValidationError(
                f"backend must be one of {allowed_backends!r}, got {self.backend!r}"
            )
        if self.profile is not None:
            require_profile(self.profile)
        require_real_float(self.timeout_s, "timeout_s", positive=True)
        require_int(self.direction_seed, "direction_seed", minimum=0)
        require_real_float(self.jacobian_eps, "jacobian_eps", positive=True)
        require_real_float(self.direction_eps, "direction_eps", positive=True)


class EngineV2:
    """Thread-safe, stateless-session v2 rate control & MuJoCo verification engine."""

    def __init__(self, config: EngineV2Config | None = None) -> None:
        self.config = config or EngineV2Config()
        self._init_lock = threading.Lock()
        self._serial_lock = threading.RLock()
        self._closed = False
        self._catalog: AssetCatalog | None = None
        self._model_entry: ModelEntry | None = None
        self._profile: str | None = None
        self._rate_model: ConditionalRateFFN | None = None
        self._rate_model_config: dict[str, Any] = {}
        self._teacher_q: np.ndarray | None = None
        self._teacher_qvel0: float = 0.0
        self._backend: Backend | None = None
        self._init_error: str | None = None

    def _ensure_initialized(self) -> None:
        if (
            self._catalog is not None
            and self._rate_model is not None
            and self._backend is not None
        ):
            return
        with self._init_lock:
            if self._closed:
                raise RequestValidationError("EngineV2 is closed")
            if (
                self._catalog is not None
                and self._rate_model is not None
                and self._backend is not None
            ):
                return
            try:
                catalog = load_asset_catalog(
                    asset_root=self.config.asset_root,
                    runtime_root=self.config.runtime_root,
                    physics_python=self.config.physics_python,
                    scripts_root=self.config.scripts_root,
                    verify_hashes=self.config.verify_assets_on_start,
                )
                requested_profile = self.config.profile or "adjacent_static_parent_v1"
                if self.config.checkpoint_path is not None:
                    ckpt_path = Path(self.config.checkpoint_path).expanduser().resolve()
                    ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
                    model_entry = ModelEntry(
                        model_id=ckpt_path.stem,
                        profile=requested_profile,
                        checkpoint_path=ckpt_path,
                        checkpoint_sha256=ckpt_sha,
                        description="Custom user checkpoint",
                    )
                else:
                    if self.config.model_id is not None:
                        model_entry = catalog.resolve_model(
                            self.config.model_id, profile=requested_profile
                        )
                        requested_profile = self.config.profile or model_entry.profile
                    else:
                        model_entry = catalog.default_model_for_profile(requested_profile)

                rate_model, rate_cfg, _ = load_rate_checkpoint(
                    model_entry.checkpoint_path,
                    expected_sha256=model_entry.checkpoint_sha256,
                )
                teacher = _load_teacher_arrays(catalog.teacher_path)
                teacher_q = np.asarray(teacher["qpos"][:, TARGET_QPOS_ADDRESS], dtype=np.float64)
                teacher_v0 = float(teacher["qvel"][0, TARGET_QPOS_ADDRESS])

                if self.config.backend_factory is not None:
                    backend = self.config.backend_factory(catalog, requested_profile)
                elif self.config.backend == "fake":
                    backend = FakeBackend(profile=requested_profile, teacher_joint_rad=teacher_q)
                elif self.config.backend == "v1_subprocess":
                    backend = V1SubprocessBackend(
                        catalog,
                        profile=requested_profile,
                        timeout_s=self.config.timeout_s,
                    )
                elif self.config.backend in ("persistent_fresh", "persistent_cached"):
                    from ..backends.persistent_mujoco import PersistentMujocoBackend

                    cache_mode = (
                        "cached_model"
                        if self.config.backend == "persistent_cached"
                        else "fresh"
                    )
                    backend = PersistentMujocoBackend(
                        catalog,
                        profile=requested_profile,
                        cache_mode=cache_mode,
                        timeout_s=self.config.timeout_s,
                    )
                else:  # pragma: no cover - validated in config
                    raise BackendUnavailableError(f"Unsupported backend {self.config.backend!r}")

                self._catalog = catalog
                self._model_entry = model_entry
                self._profile = requested_profile
                self._rate_model = rate_model
                self._rate_model_config = rate_cfg
                self._teacher_q = teacher_q
                self._teacher_qvel0 = teacher_v0
                self._backend = backend
                self._init_error = None
            except Exception as exc:
                self._init_error = f"{type(exc).__name__}: {exc}"
                raise

    @property
    def catalog(self) -> AssetCatalog:
        self._ensure_initialized()
        assert self._catalog is not None
        return self._catalog

    @property
    def backend(self) -> Backend:
        self._ensure_initialized()
        assert self._backend is not None
        return self._backend

    @property
    def profile(self) -> str:
        self._ensure_initialized()
        assert self._profile is not None
        return self._profile

    def verify_assets(self) -> dict[str, Any]:
        """Re-verify every asset hash on disk in the loaded asset catalog."""
        return verify_asset_catalog(self.catalog)

    def capabilities(self) -> dict[str, Any]:
        """Return machine-readable v2 capabilities, limits, and active backend identity."""
        try:
            self._ensure_initialized()
            ready = not self._closed
            err = "EngineV2 is closed" if self._closed else None
        except Exception as exc:
            return {
                "schema_version": "sensory_translator_v2_capabilities_v1",
                "ready": False,
                "error": f"{type(exc).__name__}: {exc}",
                "modes": ["fast", "auto", "refine"],
                "backend": {
                    "name": self.config.backend,
                    "is_certified_frozen_physics": False,
                },
            }

        assert self._catalog is not None
        assert self._model_entry is not None
        assert self._rate_model is not None
        assert self._profile is not None
        backend_identity = self.backend.describe().to_dict()
        supports_cl = isinstance(self.backend, ClosedLoopCapableBackend)
        max_hz, tau_ms, rate_scale_hz, clip_mode = self._bridge_params()
        param_count = sum(p.numel() for p in self._rate_model.parameters())
        return {
            "schema_version": "sensory_translator_v2_capabilities_v1",
            "ready": ready,
            "error": err,
            "modes": (
                ["fast", "auto", "refine", "closed_loop"]
                if supports_cl
                else ["fast", "auto", "refine"]
            ),
            "correction_optimizers": ["lbfgs_linesearch", "adam"],
            "defaults": {
                "backend": self.config.backend,
                "profile": self._profile,
                "model_id": self._model_entry.model_id,
                "goal_scale": 0.975,
                "threshold_deg": DEFAULT_THRESHOLD_DEG,
                "correction_steps": 3,
                "correction_lr_hz": 0.05,
                "correction_optimizer": "lbfgs_linesearch",
                "feedback_kp": DEFAULT_FEEDBACK_KP,
                "feedback_kd": DEFAULT_FEEDBACK_KD,
                "ext_to_flex_ratio": EXT_TO_FLEX_RATIO,
                "initial_tibia_delta_deg": 0.0,
            },
            "goal": {
                "joint": TARGET_JOINT,
                "joint_label": "LF tibia",
                "duration_ms": 420,
                "samples": STATE_STEPS,
                "dt_ms": 2,
            },
            "rate": {
                "unit": "Hz",
                "pools": 2,
                "knots_per_pool": self._rate_model.knots_per_pool,
                "max_hz": max_hz,
                "tau_ms": tau_ms,
                "rate_scale_hz": rate_scale_hz,
                "clip_boundary": clip_mode,
            },
            "scope": {
                "controlled_muscles": 2,
                "teacher_assisted_muscles": 13,
                "fixed_torso": True,
            },
            "model": {
                "model_id": self._model_entry.model_id,
                "profile": self._model_entry.profile,
                "architecture": f"MLP{self._rate_model.condition_dim}-{self._rate_model.hidden_dim}-{2 * self._rate_model.knots_per_pool}",
                "checkpoint_sha256": self._model_entry.checkpoint_sha256,
                "parameters": param_count,
            },
            "available_models": {
                mid: entry.to_dict() for mid, entry in sorted(self._catalog.all_models.items())
            },
            "asset_hashes": self._catalog.three_hash_record(self._profile),
            "backend": backend_identity,
            "limits": [
                "420 ms left-foreleg tibia trajectory; two pools; fixed torso; 13 teacher-assisted muscles.",
                "Continuous commanded rates in Hz, not electrode current or realized spikes.",
                (
                    "Open-loop modes generate whole-waveform rate plans without online feedback; closed_loop mode adds causal 2ms proprioceptive PD feedback."
                    if supports_cl
                    else "Whole-waveform generation from the goal and initial state; no online feedback."
                ),
                "Model validated on one motion family; arbitrary numerical targets may not be reachable.",
            ],
        }

    def goal_from_scale(
        self,
        scale: float,
        *,
        initial_tibia_delta_deg: float = 0.0,
    ) -> dict[str, Any]:
        """Build a registered-family trajectory scaled from the frozen teacher pose."""
        self._ensure_initialized()
        assert self._teacher_q is not None
        return goal_from_scale(
            scale,
            self._teacher_q,
            initial_tibia_delta_deg=initial_tibia_delta_deg,
        )

    def _bridge_params(self) -> tuple[float, float, float, str]:
        assert self._rate_model is not None
        max_hz = float(self._rate_model.max_rate_hz)
        bridge_cfg = (self._rate_model_config or {}).get("bridge", {})
        if not isinstance(bridge_cfg, dict):
            bridge_cfg = {}
        tau_ms = float(bridge_cfg.get("tau_ms", 20.0))
        rate_scale_hz = float(bridge_cfg.get("rate_scale_hz", 100.0))
        clip_mode = "native"
        if self._backend is not None:
            details = self._backend.describe().details or {}
            if bool(details.get("supports_one_sided_boundary_gradient", False)):
                clip_mode = "one_sided"
        return max_hz, tau_ms, rate_scale_hz, clip_mode

    def predict_nominal(
        self,
        target_rad: Any,
        *,
        initial_tibia_delta_rad: float = 0.0,
        initializer: str = "mlp",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict nominal rate knots [2, K], pool rates [420, 2], and excitation [210, 2]."""
        self._ensure_initialized()
        assert self._rate_model is not None and self._teacher_q is not None
        target_arr, _ = validate_joint_goal(target_rad)
        delta_rad = require_real_float(initial_tibia_delta_rad, "initial_tibia_delta_rad")
        q0 = float(self._teacher_q[0]) + delta_rad
        max_hz, tau_ms, rate_scale_hz, clip_mode = self._bridge_params()
        with torch.no_grad():
            if initializer == "mlp":
                cond = encode_joint_condition(
                    torch.from_numpy(target_arr).unsqueeze(0),
                    torch.tensor([q0], dtype=torch.float64),
                    torch.tensor([self._teacher_qvel0], dtype=torch.float64),
                )
                knots = self._rate_model(cond).clamp(0.0, max_hz)
            elif initializer == "fixed_nominal":
                knots = (
                    self._rate_model.base_knots_hz.unsqueeze(0)
                    .clone()
                    .clamp(0.0, max_hz)
                )
            elif initializer == "constant50":
                knots = torch.full(
                    (1, 2, self._rate_model.knots_per_pool),
                    50.0,
                    dtype=torch.float64,
                ).clamp(0.0, max_hz)
            else:
                raise RequestValidationError(
                    f"initializer must be 'mlp', 'fixed_nominal', or 'constant50', got {initializer!r}"
                )
            waveform = direct_pool_rate_waveform(knots, max_rate_hz=max_hz)
            excitation = rate_to_excitation(
                waveform,
                max_rate_hz=max_hz,
                rate_scale_hz=rate_scale_hz,
                tau_ms=tau_ms,
                clip_boundary_gradient=clip_mode,
            )
        return (
            knots[0].detach().cpu().numpy().copy(),
            waveform[0].detach().cpu().numpy().copy(),
            excitation[0].detach().cpu().numpy().copy(),
        )

    def _knots_to_controls(self, knots_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        max_hz, tau_ms, rate_scale_hz, clip_mode = self._bridge_params()
        with torch.no_grad():
            clamped = knots_tensor.detach().clamp(0.0, max_hz)
            waveform = direct_pool_rate_waveform(clamped, max_rate_hz=max_hz)
            exc = rate_to_excitation(
                waveform,
                max_rate_hz=max_hz,
                rate_scale_hz=rate_scale_hz,
                tau_ms=tau_ms,
                clip_boundary_gradient=clip_mode,
            )
        return (
            waveform[0].cpu().numpy().astype(np.float64),
            exc[0].cpu().numpy().astype(np.float64),
        )

    def _knots_vjp_step(
        self,
        knots_tensor: torch.Tensor,
        control_vjp: np.ndarray,
    ) -> torch.Tensor:
        max_hz, tau_ms, rate_scale_hz, clip_mode = self._bridge_params()
        probe = knots_tensor.detach().clone().requires_grad_(True)
        waveform = direct_pool_rate_waveform(probe, max_rate_hz=max_hz)
        exc = rate_to_excitation(
            waveform,
            max_rate_hz=max_hz,
            rate_scale_hz=rate_scale_hz,
            tau_ms=tau_ms,
            clip_boundary_gradient=clip_mode,
        )
        exc.backward(torch.from_numpy(control_vjp[None, :, :].astype(np.float64)))
        assert probe.grad is not None
        return probe.grad.detach().clone()

    def run(
        self,
        target_rad: Any,
        *,
        output: str | Path | None = None,
        mode: str = "auto",
        initializer: str = "mlp",
        threshold_deg: float = DEFAULT_THRESHOLD_DEG,
        correction_steps: int = 3,
        correction_lr_hz: float = 0.05,
        correction_optimizer: str = "lbfgs_linesearch",
        feedback_kp: float = DEFAULT_FEEDBACK_KP,
        feedback_kd: float = DEFAULT_FEEDBACK_KD,
        initial_tibia_delta_deg: float | None = None,
        initial_tibia_delta_rad: float | None = None,
        time_s: Any = None,
        render: bool = False,
        check_directions: bool = False,
        progress_callback: Callable[..., None] | None = None,
    ) -> dict[str, Any]:
        """Run isolated rate inference, MuJoCo physics verification, and optional VJP/closed-loop refinement."""
        if mode not in ("fast", "auto", "refine", "closed_loop"):
            raise RequestValidationError("mode must be fast, auto, refine, or closed_loop")
        if correction_optimizer not in ("lbfgs_linesearch", "adam"):
            raise RequestValidationError(
                f"correction_optimizer must be 'lbfgs_linesearch' or 'adam', got {correction_optimizer!r}"
            )
        threshold = require_real_float(threshold_deg, "threshold_deg", positive=True)
        steps = require_int(correction_steps, "correction_steps", minimum=0, maximum=20)
        lr_hz = require_real_float(correction_lr_hz, "correction_lr_hz", positive=True)
        kp = require_real_float(feedback_kp, "feedback_kp", nonnegative=True)
        kd = require_real_float(feedback_kd, "feedback_kd", nonnegative=True)
        if type(render) is not bool:
            raise RequestValidationError("render must be boolean")
        if type(check_directions) is not bool:
            raise RequestValidationError("check_directions must be boolean")
        if initial_tibia_delta_deg is not None and initial_tibia_delta_rad is not None:
            raise RequestValidationError("Specify at most one of initial_tibia_delta_deg or initial_tibia_delta_rad")
        if initial_tibia_delta_rad is not None:
            delta_rad = require_real_float(initial_tibia_delta_rad, "initial_tibia_delta_rad")
        elif initial_tibia_delta_deg is not None:
            delta_rad = math.radians(require_real_float(initial_tibia_delta_deg, "initial_tibia_delta_deg"))
        else:
            delta_rad = 0.0

        target_arr, time_arr = validate_joint_goal(target_rad, time_s=time_s)
        run_start = time.perf_counter()

        def _notify(stage: str, **info: Any) -> None:
            if progress_callback is not None:
                progress_callback({"stage": stage, **info})

        with self._serial_lock:
            self._ensure_initialized()
            assert self._catalog is not None and self._model_entry is not None
            assert self._profile is not None and self._teacher_q is not None
            if mode == "closed_loop" and not isinstance(self.backend, ClosedLoopCapableBackend):
                raise RequestValidationError(
                    f"Backend {self.backend.describe().name!r} does not support closed_loop mode"
                )

            out_dir: Path | None = None
            if output is not None:
                out_dir = Path(output).expanduser().resolve()
                if out_dir.exists():
                    raise RequestValidationError(f"Refuse to overwrite existing output directory: {out_dir}")
                out_dir.mkdir(parents=True, exist_ok=False)

            spec = SessionSpec(
                profile=self._profile,
                asset_manifest_sha256=self._catalog.manifest_sha256,
                teacher_sha256=self._catalog.teacher_sha256,
                raw_xml_sha256=self._catalog.raw_xml_sha256,
                profile_rule_sha256=self._catalog.three_hash_record(self._profile)["profile_rule_sha256"],
                direction_seed=self.config.direction_seed,
                jacobian_eps=self.config.jacobian_eps,
                direction_eps=self.config.direction_eps,
                cache_mode=self.backend.describe().cache_mode,
            )

            _notify("nominal_prediction")
            mlp_start = time.perf_counter()
            nominal_knots_np, nominal_waveform_np, nominal_exc_np = self.predict_nominal(
                target_arr,
                initial_tibia_delta_rad=delta_rad,
                initializer=initializer,
            )
            mlp_s = time.perf_counter() - mlp_start

            max_hz, tau_ms, rate_scale_hz, clip_mode = self._bridge_params()
            knots_param = torch.tensor(
                nominal_knots_np[None, :, :].copy(),
                dtype=torch.float64,
                requires_grad=True,
            )
            optimizer = torch.optim.Adam([knots_param], lr=lr_hz, eps=1e-10)
            initial_state = InitialState(initial_tibia_delta_rad=delta_rad)

            history: list[dict[str, Any]] = []
            candidates: list[dict[str, Any]] = []
            physics_forward_s = 0.0
            physics_vjp_s = 0.0

            session: Session | None = None
            session_succeeded = False
            try:
                session = self.backend.open_session(spec)
                max_updates = 0 if mode == "fast" else steps

                def _append_candidate(
                    *,
                    step_num: int,
                    knots_np: np.ndarray,
                    waveform_np: np.ndarray,
                    exc_np: np.ndarray,
                    ev: Evaluation,
                    mse: float,
                    rmse_deg: float,
                    used_jacobian: bool,
                    gradient_valid: bool | None,
                    eval_time_s: float,
                    extra_history: dict[str, Any] | None = None,
                ) -> bool:
                    met = bool(rmse_deg <= threshold)
                    record: dict[str, Any] = {
                        "step": step_num,
                        "mse_rad2": mse,
                        "rmse_deg": rmse_deg,
                        "threshold_met": met,
                        "used_jacobian": used_jacobian,
                        "gradient_valid": gradient_valid,
                        "eval_time_s": eval_time_s,
                    }
                    if extra_history:
                        record.update(extra_history)
                    history.append(record)
                    candidates.append({
                        "step": step_num,
                        "knots_hz": knots_np.copy(),
                        "waveform_hz": waveform_np,
                        "controls": exc_np,
                        "eval": ev,
                        "mse_rad2": mse,
                        "rmse_deg": rmse_deg,
                    })
                    return met

                if mode == "closed_loop":
                    assert isinstance(self.backend, ClosedLoopCapableBackend)
                    with torch.no_grad():
                        knots_param.clamp_(0.0, max_hz)
                    waveform0, ff_exc0 = self._knots_to_controls(knots_param)
                    _notify("evaluating", step=0, need_jacobian=False, closed_loop=True)
                    t_cl0 = time.perf_counter()
                    self.backend.reset(session, initial_state, self.config.direction_seed)
                    ev_cl = self.backend.evaluate_closed_loop(
                        session,
                        ClosedLoopRequest(
                            feedforward_controls=ff_exc0,
                            target_rad=target_arr,
                            kp=kp,
                            kd=kd,
                            ext_to_flex_ratio=EXT_TO_FLEX_RATIO,
                            request_id="closed_loop_0",
                        ),
                    )
                    dt_cl = time.perf_counter() - t_cl0
                    physics_forward_s += dt_cl
                    realized_exc = ev_cl.traces["interval_applied_ctrl"][:, TARGET_ACTUATOR_COLUMNS].copy()
                    mse_cl, rmse_cl = _mse_and_rmse_deg(ev_cl.joint_rad, target_arr)
                    _append_candidate(
                        step_num=0,
                        knots_np=knots_param[0].detach().cpu().numpy(),
                        waveform_np=waveform0,
                        exc_np=realized_exc,
                        ev=ev_cl,
                        mse=mse_cl,
                        rmse_deg=rmse_cl,
                        used_jacobian=False,
                        gradient_valid=None,
                        eval_time_s=dt_cl,
                        extra_history={
                            "closed_loop": True,
                            "feedback_kp": kp,
                            "feedback_kd": kd,
                        },
                    )
                elif max_updates > 0 and correction_optimizer == "lbfgs_linesearch":
                    # Step 0 evaluation before any update (preserve refine eager-VJP + fail-after-eval-1 contracts)
                    with torch.no_grad():
                        knots_param.clamp_(0.0, max_hz)
                    waveform0, exc0 = self._knots_to_controls(knots_param)
                    step0_vjp = mode == "refine"
                    _notify("evaluating", step=0, need_jacobian=step0_vjp)
                    t_eval0 = time.perf_counter()
                    self.backend.reset(session, initial_state, self.config.direction_seed)
                    ev0 = self.backend.evaluate(
                        session,
                        EvaluationRequest(
                            controls=exc0,
                            need_jacobian=step0_vjp,
                            mse_target_rad=target_arr if step0_vjp else None,
                            check_directions=bool(check_directions and step0_vjp),
                            direction_seed=self.config.direction_seed,
                            clip_boundary_gradient=clip_mode,
                            request_id="step_0" if step0_vjp else "step_0_fwd",
                        ),
                    )
                    dt_eval0 = time.perf_counter() - t_eval0
                    if step0_vjp:
                        physics_vjp_s += dt_eval0
                    else:
                        physics_forward_s += dt_eval0
                    mse0, rmse0 = _mse_and_rmse_deg(ev0.joint_rad, target_arr)
                    met0 = _append_candidate(
                        step_num=0,
                        knots_np=knots_param[0].detach().cpu().numpy(),
                        waveform_np=waveform0,
                        exc_np=exc0,
                        ev=ev0,
                        mse=mse0,
                        rmse_deg=rmse0,
                        used_jacobian=step0_vjp,
                        gradient_valid=ev0.gradient_valid if step0_vjp else None,
                        eval_time_s=dt_eval0,
                    )

                    if not (mode == "auto" and met0) and not (step0_vjp and (ev0.control_vjp is None or ev0.gradient_valid is False)):
                        s_hist: list[np.ndarray] = []
                        y_hist: list[np.ndarray] = []
                        prev_x_vec: np.ndarray | None = None
                        prev_g_vec: np.ndarray | None = None
                        history_m = 5

                        def _run_line_search(
                            x_base_np: np.ndarray,
                            mse_base: float,
                            direction_np: np.ndarray,
                            alphas: tuple[float, ...],
                            upd_idx: int,
                            trial_tag: str,
                        ) -> tuple[dict[str, Any] | None, int]:
                            best_trial: dict[str, Any] | None = None
                            tried = 0
                            seen_controls: list[np.ndarray] = [x_base_np]
                            for j, alpha in enumerate(alphas):
                                cand_knots_np = np.clip(x_base_np + alpha * direction_np, 0.0, max_hz)
                                if any(np.allclose(cand_knots_np, prev, atol=1e-12, rtol=0.0) for prev in seen_controls):
                                    continue
                                seen_controls.append(cand_knots_np)
                                tried += 1
                                wf_t, exc_t = self._knots_to_controls(
                                    torch.from_numpy(cand_knots_np[None, :, :])
                                )
                                t_ls0 = time.perf_counter()
                                self.backend.reset(
                                    session,
                                    initial_state,
                                    self.config.direction_seed + (upd_idx + 1) * 100 + j,
                                )
                                ev_t = self.backend.evaluate(
                                    session,
                                    EvaluationRequest(
                                        controls=exc_t,
                                        need_jacobian=False,
                                        request_id=f"step_{upd_idx + 1}_{trial_tag}_{j}",
                                    ),
                                )
                                dt_ls = time.perf_counter() - t_ls0
                                nonlocal physics_forward_s
                                physics_forward_s += dt_ls
                                mse_t, rmse_t = _mse_and_rmse_deg(ev_t.joint_rad, target_arr)
                                limit = best_trial["mse_rad2"] if best_trial is not None else mse_base
                                if mse_t < limit:
                                    best_trial = {
                                        "knots_hz": cand_knots_np,
                                        "waveform_hz": wf_t,
                                        "controls": exc_t,
                                        "eval": ev_t,
                                        "mse_rad2": mse_t,
                                        "rmse_deg": rmse_t,
                                        "eval_time_s": dt_ls,
                                        "alpha": float(alpha),
                                    }
                            return best_trial, tried

                        for upd_idx in range(max_updates):
                            curr = candidates[-1]
                            x_curr_np = curr["knots_hz"]
                            x_curr_vec = x_curr_np.reshape(-1)
                            x_curr_t = torch.tensor(x_curr_np[None, :, :].copy(), dtype=torch.float64, requires_grad=True)

                            if upd_idx == 0 and step0_vjp:
                                ev_vjp = ev0
                            else:
                                cot = (2.0 / STATE_STEPS) * (curr["eval"].joint_rad - target_arr)
                                _notify("correcting", step=upd_idx)
                                t_vjp0 = time.perf_counter()
                                self.backend.reset(session, initial_state, self.config.direction_seed + upd_idx)
                                ev_vjp = self.backend.evaluate(
                                    session,
                                    EvaluationRequest(
                                        controls=curr["controls"],
                                        need_jacobian=True,
                                        cotangent_qpos=cot,
                                        check_directions=bool(check_directions and upd_idx == 0),
                                        direction_seed=self.config.direction_seed,
                                        clip_boundary_gradient=clip_mode,
                                        request_id=f"step_{upd_idx}_vjp",
                                    ),
                                )
                                dt_vjp = time.perf_counter() - t_vjp0
                                physics_vjp_s += dt_vjp
                                history[-1]["used_jacobian"] = True
                                history[-1]["gradient_valid"] = ev_vjp.gradient_valid
                                history[-1]["vjp_time_s"] = dt_vjp
                                if ev_vjp.control_vjp is None or ev_vjp.gradient_valid is False:
                                    break

                            assert ev_vjp.control_vjp is not None
                            grad_curr_np = self._knots_vjp_step(x_curr_t, ev_vjp.control_vjp)[0].cpu().numpy()
                            g_curr_vec = grad_curr_np.reshape(-1)

                            if prev_x_vec is not None and prev_g_vec is not None:
                                sk = x_curr_vec - prev_x_vec
                                yk = g_curr_vec - prev_g_vec
                                if float(np.dot(yk, sk)) > 1e-20:
                                    s_hist.append(sk)
                                    y_hist.append(yk)
                                    if len(s_hist) > history_m:
                                        s_hist.pop(0)
                                        y_hist.pop(0)

                            d_np = _lbfgs_direction(
                                grad_curr_np,
                                s_hist,
                                y_hist,
                                init_step_rms_hz=min(lr_hz, 15.0),
                                max_step_rms_hz=15.0,
                            )
                            if np.all(d_np == 0.0):
                                break

                            alphas = (2.0, 1.0, 0.5, 0.25, 0.1, 0.025, 0.005)
                            best_trial, tried_count = _run_line_search(
                                x_curr_np, curr["mse_rad2"], d_np, alphas, upd_idx, "ls"
                            )
                            if best_trial is None and s_hist:
                                s_hist.clear()
                                y_hist.clear()
                                d_sd = _lbfgs_direction(
                                    grad_curr_np,
                                    s_hist,
                                    y_hist,
                                    init_step_rms_hz=min(lr_hz, 15.0),
                                    max_step_rms_hz=15.0,
                                )
                                best_trial, tried_sd = _run_line_search(
                                    x_curr_np,
                                    curr["mse_rad2"],
                                    d_sd,
                                    (2.0, 0.5, 0.1, 0.025, 0.005, 0.001),
                                    upd_idx,
                                    "sd",
                                )
                                tried_count += tried_sd

                            if best_trial is None:
                                break

                            prev_x_vec = x_curr_vec.copy()
                            prev_g_vec = g_curr_vec.copy()
                            met_next = _append_candidate(
                                step_num=upd_idx + 1,
                                knots_np=best_trial["knots_hz"],
                                waveform_np=best_trial["waveform_hz"],
                                exc_np=best_trial["controls"],
                                ev=best_trial["eval"],
                                mse=best_trial["mse_rad2"],
                                rmse_deg=best_trial["rmse_deg"],
                                used_jacobian=False,
                                gradient_valid=None,
                                eval_time_s=best_trial["eval_time_s"],
                                extra_history={
                                    "line_search_alpha": best_trial["alpha"],
                                    "line_search_trials": tried_count,
                                    "lbfgs_pairs": len(s_hist),
                                },
                            )
                            if mode == "auto" and met_next:
                                break
                elif mode == "refine" and max_updates > 0:
                    for step in range(max_updates + 1):
                        is_last = step == max_updates
                        need_vjp = not is_last
                        with torch.no_grad():
                            knots_param.clamp_(0.0, max_hz)
                        waveform_np, exc_np = self._knots_to_controls(knots_param)
                        _notify("evaluating", step=step, need_jacobian=need_vjp)
                        t_eval0 = time.perf_counter()
                        self.backend.reset(session, initial_state, self.config.direction_seed + step)
                        ev = self.backend.evaluate(
                            session,
                            EvaluationRequest(
                                controls=exc_np,
                                need_jacobian=need_vjp,
                                mse_target_rad=target_arr if need_vjp else None,
                                check_directions=bool(check_directions and need_vjp),
                                direction_seed=self.config.direction_seed,
                                clip_boundary_gradient=clip_mode if need_vjp else "native",
                                request_id=f"step_{step}",
                            ),
                        )
                        dt_eval = time.perf_counter() - t_eval0
                        if need_vjp:
                            physics_vjp_s += dt_eval
                        else:
                            physics_forward_s += dt_eval

                        mse, rmse_deg = _mse_and_rmse_deg(ev.joint_rad, target_arr)
                        _append_candidate(
                            step_num=step,
                            knots_np=knots_param[0].detach().cpu().numpy(),
                            waveform_np=waveform_np,
                            exc_np=exc_np,
                            ev=ev,
                            mse=mse,
                            rmse_deg=rmse_deg,
                            used_jacobian=need_vjp,
                            gradient_valid=ev.gradient_valid,
                            eval_time_s=dt_eval,
                        )
                        if need_vjp:
                            if ev.control_vjp is None or ev.gradient_valid is False:
                                break
                            grad_knots = self._knots_vjp_step(knots_param, ev.control_vjp)
                            optimizer.zero_grad(set_to_none=True)
                            knots_param.grad = grad_knots
                            torch.nn.utils.clip_grad_norm_([knots_param], max_norm=1.0)
                            optimizer.step()
                else:
                    # Fast mode or auto mode (adam): forward first, only spend VJP compute if threshold not met.
                    for step in range(max_updates + 1):
                        with torch.no_grad():
                            knots_param.clamp_(0.0, max_hz)
                        waveform_np, exc_np = self._knots_to_controls(knots_param)
                        _notify("evaluating", step=step, need_jacobian=False)
                        t_eval0 = time.perf_counter()
                        self.backend.reset(session, initial_state, self.config.direction_seed + step)
                        ev = self.backend.evaluate(
                            session,
                            EvaluationRequest(
                                controls=exc_np,
                                need_jacobian=False,
                                request_id=f"step_{step}_fwd",
                            ),
                        )
                        dt_fwd = time.perf_counter() - t_eval0
                        physics_forward_s += dt_fwd

                        mse, rmse_deg = _mse_and_rmse_deg(ev.joint_rad, target_arr)
                        met = _append_candidate(
                            step_num=step,
                            knots_np=knots_param[0].detach().cpu().numpy(),
                            waveform_np=waveform_np,
                            exc_np=exc_np,
                            ev=ev,
                            mse=mse,
                            rmse_deg=rmse_deg,
                            used_jacobian=False,
                            gradient_valid=None,
                            eval_time_s=dt_fwd,
                        )
                        if met or step >= max_updates:
                            break

                        # Request VJP at current candidate using observed q_actual -> exact cotangent
                        cot = (2.0 / STATE_STEPS) * (ev.joint_rad - target_arr)
                        _notify("correcting", step=step)
                        t_vjp0 = time.perf_counter()
                        self.backend.reset(session, initial_state, self.config.direction_seed + step)
                        ev_vjp = self.backend.evaluate(
                            session,
                            EvaluationRequest(
                                controls=exc_np,
                                need_jacobian=True,
                                cotangent_qpos=cot,
                                check_directions=bool(check_directions and step == 0),
                                direction_seed=self.config.direction_seed,
                                clip_boundary_gradient=clip_mode,
                                request_id=f"step_{step}_vjp",
                            ),
                        )
                        dt_vjp = time.perf_counter() - t_vjp0
                        physics_vjp_s += dt_vjp
                        history[-1]["used_jacobian"] = True
                        history[-1]["gradient_valid"] = ev_vjp.gradient_valid
                        history[-1]["vjp_time_s"] = dt_vjp
                        if ev_vjp.control_vjp is None or ev_vjp.gradient_valid is False:
                            break
                        grad_knots = self._knots_vjp_step(knots_param, ev_vjp.control_vjp)
                        optimizer.zero_grad(set_to_none=True)
                        knots_param.grad = grad_knots
                        torch.nn.utils.clip_grad_norm_([knots_param], max_norm=1.0)
                        optimizer.step()

                best = min(candidates, key=lambda item: item["mse_rad2"])
                best_eval: Evaluation = best["eval"]

                _notify("final_verification")
                replay_start = time.perf_counter()
                self.backend.reset(session, initial_state, self.config.direction_seed + 9999)
                replay_eval = self.backend.evaluate(
                    session,
                    EvaluationRequest(
                        controls=best["controls"],
                        need_jacobian=False,
                        request_id="independent_replay",
                    ),
                )
                physics_forward_s += time.perf_counter() - replay_start
                max_q_diff = float(np.max(np.abs(replay_eval.joint_rad - best_eval.joint_rad)))
                if max_q_diff > 1e-12:
                    raise RequestValidationError(
                        f"Independent replay discrepancy exceeded 1e-12 rad: {max_q_diff}"
                    )
                for trace_key in TRACE_KEYS:
                    if not np.allclose(replay_eval.traces[trace_key], best_eval.traces[trace_key],
                                       rtol=1e-12, atol=1e-12):
                        max_trace_diff = float(np.max(np.abs(
                            replay_eval.traces[trace_key] - best_eval.traces[trace_key]
                        )))
                        raise RequestValidationError(
                            f"Independent replay trace {trace_key!r} discrepancy exceeded tolerance: {max_trace_diff}"
                        )
                best_eval = replay_eval
                independent_replay_verified = True

                render_info: dict[str, Any] = {"rendered": False, "artifacts": {}}
                if render and out_dir is not None:
                    if isinstance(self.backend, RenderCapableBackend):
                        _notify("rendering")
                        render_res = self.backend.render_rollout(
                            session,
                            best["controls"],
                            out_dir,
                            stem="simulation_top_view",
                            initial=initial_state,
                        )
                        render_arts: dict[str, str] = {}
                        if render_res.mp4_path and render_res.mp4_path.is_file():
                            render_arts["video_mp4"] = render_res.mp4_path.name
                        if render_res.webm_path and render_res.webm_path.is_file():
                            render_arts["video_webm"] = render_res.webm_path.name
                        if render_res.preview_png_path and render_res.preview_png_path.is_file():
                            render_arts["preview_png"] = render_res.preview_png_path.name
                        render_info = {
                            "rendered": True,
                            "visible": render_res.visible,
                            "frames": render_res.frame_count,
                            "artifacts": render_arts,
                        }
                    else:
                        render_info = {
                            "rendered": False,
                            "reason": f"Backend {self.backend.describe().name!r} does not implement rendering",
                            "artifacts": {},
                        }
                elif render:
                    render_info = {
                        "rendered": False,
                        "reason": "render=True requires output directory",
                        "artifacts": {},
                    }
                session_succeeded = True
            except Exception as exc:
                if out_dir is not None:
                    partial_name = _persist_partial_evidence(
                        out_dir,
                        profile=self._profile,
                        model_id=self._model_entry.model_id,
                        mode=mode,
                        initializer=initializer,
                        target_arr=target_arr,
                        delta_rad=delta_rad,
                        history=history,
                        candidates=candidates,
                        exc=exc,
                        elapsed_s=time.perf_counter() - run_start,
                    )
                    _notify("failed", error=f"{type(exc).__name__}: {exc}", partial_evidence=partial_name)
                raise
            finally:
                if session is not None:
                    if session_succeeded:
                        self.backend.close_session(session)
                    else:
                        try:
                            self.backend.close_session(session)
                        except Exception:
                            pass

            total_s = time.perf_counter() - run_start
            goal_id = hashlib.sha256(target_arr.tobytes()).hexdigest()[:16]
            feedback_info = (
                {
                    "kp": kp,
                    "kd": kd,
                    "ext_to_flex_ratio": EXT_TO_FLEX_RATIO,
                    "dt_ms": 2,
                }
                if mode == "closed_loop"
                else None
            )
            protocol_rate = MotorRateProtocol(
                goal_id=goal_id,
                command_pool_rates_hz=best["waveform_hz"],
                behavior_simulation_evaluated=True,
                joint_rmse_deg=float(best["rmse_deg"]),
                accuracy_threshold_deg=threshold,
                max_rate_hz=max_hz,
                tau_ms=tau_ms,
                rate_scale_hz=rate_scale_hz,
                clip_boundary=clip_mode,
                protocol_kind=(
                    "hybrid_feedforward_rate_with_causal_proprioceptive_feedback"
                    if mode == "closed_loop"
                    else "continuous_pool_rate_waveform"
                ),
                extra_metadata=(
                    {
                        "feedback": feedback_info,
                        "realized_excitation_max": np.max(best["controls"], axis=0).tolist(),
                    }
                    if feedback_info is not None
                    else None
                ),
            ).to_dict()

            backend_id = self.backend.describe()
            if not backend_id.is_certified_frozen_physics:
                protocol_rate["behavior_simulation_evaluated"] = False

            protocol_physics = {
                "schema_version": "sensory_translator_v2_physics_verification_v1",
                "mode": mode,
                "profile": self._profile,
                "correction_optimizer": correction_optimizer,
                "clip_boundary_gradient": clip_mode,
                "feedback": feedback_info,
                "backend": backend_id.to_dict(),
                "is_certified_frozen_physics": backend_id.is_certified_frozen_physics,
                "independent_replay_verified": independent_replay_verified,
                "selected_step": int(best["step"]),
                "initial_tibia_delta_rad": float(delta_rad),
                "initial_tibia_delta_deg": float(math.degrees(delta_rad)),
                "nominal_mse_rad2": float(candidates[0]["mse_rad2"]),
                "nominal_rmse_deg": float(candidates[0]["rmse_deg"]),
                "best_mse_rad2": float(best["mse_rad2"]),
                "best_rmse_deg": float(best["rmse_deg"]),
                "threshold_deg": threshold,
                "accuracy_threshold_met": bool(best["rmse_deg"] <= threshold),
                "target_joint": TARGET_JOINT,
                "target_qpos_address": TARGET_QPOS_ADDRESS,
                "target_actuator_columns": list(TARGET_ACTUATOR_COLUMNS),
                "target_actuator_names": list(TARGET_ACTUATOR_NAMES),
                "independent_teacher_gate": best_eval.independent_teacher_gate,
                "timing_s": {
                    "mlp_s": mlp_s,
                    "physics_forward_s": physics_forward_s,
                    "physics_vjp_s": physics_vjp_s,
                    "total_s": total_s,
                },
                "history": history,
            }

            result: dict[str, Any] = {
                "schema_version": "sensory_translator_v2_run_v1",
                "created_utc": utc_now_iso(),
                "mode": mode,
                "initializer": initializer,
                "correction_optimizer": correction_optimizer,
                "clip_boundary_gradient": clip_mode,
                "feedback": feedback_info,
                "profile": self._profile,
                "model_id": self._model_entry.model_id,
                "backend": backend_id.to_dict(),
                "is_certified_frozen_physics": backend_id.is_certified_frozen_physics,
                "selected_step": int(best["step"]),
                "correction_steps_taken": len(candidates) - 1,
                "nominal_rmse_deg": float(candidates[0]["rmse_deg"]),
                "best_rmse_deg": float(best["rmse_deg"]),
                "best_mse_rad2": float(best["mse_rad2"]),
                "threshold_deg": threshold,
                "accuracy_threshold_met": bool(best["rmse_deg"] <= threshold),
                "independent_replay_verified": independent_replay_verified,
                "initial_tibia_delta_deg": float(math.degrees(delta_rad)),
                "target_rad": target_arr.tolist(),
                "actual_rad": best_eval.joint_rad.tolist(),
                "time_s": time_arr.tolist(),
                "rate_knots_hz": best["knots_hz"].tolist(),
                "trace": {
                    "time_ms": (time_arr * 1000.0).tolist(),
                    "target_deg": np.rad2deg(target_arr).tolist(),
                    "actual_deg": np.rad2deg(best_eval.joint_rad).tolist(),
                    "rate_time_ms": list(range(420)),
                    "rates_hz": best["waveform_hz"].tolist(),
                },
                "max_abs_joint_error_deg": float(np.max(np.abs(np.rad2deg(best_eval.joint_rad - target_arr)))),
                "nominal_max_abs_joint_error_deg": float(np.max(np.abs(np.rad2deg(candidates[0]["eval"].joint_rad - target_arr)))),
                "timing_s": protocol_physics["timing_s"],
                "history": history,
                "render": render_info,
                "artifacts": {},
            }

            if out_dir is not None:
                traces_npz = {key: best_eval.traces[key] for key in TRACE_KEYS}
                traces_npz["target_rad"] = target_arr
                traces_npz["actual_rad"] = best_eval.joint_rad
                traces_npz["command_pool_rates_hz"] = best["waveform_hz"]
                traces_npz["rate_knots_hz"] = best["knots_hz"]
                traces_npz["excitation"] = best["controls"]
                if mode == "closed_loop":
                    traces_npz["feedforward_excitation"] = nominal_exc_np
                artifacts = write_run_artifacts(
                    out_dir,
                    manifest=result,
                    protocol_rate=protocol_rate,
                    protocol_physics=protocol_physics,
                    traces_npz=traces_npz,
                )
                artifacts.update(render_info.get("artifacts", {}))
                result["artifacts"] = artifacts
                write_json_atomic(out_dir / "manifest.json", result)

            _notify("completed", best_rmse_deg=result["best_rmse_deg"])
            return result

    def close(self) -> None:
        """Close backend processes cleanly; idempotent."""
        with self._serial_lock:
            self._closed = True
            if self._backend is not None:
                self._backend.close()
