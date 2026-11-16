"""V1 frozen subprocess adapter implementing the v2 Backend & Render protocols."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from ..assets import AssetCatalog
from ..contracts import (
    BackendIdentity,
    BackendSessionError,
    BackendTimeoutError,
    BackendUnavailableError,
    Evaluation,
    EvaluationRequest,
    InitialState,
    RenderCapableBackend,
    RenderResult,
    Session,
    SessionSpec,
    TARGET_QPOS_ADDRESS,
    TRACE_KEYS,
    require_int,
    validate_controls,
)
from ..evidence import utc_now_iso
from .base import (
    require_open_session,
    require_reset_session,
    validate_session_spec_against_catalog,
)


class V1SubprocessBackend(RenderCapableBackend):
    """Wraps frozen v1 subprocess worker scripts through the stateless v2 Backend API."""

    def __init__(
        self,
        catalog: AssetCatalog,
        *,
        profile: str = "adjacent_static_parent_v1",
        timeout_s: float = 90.0,
    ) -> None:
        if catalog.scripts_root is None:
            raise BackendUnavailableError(
                "V1SubprocessBackend requires v1 checkout scripts (absent in stripped bundle)"
            )
        if not catalog.physics_python.is_file():
            raise BackendUnavailableError(
                f"Configured physics_python does not exist: {catalog.physics_python}"
            )
        self.catalog = catalog
        self.profile = profile
        self.timeout_s = float(timeout_s)
        self._lock = threading.RLock()
        self._sessions: dict[str, Session] = {}
        self._closed = False

    def _script_for(self, profile: str, *, need_jacobian: bool) -> Path:
        assert self.catalog.scripts_root is not None
        if profile == "original_v1":
            name = "differentiate_neuromuscle_plant.py" if need_jacobian else "render_neuromuscle_top_view.py"
        elif profile == "multiccd_v1":
            name = "neuromuscle_multiccd_worker.py"
        elif profile == "adjacent_static_parent_v1":
            name = "neuromuscle_adjacent_worker.py"
        else:
            raise BackendSessionError(f"Unsupported profile: {profile!r}")
        script = self.catalog.scripts_root / name
        if not script.is_file():
            raise BackendUnavailableError(f"Missing frozen v1 script {script}")
        return script

    def describe(self) -> BackendIdentity:
        return BackendIdentity(
            name="v1_subprocess",
            profile=self.profile,
            backend_version="2.0.0-v1adapter",
            is_certified_frozen_physics=True,
            cache_mode="subprocess",
            asset_hashes=self.catalog.three_hash_record(self.profile),
            runtime_versions={"physics_python": str(self.catalog.physics_python)},
            details={
                "scripts_root": str(self.catalog.scripts_root),
                "execution_model": "subprocess_per_evaluation",
            },
        )

    def open_session(self, spec: SessionSpec) -> Session:
        with self._lock:
            if self._closed:
                raise BackendSessionError("V1SubprocessBackend is closed")
            hashes = validate_session_spec_against_catalog(spec, self.catalog)
            session_id = uuid.uuid4().hex
            identity = BackendIdentity(
                name="v1_subprocess",
                profile=spec.profile,
                backend_version="2.0.0-v1adapter",
                is_certified_frozen_physics=True,
                cache_mode="subprocess",
                asset_hashes=hashes,
                runtime_versions={"physics_python": str(self.catalog.physics_python)},
            )
            session = Session(
                session_id=session_id,
                backend_name="v1_subprocess",
                spec=spec,
                identity=identity,
                created_utc=utc_now_iso(),
            )
            self._sessions[session_id] = session
            return session

    def reset(self, session: Session, initial: InitialState, seed: int) -> None:
        with self._lock:
            if self._closed:
                raise BackendSessionError("V1SubprocessBackend is closed")
            tracked = require_open_session(session, self._sessions)
            require_int(seed, "seed", minimum=0)
            tracked.active_initial = initial
            tracked.active_seed = seed
            session.active_initial = initial
            session.active_seed = seed

    def _run_subprocess(
        self,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
    ) -> float:
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendTimeoutError(
                f"v1 worker subprocess timed out after {self.timeout_s}s: {cmd[1]}"
            ) from exc
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or proc.stdout or "").strip()[-800:]
            raise BackendSessionError(
                f"v1 worker subprocess failed (exit {proc.returncode}): {stderr_tail}"
            )
        return time.perf_counter() - t0

    def evaluate(self, session: Session, request: EvaluationRequest) -> Evaluation:
        tracked = require_reset_session(session, self._sessions)
        assert tracked.active_initial is not None
        if request.clip_boundary_gradient != "native":
            from ..contracts import RequestValidationError

            raise RequestValidationError(
                "V1SubprocessBackend only supports clip_boundary_gradient='native'"
            )
        profile = tracked.spec.profile

        cotangent = request.cotangent_qpos
        if request.need_jacobian and cotangent is None:
            assert request.mse_target_rad is not None
            fwd_eval = self.evaluate(
                session,
                EvaluationRequest(
                    controls=request.controls,
                    need_jacobian=False,
                    request_id=f"{request.request_id}_prefwd" if request.request_id else "",
                ),
            )
            cotangent = (2.0 / len(fwd_eval.joint_rad)) * (
                fwd_eval.joint_rad - request.mse_target_rad
            )

        script = self._script_for(profile, need_jacobian=request.need_jacobian)

        with tempfile.TemporaryDirectory(prefix="st_v2_v1sub_") as tmpdir:
            tmp = Path(tmpdir)
            in_npz = tmp / "input.npz"
            out_npz = tmp / "output.npz"
            report_json = tmp / "report.json"

            arrays: dict[str, Any] = {
                "excitation": request.controls[None, :, :].astype(np.float64),
                "initial_tibia_delta_rad": np.array(
                    [tracked.active_initial.initial_tibia_delta_rad], dtype=np.float64
                ),
            }
            if request.need_jacobian and cotangent is not None:
                arrays["qpos_cotangent"] = cotangent[None, :].astype(np.float64)
            np.savez(in_npz, **arrays)

            cmd = [
                str(self.catalog.physics_python),
                str(script),
                "--input",
                str(in_npz),
                "--output",
                str(out_npz),
                "--report",
                str(report_json),
                "--teacher",
                str(self.catalog.teacher_path),
            ]
            if request.need_jacobian or profile != "original_v1":
                if profile != "original_v1" and request.need_jacobian:
                    cmd.append("--with-gradient")
                if request.need_jacobian:
                    cmd.extend([
                        "--jacobian-eps",
                        repr(tracked.spec.jacobian_eps),
                        "--direction-eps",
                        repr(tracked.spec.direction_eps),
                        "--check-directions",
                        str(2 if request.check_directions else 0),
                    ])

            elapsed_s = self._run_subprocess(cmd)
            if not out_npz.is_file() or not report_json.is_file():
                raise BackendSessionError("v1 worker exited 0 without producing output artifacts")
            report = json.loads(report_json.read_text(encoding="utf-8"))
            with np.load(out_npz, allow_pickle=False) as data:
                traces = {key: np.asarray(data[key][0], dtype=np.float64) for key in TRACE_KEYS}
                control_vjp = (
                    np.asarray(data["excitation_gradient"][0], dtype=np.float64)
                    if request.need_jacobian and "excitation_gradient" in data.files
                    else None
                )

        dir_checks = report.get("direction_checks")
        normalized_checks: dict[str, Any] | None = None
        gradient_valid = None
        if dir_checks is not None:
            if isinstance(dir_checks, dict):
                normalized_checks = dir_checks
            elif isinstance(dir_checks, list):
                # v1 batch workers emit per-trial list[list[dict]] when batch=1.
                items = dir_checks[0] if (len(dir_checks) == 1 and isinstance(dir_checks[0], list)) else dir_checks
                if all(isinstance(x, dict) for x in items):
                    all_ok = len(items) > 0 and all(bool(x.get("passed")) for x in items)
                    normalized_checks = {"all_passed": all_ok, "checks": items}
                else:
                    normalized_checks = {"all_passed": False, "items": dir_checks}

        if request.need_jacobian:
            if request.check_directions:
                gradient_valid = bool(
                    normalized_checks is not None and normalized_checks.get("all_passed", False)
                )
            else:
                gradient_valid = control_vjp is not None and bool(np.isfinite(control_vjp).all())

        with self._lock:
            tracked.evaluation_count += 1
            session.evaluation_count = tracked.evaluation_count

        return Evaluation(
            request_id=request.request_id or uuid.uuid4().hex,
            session_id=tracked.session_id,
            profile=profile,
            is_certified_frozen_physics=True,
            traces=traces,
            joint_rad=traces["qpos"][:, TARGET_QPOS_ADDRESS],
            control_vjp=control_vjp,
            gradient_valid=gradient_valid,
            direction_checks=normalized_checks,
            independent_teacher_gate=report.get("independent_teacher_gate") or report.get("reference_gate"),
            timing_s={"subprocess_s": elapsed_s, "total_s": elapsed_s},
            metadata={
                "v1_script": script.name,
                "report_schema": report.get("schema") or report.get("schema_version"),
            },
        )

    def render_rollout(
        self,
        session: Session,
        controls: np.ndarray,
        output_dir: Path,
        *,
        stem: str = "simulation_top_view",
        initial: InitialState | None = None,
    ) -> RenderResult:
        tracked = require_open_session(session, self._sessions)
        init_state = initial or tracked.active_initial or InitialState(0.0)
        validated_controls = validate_controls(controls)
        profile = tracked.spec.profile
        script = (
            self.catalog.scripts_root / "render_neuromuscle_top_view.py"
            if profile == "original_v1"
            else self._script_for(profile, need_jacobian=False)
        )
        assert self.catalog.scripts_root is not None
        if not script.is_file():
            raise BackendUnavailableError(f"Render script missing: {script}")

        output_dir.mkdir(parents=True, exist_ok=True)
        mp4_path = output_dir / f"{stem}.mp4"
        with tempfile.TemporaryDirectory(prefix="st_v2_v1rnd_") as tmpdir:
            tmp = Path(tmpdir)
            in_npz = tmp / "input.npz"
            out_npz = tmp / "output.npz"
            report_json = tmp / "render_report.json"
            np.savez(
                in_npz,
                excitation=validated_controls[None, :, :].astype(np.float64),
                initial_tibia_delta_rad=np.array([init_state.initial_tibia_delta_rad], dtype=np.float64),
            )
            cmd = [
                str(self.catalog.physics_python),
                str(script),
                "--input",
                str(in_npz),
                "--output",
                str(out_npz),
                "--report",
                str(report_json),
                "--teacher",
                str(self.catalog.teacher_path),
                "--render-trial",
                "0",
                "--video",
                str(mp4_path),
            ]
            env = dict(os.environ)
            env.setdefault("MUJOCO_GL", "egl")
            self._run_subprocess(cmd, env=env)
            report = json.loads(report_json.read_text(encoding="utf-8"))

        webm_path = mp4_path.with_suffix(".webm")
        png_path = output_dir / f"{stem}_preview.png"
        render_info = report.get("render", {}) if isinstance(report.get("render"), dict) else {}
        return RenderResult(
            mp4_path=mp4_path if mp4_path.is_file() else None,
            webm_path=webm_path if webm_path.is_file() else None,
            preview_png_path=png_path if png_path.is_file() else None,
            frame_count=int(render_info.get("frames", 42)),
            visible=bool(render_info.get("visible", mp4_path.is_file())),
            report=report,
        )

    def close_session(self, session: Session) -> None:
        with self._lock:
            tracked = self._sessions.get(session.session_id)
            if tracked is not None:
                tracked.closed = True
            session.closed = True

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for session in self._sessions.values():
                session.closed = True
