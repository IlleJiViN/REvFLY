"""Persistent MuJoCo IPC backend supporting M2 (fresh) and M3 (cached_model)."""

from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from ..assets import AssetCatalog, materialize_profile_model
from ..contracts import (
    BackendIdentity,
    BackendSessionError,
    BackendUnavailableError,
    ClosedLoopRequest,
    Evaluation,
    EvaluationRequest,
    InitialState,
    LF_ACTUATOR_COLUMNS,
    LF_QPOS_ADDRESSES,
    MultiJointClosedLoopRequest,
    MultiJointEvaluationRequest,
    RenderCapableBackend,
    RenderResult,
    Session,
    SessionSpec,
    TARGET_QPOS_ADDRESS,
    TRACE_KEYS,
    require_int,
    validate_controls,
    validate_index_tuple,
    validate_matrix,
)
from ..evidence import utc_now_iso
from .base import (
    require_open_session,
    require_reset_session,
    validate_session_spec_against_catalog,
)
from .worker_protocol import PROTOCOL_VERSION, read_frame_fd, write_frame


class PersistentMujocoBackend(RenderCapableBackend):
    """Long-lived worker-process MuJoCo backend eliminating per-call Python/XML startup."""

    def __init__(
        self,
        catalog: AssetCatalog,
        *,
        profile: str = "adjacent_static_parent_v1",
        cache_mode: str = "cached_model",
        timeout_s: float = 90.0,
    ) -> None:
        if cache_mode not in ("fresh", "cached_model"):
            raise BackendSessionError(f"Unsupported cache_mode: {cache_mode!r}")
        if not catalog.physics_python.is_file():
            raise BackendUnavailableError(
                f"Configured physics_python does not exist: {catalog.physics_python}"
            )
        self.catalog = catalog
        self.profile = profile
        self.cache_mode = cache_mode
        self.timeout_s = float(timeout_s)
        self._lock = threading.RLock()
        self._sessions: dict[str, Session] = {}
        self._closed = False

        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr_lines: deque[str] = deque(maxlen=64)
        self._stderr_thread: threading.Thread | None = None
        self._handshake_info: dict[str, Any] = {}
        self._last_handshake_s: float = 0.0
        self._materialized = materialize_profile_model(self.catalog, self.profile)

    def _stderr_tail(self) -> str:
        return "\n".join(self._stderr_lines).strip()

    def _terminate_worker(self) -> None:
        proc = self._proc
        self._proc = None
        self._handshake_info = {}
        for session in self._sessions.values():
            session.closed = True
        self._sessions.clear()
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        try:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=2.0)
        except Exception:
            pass
        try:
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
        except OSError:
            pass

    def _drain_stderr(self, pipe: Any) -> None:
        try:
            for raw in iter(pipe.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    self._stderr_lines.append(line)
        except Exception:
            pass

    def _ensure_worker(self) -> dict[str, Any]:
        if self._closed:
            raise BackendSessionError("PersistentMujocoBackend is closed")
        if self._proc is not None and self._proc.poll() is None and self._handshake_info:
            return self._handshake_info
        self._terminate_worker()
        self._stderr_lines.clear()
        self._materialized = materialize_profile_model(self.catalog, self.profile)

        pkg_root = str(Path(__file__).resolve().parents[2])
        env = dict(os.environ)
        env.setdefault("MUJOCO_GL", "egl")
        existing_pp = env.get("PYTHONPATH", "")
        pp_parts = [p for p in existing_pp.split(os.pathsep) if p]
        if pkg_root not in pp_parts:
            pp_parts.append(pkg_root)
        env["PYTHONPATH"] = os.pathsep.join(pp_parts)

        t0 = time.perf_counter()
        proc = subprocess.Popen(
            [str(self.catalog.physics_python), "-u", "-m", "sensory_translator_v2.backends.mujoco_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self._proc = proc
        assert proc.stderr is not None and proc.stdin is not None and proc.stdout is not None
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(proc.stderr,),
            name="v2-worker-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

        try:
            write_frame(
                proc.stdin,
                {
                    "op": "handshake",
                    "protocol_version": PROTOCOL_VERSION,
                    "profile": self.profile,
                    "xml_path": str(self._materialized["xml_path"]),
                    "teacher_path": str(self.catalog.teacher_path),
                    "expected_teacher_sha256": self.catalog.teacher_sha256,
                    "cache_mode": self.cache_mode,
                },
            )
            resp = read_frame_fd(proc.stdout.fileno(), timeout_s=self.timeout_s)
            if resp is None:
                raise BackendSessionError(
                    f"Worker closed during handshake. stderr:\n{self._stderr_tail()}"
                )
            header, _ = resp
            if not header.get("ok"):
                raise BackendSessionError(
                    f"Worker handshake rejected ({header.get('error_type')}): {header.get('error')}"
                )
            self._handshake_info = header
            self._last_handshake_s = time.perf_counter() - t0
            return header
        except Exception:
            self._terminate_worker()
            raise

    def _require_session_worker(self) -> subprocess.Popen[bytes]:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            self._terminate_worker()
            raise BackendSessionError("Persistent worker exited; open a new session")
        return proc

    def _rpc(
        self,
        session: Session,
        header: dict[str, Any],
        arrays: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        with self._lock:
            proc = self._require_session_worker()
            assert proc.stdin is not None and proc.stdout is not None
            try:
                write_frame(proc.stdin, header, arrays)
                resp = read_frame_fd(proc.stdout.fileno(), timeout_s=self.timeout_s)
                if resp is None:
                    raise BackendSessionError(
                        f"Worker terminated unexpectedly. stderr:\n{self._stderr_tail()}"
                    )
                resp_hdr, resp_arrays = resp
            except Exception as exc:
                self._terminate_worker()
                if isinstance(exc, BackendSessionError):
                    raise
                raise BackendSessionError(
                    f"Persistent worker IPC failed: {exc}. stderr:\n{self._stderr_tail()}"
                ) from exc

            if not resp_hdr.get("ok"):
                self._invalidate_session(session)
                raise BackendSessionError(
                    f"Worker error ({resp_hdr.get('error_type')}): {resp_hdr.get('error')}"
                )
            return resp_hdr, resp_arrays

    def _invalidate_session(self, session: Session) -> None:
        tracked = self._sessions.pop(session.session_id, None)
        if tracked is not None:
            tracked.closed = True
        session.closed = True

    def ensure_ready(self) -> dict[str, Any]:
        """Synchronously ensure persistent worker is spawned, verified, and warm."""
        with self._lock:
            return dict(self._ensure_worker())

    def describe(self) -> BackendIdentity:
        name = "persistent_cached" if self.cache_mode == "cached_model" else "persistent_fresh"
        return BackendIdentity(
            name=name,
            profile=self.profile,
            backend_version="2.0.0-ipc",
            is_certified_frozen_physics=True,
            cache_mode=self.cache_mode,
            asset_hashes=dict(self._materialized["three_hashes"]),
            runtime_versions=dict(
                self._handshake_info.get(
                    "runtime_versions",
                    {"physics_python": str(self.catalog.physics_python)},
                )
            ),
            details={
                "worker_pid": self._handshake_info.get("worker_pid"),
                "worker_alive": bool(self._proc is not None and self._proc.poll() is None),
                "execution_model": f"persistent_worker_{self.cache_mode}",
                "supports_one_sided_boundary_gradient": True,
                "supports_closed_loop": True,
                "supports_multijoint": True,
                "max_controlled_muscles": 15,
            },
        )

    def open_session(self, spec: SessionSpec) -> Session:
        with self._lock:
            if self._closed:
                raise BackendSessionError("PersistentMujocoBackend is closed")
            validate_session_spec_against_catalog(spec, self.catalog, expected_profile=self.profile)
            self._ensure_worker()
            session_id = uuid.uuid4().hex
            identity = self.describe()
            session = Session(
                session_id=session_id,
                backend_name=identity.name,
                spec=spec,
                identity=identity,
                created_utc=utc_now_iso(),
            )
            self._sessions[session_id] = session
            return session

    def reset(self, session: Session, initial: InitialState, seed: int) -> None:
        with self._lock:
            if self._closed:
                raise BackendSessionError("PersistentMujocoBackend is closed")
            tracked = require_open_session(session, self._sessions)
            self._require_session_worker()
            require_int(seed, "seed", minimum=0)
            tracked.active_initial = initial
            tracked.active_seed = seed
            session.active_initial = initial
            session.active_seed = seed

    def _eval_worker_rpc(
        self,
        session: Session,
        tracked: Session,
        header: dict[str, Any],
        in_arrays: dict[str, np.ndarray],
        req_id: str,
        *,
        need_jacobian: bool,
        check_directions: bool,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Evaluation:
        t0 = time.perf_counter()
        resp_hdr, resp_arrays = self._rpc(
            session,
            header,
            in_arrays,
        )
        ipc_total_s = time.perf_counter() - t0

        traces = {key: resp_arrays[key] for key in TRACE_KEYS}
        control_vjp = resp_arrays.get("control_vjp") if need_jacobian else None
        dir_checks = resp_hdr.get("direction_checks")
        if need_jacobian:
            if check_directions:
                gradient_valid = bool(
                    isinstance(dir_checks, dict) and dir_checks.get("all_passed", False)
                )
            else:
                gradient_valid = bool(control_vjp is not None and np.isfinite(control_vjp).all())
        else:
            gradient_valid = None

        tracked.evaluation_count += 1
        session.evaluation_count = tracked.evaluation_count
        worker_timing = dict(resp_hdr.get("timing_s", {}))
        worker_timing["ipc_roundtrip_s"] = ipc_total_s
        worker_timing["total_s"] = ipc_total_s

        metadata = {
            "worker_pid": self._handshake_info.get("worker_pid"),
            "cache_mode": self.cache_mode,
            "seed": tracked.active_seed,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return Evaluation(
            request_id=req_id,
            session_id=tracked.session_id,
            profile=tracked.spec.profile,
            is_certified_frozen_physics=True,
            traces=traces,
            joint_rad=traces["qpos"][:, TARGET_QPOS_ADDRESS],
            control_vjp=control_vjp,
            gradient_valid=gradient_valid,
            direction_checks=dir_checks,
            independent_teacher_gate=resp_hdr.get("independent_teacher_gate"),
            timing_s=worker_timing,
            metadata=metadata,
        )

    def evaluate_multijoint(
        self,
        session: Session,
        request: MultiJointEvaluationRequest,
    ) -> Evaluation:
        with self._lock:
            tracked = require_reset_session(session, self._sessions)
            req_id = request.request_id or uuid.uuid4().hex
            in_arrays: dict[str, np.ndarray] = {"controls": request.controls}
            if request.initial_qpos_delta_rad is not None:
                in_arrays["initial_qpos_delta_rad"] = request.initial_qpos_delta_rad
            if request.cotangent_qpos is not None:
                in_arrays["cotangent_qpos"] = request.cotangent_qpos
            if request.mse_target_rad is not None:
                in_arrays["mse_target_rad"] = request.mse_target_rad
            if request.joint_weights is not None:
                in_arrays["joint_weights"] = request.joint_weights
            return self._eval_worker_rpc(
                session,
                tracked,
                {
                    "op": "evaluate_multijoint",
                    "session_id": tracked.session_id,
                    "request_id": req_id,
                    "control_columns": list(request.control_columns),
                    "target_qpos_indices": list(request.target_qpos_indices),
                    "need_jacobian": request.need_jacobian,
                    "jacobian_eps": tracked.spec.jacobian_eps,
                    "check_directions": request.check_directions,
                    "direction_seed": request.direction_seed,
                    "direction_eps": tracked.spec.direction_eps,
                    "clip_boundary_gradient": request.clip_boundary_gradient,
                },
                in_arrays,
                req_id,
                need_jacobian=request.need_jacobian,
                check_directions=request.check_directions,
                extra_metadata={
                    "control_columns": list(request.control_columns),
                    "target_qpos_indices": list(request.target_qpos_indices),
                    "controlled_muscles": len(request.control_columns),
                    "teacher_assisted_muscles": 15 - len(request.control_columns),
                },
            )

    def evaluate_multijoint_closed_loop(
        self,
        session: Session,
        request: MultiJointClosedLoopRequest,
    ) -> Evaluation:
        with self._lock:
            tracked = require_reset_session(session, self._sessions)
            req_id = request.request_id or uuid.uuid4().hex
            in_arrays: dict[str, np.ndarray] = {
                "feedforward_controls": request.feedforward_controls,
                "target_rad": request.target_rad,
                "kp": np.asarray(request.kp, dtype=np.float64),
                "kd": np.asarray(request.kd, dtype=np.float64),
            }
            if request.initial_qpos_delta_rad is not None:
                in_arrays["initial_qpos_delta_rad"] = request.initial_qpos_delta_rad
            return self._eval_worker_rpc(
                session,
                tracked,
                {
                    "op": "evaluate_multijoint_closed_loop",
                    "session_id": tracked.session_id,
                    "request_id": req_id,
                    "control_columns": list(request.control_columns),
                    "target_qpos_indices": list(request.target_qpos_indices),
                    "jacobian_refresh_steps": request.jacobian_refresh_steps,
                    "damping": request.damping,
                    "feedback_clip": request.feedback_clip,
                },
                in_arrays,
                req_id,
                need_jacobian=False,
                check_directions=False,
                extra_metadata={
                    "control_columns": list(request.control_columns),
                    "target_qpos_indices": list(request.target_qpos_indices),
                    "controlled_muscles": len(request.control_columns),
                    "teacher_assisted_muscles": 15 - len(request.control_columns),
                    "closed_loop": True,
                },
            )

    def render_multijoint_rollout(
        self,
        session: Session,
        controls: np.ndarray,
        output_dir: Path,
        *,
        control_columns: tuple[int, ...] = LF_ACTUATOR_COLUMNS,
        target_qpos_indices: tuple[int, ...] = LF_QPOS_ADDRESSES,
        initial_qpos_delta_rad: np.ndarray | None = None,
        stem: str = "multijoint_top_view",
    ) -> RenderResult:
        with self._lock:
            tracked = require_open_session(session, self._sessions)
            cols = validate_index_tuple(control_columns, "control_columns", upper_bound=15)
            qidxs = validate_index_tuple(target_qpos_indices, "target_qpos_indices", upper_bound=14)
            validated = validate_matrix(controls, "controls", rows=210, cols=len(cols), bounds=(0.0, 1.0))
            in_arrays: dict[str, np.ndarray] = {"controls": validated}
            if initial_qpos_delta_rad is not None:
                in_arrays["initial_qpos_delta_rad"] = np.asarray(initial_qpos_delta_rad, dtype=np.float64)
            resp_hdr, _ = self._rpc(
                session,
                {
                    "op": "render_multijoint",
                    "session_id": tracked.session_id,
                    "control_columns": list(cols),
                    "target_qpos_indices": list(qidxs),
                    "output_dir": str(output_dir.resolve()),
                    "stem": stem,
                },
                in_arrays,
            )
            info = resp_hdr.get("render", {})
            return RenderResult(
                mp4_path=Path(info["mp4_path"]) if info.get("mp4_path") else None,
                webm_path=Path(info["webm_path"]) if info.get("webm_path") else None,
                preview_png_path=Path(info["preview_png_path"]) if info.get("preview_png_path") else None,
                frame_count=int(info.get("frames", 42)),
                visible=bool(info.get("visible", False)),
                report=info,
            )

    def evaluate(self, session: Session, request: EvaluationRequest) -> Evaluation:
        with self._lock:
            tracked = require_reset_session(session, self._sessions)
            assert tracked.active_initial is not None
            req_id = request.request_id or uuid.uuid4().hex
            in_arrays: dict[str, np.ndarray] = {"controls": request.controls}
            if request.cotangent_qpos is not None:
                in_arrays["cotangent_qpos"] = request.cotangent_qpos
            if request.mse_target_rad is not None:
                in_arrays["mse_target_rad"] = request.mse_target_rad
            return self._eval_worker_rpc(
                session,
                tracked,
                {
                    "op": "evaluate",
                    "session_id": tracked.session_id,
                    "request_id": req_id,
                    "initial_tibia_delta_rad": tracked.active_initial.initial_tibia_delta_rad,
                    "need_jacobian": request.need_jacobian,
                    "jacobian_eps": tracked.spec.jacobian_eps,
                    "check_directions": request.check_directions,
                    "direction_seed": request.direction_seed,
                    "direction_eps": tracked.spec.direction_eps,
                    "clip_boundary_gradient": request.clip_boundary_gradient,
                },
                in_arrays,
                req_id,
                need_jacobian=request.need_jacobian,
                check_directions=request.check_directions,
            )

    def evaluate_closed_loop(self, session: Session, request: ClosedLoopRequest) -> Evaluation:
        with self._lock:
            tracked = require_reset_session(session, self._sessions)
            assert tracked.active_initial is not None
            req_id = request.request_id or uuid.uuid4().hex
            return self._eval_worker_rpc(
                session,
                tracked,
                {
                    "op": "evaluate_closed_loop",
                    "session_id": tracked.session_id,
                    "request_id": req_id,
                    "initial_tibia_delta_rad": tracked.active_initial.initial_tibia_delta_rad,
                    "kp": request.kp,
                    "kd": request.kd,
                    "ext_to_flex_ratio": request.ext_to_flex_ratio,
                },
                {
                    "feedforward_controls": request.feedforward_controls,
                    "target_rad": request.target_rad,
                },
                req_id,
                need_jacobian=False,
                check_directions=False,
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
        with self._lock:
            tracked = require_open_session(session, self._sessions)
            init_state = initial or tracked.active_initial or InitialState(0.0)
            validated = validate_controls(controls)
            resp_hdr, _ = self._rpc(
                session,
                {
                    "op": "render",
                    "session_id": tracked.session_id,
                    "initial_tibia_delta_rad": init_state.initial_tibia_delta_rad,
                    "output_dir": str(output_dir.resolve()),
                    "stem": stem,
                },
                {"controls": validated},
            )
            info = resp_hdr.get("render", {})
            return RenderResult(
                mp4_path=Path(info["mp4_path"]) if info.get("mp4_path") else None,
                webm_path=Path(info["webm_path"]) if info.get("webm_path") else None,
                preview_png_path=Path(info["preview_png_path"]) if info.get("preview_png_path") else None,
                frame_count=int(info.get("frames", 42)),
                visible=bool(info.get("visible", False)),
                report=info,
            )

    def close_session(self, session: Session) -> None:
        with self._lock:
            self._invalidate_session(session)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            proc = self._proc
            if proc is not None and proc.poll() is None and proc.stdin is not None:
                try:
                    write_frame(proc.stdin, {"op": "close"})
                    if proc.stdout is not None:
                        read_frame_fd(proc.stdout.fileno(), timeout_s=2.0)
                    proc.wait(timeout=2.0)
                except Exception:
                    pass
            self._terminate_worker()
