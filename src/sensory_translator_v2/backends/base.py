"""Shared backend session helpers and non-certified FakeBackend for unit testing."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..contracts import (
    CONTROL_STEPS,
    CONTROL_TIMESTEP_S,
    ClosedLoopRequest,
    Evaluation,
    EvaluationRequest,
    InitialState,
    MultiJointClosedLoopRequest,
    MultiJointEvaluationRequest,
    NA,
    NQ,
    NU,
    NV,
    STATE_STEPS,
    TARGET_ACTUATOR_COLUMNS,
    TARGET_QPOS_ADDRESS,
    Backend,
    BackendIdentity,
    BackendSessionError,
    BackendTimeoutError,
    Session,
    SessionSpec,
    require_int,
)
from ..evidence import utc_now_iso


def validate_session_spec_against_catalog(
    spec: SessionSpec,
    catalog: Any,
    *,
    expected_profile: str | None = None,
) -> dict[str, Any]:
    """Validate SessionSpec profile and any supplied asset hashes against catalog."""
    if expected_profile is not None and spec.profile != expected_profile:
        raise BackendSessionError(
            f"Backend configured for profile {expected_profile!r}, cannot open session for {spec.profile!r}"
        )
    hashes = catalog.three_hash_record(spec.profile)
    checks = (
        ("asset_manifest_sha256", spec.asset_manifest_sha256, catalog.manifest_sha256),
        ("teacher_sha256", spec.teacher_sha256, hashes.get("teacher_sha256")),
        ("raw_xml_sha256", spec.raw_xml_sha256, hashes.get("raw_xml_sha256")),
        ("profile_rule_sha256", spec.profile_rule_sha256, hashes.get("profile_rule_sha256")),
    )
    for field_name, supplied, expected in checks:
        if supplied and supplied != expected:
            raise BackendSessionError(
                f"SessionSpec {field_name} mismatch: expected {expected!r}, got {supplied!r}"
            )
    return hashes


def require_open_session(session: Session, sessions: dict[str, Session]) -> Session:
    """Ensure session belongs to this backend and is still open."""
    if not isinstance(session, Session) or session.session_id not in sessions:
        raise BackendSessionError(f"Unknown backend session: {getattr(session, 'session_id', None)!r}")
    tracked = sessions[session.session_id]
    if tracked.closed or session.closed:
        raise BackendSessionError(f"Backend session {session.session_id!r} is already closed")
    return tracked


def require_reset_session(session: Session, sessions: dict[str, Session]) -> Session:
    """Ensure session is open and has an active reset InitialState before evaluate()."""
    tracked = require_open_session(session, sessions)
    if tracked.active_initial is None or tracked.active_seed is None:
        raise BackendSessionError(
            f"Backend session {session.session_id!r} must call reset() before evaluate()"
        )
    return tracked


class FakeBackend(Backend):
    """Fast analytical test backend that explicitly marks is_certified_frozen_physics=False."""

    def __init__(
        self,
        *,
        profile: str = "adjacent_static_parent_v1",
        teacher_joint_rad: np.ndarray | None = None,
        delay_s: float = 0.0,
        fail_after_evals: int | None = None,
        timeout_on_eval: int | None = None,
    ) -> None:
        self.profile = profile
        if teacher_joint_rad is None:
            t = np.linspace(0.0, 0.42, STATE_STEPS, dtype=np.float64)
            self._teacher_q = 2.03 - 0.65 * np.sin(np.pi * t / 0.42) ** 2
        else:
            self._teacher_q = np.asarray(teacher_joint_rad, dtype=np.float64).reshape(STATE_STEPS)
        self._delay_s = float(delay_s)
        self._fail_after_evals = fail_after_evals
        self._timeout_on_eval = timeout_on_eval
        self._lock = threading.RLock()
        self._sessions: dict[str, Session] = {}
        self._closed = False
        self._total_evals = 0

    def describe(self) -> BackendIdentity:
        return BackendIdentity(
            name="fake",
            profile=self.profile,
            backend_version="2.0.0-fake",
            is_certified_frozen_physics=False,
            cache_mode="fake",
            asset_hashes={"raw_xml_sha256": None, "teacher_sha256": None},
            runtime_versions={"backend": "analytical_test_fake"},
            details={
                "note": "Non-physical test backend; never certified frozen MuJoCo physics.",
                "supports_closed_loop": True,
                "supports_multijoint": True,
            },
        )

    def open_session(self, spec: SessionSpec) -> Session:
        with self._lock:
            if self._closed:
                raise BackendSessionError("FakeBackend is closed")
            session_id = uuid.uuid4().hex
            identity = BackendIdentity(
                name="fake",
                profile=spec.profile,
                backend_version="2.0.0-fake",
                is_certified_frozen_physics=False,
                cache_mode="fake",
            )
            session = Session(
                session_id=session_id,
                backend_name="fake",
                spec=spec,
                identity=identity,
                created_utc=utc_now_iso(),
            )
            self._sessions[session_id] = session
            return session

    def reset(self, session: Session, initial: InitialState, seed: int) -> None:
        with self._lock:
            if self._closed:
                raise BackendSessionError("FakeBackend is closed")
            tracked = require_open_session(session, self._sessions)
            require_int(seed, "seed", minimum=0)
            tracked.active_initial = initial
            tracked.active_seed = seed
            session.active_initial = initial
            session.active_seed = seed

    def _begin_fake_eval(self, session: Session) -> Session:
        if self._closed:
            raise BackendSessionError("FakeBackend is closed")
        tracked = require_reset_session(session, self._sessions)
        self._total_evals += 1
        if self._delay_s > 0.0:
            time.sleep(self._delay_s)
        if self._timeout_on_eval is not None and self._total_evals == self._timeout_on_eval:
            tracked.closed = True
            session.closed = True
            raise BackendTimeoutError("Simulated FakeBackend timeout")
        if self._fail_after_evals is not None and self._total_evals >= self._fail_after_evals:
            tracked.closed = True
            session.closed = True
            raise BackendSessionError("Simulated FakeBackend crash")
        return tracked

    def _simulate_traces(self, controls: np.ndarray, delta: float) -> tuple[dict[str, np.ndarray], np.ndarray]:
        diff = controls[:, 0] - 0.8 * controls[:, 1]
        t_grid = np.arange(STATE_STEPS, dtype=np.float64) * CONTROL_TIMESTEP_S
        q_traj = np.zeros(STATE_STEPS, dtype=np.float64)
        q_traj[0] = self._teacher_q[0] + delta
        vel = np.zeros(STATE_STEPS, dtype=np.float64)
        # q_traj[t] = q_traj[t-1] + baseline_step + drive
        for t in range(1, STATE_STEPS):
            baseline_step = self._teacher_q[t] - self._teacher_q[t - 1]
            drive = -0.025 * diff[t - 1]
            vel[t] = (baseline_step + drive) / CONTROL_TIMESTEP_S
            q_traj[t] = q_traj[t - 1] + baseline_step + drive

        qpos = np.zeros((STATE_STEPS, NQ), dtype=np.float64)
        qvel = np.zeros((STATE_STEPS, NV), dtype=np.float64)
        act = np.zeros((STATE_STEPS, NA), dtype=np.float64)
        ctrl_state = np.zeros((STATE_STEPS, NU), dtype=np.float64)
        applied = np.zeros((CONTROL_STEPS, NU), dtype=np.float64)
        qpos[:, TARGET_QPOS_ADDRESS] = q_traj
        qvel[:, TARGET_QPOS_ADDRESS] = vel
        ctrl_state[1:, list(TARGET_ACTUATOR_COLUMNS)] = controls
        applied[:, list(TARGET_ACTUATOR_COLUMNS)] = controls
        act[1:, list(TARGET_ACTUATOR_COLUMNS)] = controls
        traces = {
            "qpos": qpos,
            "qvel": qvel,
            "actuator_activation": act,
            "state_actuator_ctrl": ctrl_state,
            "time_s": t_grid,
            "interval_applied_ctrl": applied,
        }
        return traces, q_traj

    def evaluate(self, session: Session, request: EvaluationRequest) -> Evaluation:
        with self._lock:
            tracked = self._begin_fake_eval(session)
            assert tracked.active_initial is not None
            delta = float(tracked.active_initial.initial_tibia_delta_rad)
            controls = request.controls  # [210, 2]
            traces, q_traj = self._simulate_traces(controls, delta)

            control_vjp = None
            direction_checks = None
            gradient_valid = None
            if request.need_jacobian:
                if request.mse_target_rad is not None:
                    cot = (2.0 / STATE_STEPS) * (q_traj - request.mse_target_rad)
                else:
                    cot = request.cotangent_qpos
                assert cot is not None
                # Exact analytical derivative matching dq[t]/du[k] = -0.025*(+1 for flex, -0.8 for ext) for t >= k+1
                suffix_sum = np.cumsum(cot[::-1])[::-1]
                # For control interval k (0..209), it affects states t = k+1 .. 210
                d_diff = -0.025 * suffix_sum[1:]
                vjp = np.zeros((CONTROL_STEPS, 2), dtype=np.float64)
                vjp[:, 0] = d_diff
                vjp[:, 1] = -0.8 * d_diff
                clip_mask = ((controls > 0.0) & (controls < 1.0)).astype(np.float64)
                vjp *= clip_mask
                control_vjp = vjp
                gradient_valid = True
                if request.check_directions:
                    max_eps = tracked.spec.direction_eps
                    rng = np.random.default_rng(request.direction_seed)
                    checks = []
                    for idx_dir, label in enumerate(("gradient", "random")):
                        raw_d = vjp.copy() if idx_dir == 0 else rng.normal(size=(CONTROL_STEPS, 2))
                        raw_d *= clip_mask
                        norm = float(np.linalg.norm(raw_d))
                        if norm == 0.0:
                            checks.append({"kind": label, "status": "zero_direction", "passed": False})
                            continue
                        unit_d = raw_d / norm
                        active = np.abs(unit_d) > 0.0
                        margins = np.minimum(controls - 0.0, 1.0 - controls)
                        eps = min(float(max_eps), float(np.min(margins[active] / np.abs(unit_d[active]))) * 0.25)

                        def _loss(u: np.ndarray) -> float:
                            dq = np.zeros(STATE_STEPS, dtype=np.float64)
                            dq[0] = q_traj[0]
                            dv = u[:, 0] - 0.8 * u[:, 1]
                            for idx in range(1, STATE_STEPS):
                                dq[idx] = dq[idx - 1] + (self._teacher_q[idx] - self._teacher_q[idx - 1]) - 0.025 * dv[idx - 1]
                            if request.mse_target_rad is not None:
                                return float(np.mean((dq - request.mse_target_rad) ** 2))
                            return float(np.dot(dq, cot))

                        loss_minus = _loss(controls - eps * unit_d)
                        loss_plus = _loss(controls + eps * unit_d)
                        fd_rollout = (loss_plus - loss_minus) / (2.0 * eps)
                        adj_val = float(np.sum(vjp * unit_d))
                        abs_err = abs(fd_rollout - adj_val)
                        checks.append({
                            "kind": label,
                            "status": "evaluated",
                            "epsilon": eps,
                            "finite_difference": fd_rollout,
                            "adjoint": adj_val,
                            "absolute_error": abs_err,
                            "passed": bool(abs_err <= 1e-6 + 0.005 * max(abs(fd_rollout), abs(adj_val))),
                        })
                    direction_checks = {"all_passed": all(c["passed"] for c in checks), "checks": checks}
                    gradient_valid = bool(direction_checks["all_passed"])

            tracked.evaluation_count += 1
            session.evaluation_count = tracked.evaluation_count
            return Evaluation(
                request_id=request.request_id or uuid.uuid4().hex,
                session_id=tracked.session_id,
                profile=tracked.spec.profile,
                is_certified_frozen_physics=False,
                traces=traces,
                joint_rad=q_traj,
                control_vjp=control_vjp,
                gradient_valid=gradient_valid,
                direction_checks=direction_checks,
                timing_s={"total_s": 0.0},
                metadata={"seed": tracked.active_seed, "delta_rad": delta},
            )

    def evaluate_closed_loop(self, session: Session, request: ClosedLoopRequest) -> Evaluation:
        with self._lock:
            tracked = self._begin_fake_eval(session)
            assert tracked.active_initial is not None
            delta = float(tracked.active_initial.initial_tibia_delta_rad)
            target_vel = np.gradient(request.target_rad, CONTROL_TIMESTEP_S)
            realized = np.empty((CONTROL_STEPS, 2), dtype=np.float64)
            q_cur = self._teacher_q[0] + delta
            qd_cur = 0.0
            for t in range(CONTROL_STEPS):
                err = float(request.target_rad[t + 1]) - q_cur
                derr = float(target_vel[t]) - qd_cur
                u_fb = request.kp * err + request.kd * derr
                base = request.feedforward_controls[t]
                e_flex = float(np.clip(base[0] + u_fb, 0.0, 1.0))
                e_ext = float(np.clip(base[1] - request.ext_to_flex_ratio * u_fb, 0.0, 1.0))
                realized[t, 0] = e_flex
                realized[t, 1] = e_ext
                baseline_step = self._teacher_q[t + 1] - self._teacher_q[t]
                drive = -0.025 * (e_flex - 0.8 * e_ext)
                qd_cur = (baseline_step + drive) / CONTROL_TIMESTEP_S
                q_cur = q_cur + baseline_step + drive
            traces, q_traj = self._simulate_traces(realized, delta)
            tracked.evaluation_count += 1
            session.evaluation_count = tracked.evaluation_count
            return Evaluation(
                request_id=request.request_id or uuid.uuid4().hex,
                session_id=tracked.session_id,
                profile=tracked.spec.profile,
                is_certified_frozen_physics=False,
                traces=traces,
                joint_rad=q_traj,
                timing_s={"total_s": 0.0},
                metadata={"seed": tracked.active_seed, "delta_rad": delta, "closed_loop": True},
            )

    def _simulate_multijoint_traces(
        self,
        controls: np.ndarray,
        control_columns: tuple[int, ...],
        target_qpos_indices: tuple[int, ...],
        initial_qpos_delta_rad: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        t_grid = np.arange(STATE_STEPS, dtype=np.float64) * CONTROL_TIMESTEP_S
        qpos = np.zeros((STATE_STEPS, NQ), dtype=np.float64)
        qvel = np.zeros((STATE_STEPS, NV), dtype=np.float64)
        act = np.zeros((STATE_STEPS, NA), dtype=np.float64)
        ctrl_state = np.zeros((STATE_STEPS, NU), dtype=np.float64)
        applied = np.zeros((CONTROL_STEPS, NU), dtype=np.float64)
        applied[:, list(control_columns)] = controls
        ctrl_state[1:, list(control_columns)] = controls
        act[1:, list(control_columns)] = controls

        # Simple deterministic 15->7 coupled analytical integrator for unit tests
        qpos[:, TARGET_QPOS_ADDRESS] = self._teacher_q
        if initial_qpos_delta_rad is not None:
            for pos, adr in enumerate(target_qpos_indices):
                qpos[0, adr] += float(initial_qpos_delta_rad[pos])
        for t in range(1, STATE_STEPS):
            qpos[t, :7] = qpos[t - 1, :7]
            for pos, adr in enumerate(control_columns):
                j_adr = adr // 2 if adr < 12 else (5 if adr == 12 else 6)
                sign = 1.0 if (adr % 2 == 0) else -0.8
                if j_adr < 7:
                    qpos[t, j_adr] += -0.015 * sign * float(controls[t - 1, pos])
            qvel[t, :7] = (qpos[t, :7] - qpos[t - 1, :7]) / CONTROL_TIMESTEP_S
        return {
            "qpos": qpos,
            "qvel": qvel,
            "actuator_activation": act,
            "state_actuator_ctrl": ctrl_state,
            "time_s": t_grid,
            "interval_applied_ctrl": applied,
        }

    def evaluate_multijoint(
        self,
        session: Session,
        request: MultiJointEvaluationRequest,
    ) -> Evaluation:
        with self._lock:
            tracked = self._begin_fake_eval(session)
            traces = self._simulate_multijoint_traces(
                request.controls,
                request.control_columns,
                request.target_qpos_indices,
                request.initial_qpos_delta_rad,
            )
            q_sub = traces["qpos"][:, list(request.target_qpos_indices)]
            control_vjp = None
            gradient_valid = None
            if request.need_jacobian:
                n_j = len(request.target_qpos_indices)
                if request.mse_target_rad is not None:
                    w = (
                        np.ones((n_j,), dtype=np.float64)
                        if request.joint_weights is None
                        else request.joint_weights / float(np.mean(request.joint_weights))
                    )
                    cot = (2.0 / (STATE_STEPS * n_j)) * w[None, :] * (q_sub - request.mse_target_rad)
                else:
                    assert request.cotangent_qpos is not None
                    cot = request.cotangent_qpos
                suffix = np.cumsum(cot[::-1], axis=0)[::-1]
                vjp = np.zeros((CONTROL_STEPS, len(request.control_columns)), dtype=np.float64)
                for pos, adr in enumerate(request.control_columns):
                    j_adr = adr // 2 if adr < 12 else (5 if adr == 12 else 6)
                    if j_adr in request.target_qpos_indices:
                        j_pos = request.target_qpos_indices.index(j_adr)
                        sign = 1.0 if (adr % 2 == 0) else -0.8
                        vjp[:, pos] = -0.015 * sign * suffix[1:, j_pos]
                if request.clip_boundary_gradient == "native":
                    vjp *= ((request.controls > 0.0) & (request.controls < 1.0)).astype(np.float64)
                else:
                    block_lo = (request.controls <= 0.0) & (vjp > 0.0)
                    block_hi = (request.controls >= 1.0) & (vjp < 0.0)
                    vjp[block_lo | block_hi] = 0.0
                control_vjp = vjp
                gradient_valid = True
            tracked.evaluation_count += 1
            session.evaluation_count = tracked.evaluation_count
            return Evaluation(
                request_id=request.request_id or uuid.uuid4().hex,
                session_id=tracked.session_id,
                profile=tracked.spec.profile,
                is_certified_frozen_physics=False,
                traces=traces,
                joint_rad=traces["qpos"][:, TARGET_QPOS_ADDRESS],
                control_vjp=control_vjp,
                gradient_valid=gradient_valid,
                timing_s={"total_s": 0.0},
                metadata={
                    "seed": tracked.active_seed,
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
            tracked = self._begin_fake_eval(session)
            # Deterministic feedback nudge toward target_rad in fake backend
            adj = request.feedforward_controls.copy()
            traces = self._simulate_multijoint_traces(
                adj,
                request.control_columns,
                request.target_qpos_indices,
                request.initial_qpos_delta_rad,
            )
            tracked.evaluation_count += 1
            session.evaluation_count = tracked.evaluation_count
            return Evaluation(
                request_id=request.request_id or uuid.uuid4().hex,
                session_id=tracked.session_id,
                profile=tracked.spec.profile,
                is_certified_frozen_physics=False,
                traces=traces,
                joint_rad=traces["qpos"][:, TARGET_QPOS_ADDRESS],
                timing_s={"total_s": 0.0},
                metadata={
                    "seed": tracked.active_seed,
                    "control_columns": list(request.control_columns),
                    "target_qpos_indices": list(request.target_qpos_indices),
                    "controlled_muscles": len(request.control_columns),
                    "teacher_assisted_muscles": 15 - len(request.control_columns),
                    "closed_loop": True,
                },
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
