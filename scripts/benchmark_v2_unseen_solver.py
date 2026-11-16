"""Run a paired, wall-time-limited comparison on the frozen V2 foreleg plant."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from sensory_translator_v2.contracts import (
    LF_ACTUATOR_COLUMNS,
    LF_JOINT_NAMES,
    LF_QPOS_ADDRESSES,
    TRACE_KEYS,
    BackendSessionError,
    Evaluation,
    InitialState,
    MultiJointCapableBackend,
    MultiJointEvaluationRequest,
    Session,
    SessionSpec,
    require_real_float,
    validate_vector,
)
from sensory_translator_v2.engine import MultiJointEngine
from sensory_translator_v2.goals import validate_multijoint_goal
from sensory_translator_v2.rate.bridge import (
    direct_multichannel_rate_waveform,
    multichannel_rate_to_excitation,
)
from sensory_translator_v2.rate.optim import lbfgs_direction

INITIAL_STEP_RMS_HZ = 1.0
MAX_STEP_RMS_HZ = 12.0
LINE_SEARCH_FACTORS = (2.0, 1.0, 0.5, 0.25, 0.1)
DEFAULT_THRESHOLD_DEG = float(np.degrees(0.1))


def _rmse_deg(error_rad: np.ndarray) -> float:
    return float(np.degrees(np.sqrt(np.mean(error_rad**2))))


class UnseenBenchmarkSolver:
    """Compare initializers with a shared projected L-BFGS and physics worker.

    Construct once before the trials. Asset validation, checkpoint loading,
    worker startup, and a synthetic warm-up do not consume a trial budget.
    """

    def __init__(self, engine: MultiJointEngine) -> None:
        engine._ensure_init()
        backend, catalog, model = engine._backend, engine._catalog, engine._model
        if not isinstance(backend, MultiJointCapableBackend) or catalog is None or model is None:
            raise ValueError("The benchmark requires a loaded multi-joint engine")
        identity = backend.describe()
        if not identity.is_certified_frozen_physics:
            raise ValueError("The benchmark requires the registered MuJoCo plant")
        self.engine = engine
        self.backend = backend
        self.spec = SessionSpec(
            profile=engine.config.profile,
            asset_manifest_sha256=catalog.manifest_sha256,
            teacher_sha256=catalog.teacher_sha256,
            raw_xml_sha256=catalog.raw_xml_sha256,
            profile_rule_sha256=catalog.three_hash_record(engine.config.profile)["profile_rule_sha256"],
            cache_mode=identity.cache_mode,
        )
        self.max_hz, self.tau_ms, self.rate_scale_hz, self.clip_mode = engine._bridge_params()
        self.base_knots = model.base_knots_hz.detach().cpu().numpy().copy()
        self.base_knots = np.clip(self.base_knots, 0.0, self.max_hz)
        self.warmup = self._warm_worker()

    def _excitation(self, knots: torch.Tensor) -> torch.Tensor:
        waveform = direct_multichannel_rate_waveform(
            knots.clamp(0.0, self.max_hz), max_rate_hz=self.max_hz,
        )
        return multichannel_rate_to_excitation(
            waveform,
            max_rate_hz=self.max_hz,
            rate_scale_hz=self.rate_scale_hz,
            tau_ms=self.tau_ms,
            clip_boundary_gradient=self.clip_mode,
        )

    def _controls(self, knots: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self._excitation(torch.from_numpy(knots[None, :, :]))[0].numpy()

    def _gradient(self, knots: np.ndarray, control_vjp: np.ndarray) -> np.ndarray:
        state = torch.tensor(knots[None, :, :], dtype=torch.float64, requires_grad=True)
        excitation = self._excitation(state)
        excitation.backward(torch.from_numpy(control_vjp[None, :, :]))
        if state.grad is None or not torch.isfinite(state.grad).all():
            raise BackendSessionError("Physical VJP did not yield a finite knot gradient")
        return state.grad[0].detach().numpy().copy()

    @contextmanager
    def _session(self, seed: int):
        session = self.backend.open_session(self.spec)
        try:
            self.backend.reset(session, InitialState(0.0), seed=seed)
            yield session
        finally:
            self.backend.close_session(session)

    @staticmethod
    def _positions(evaluation: Evaluation) -> np.ndarray:
        return evaluation.traces["qpos"][:, LF_QPOS_ADDRESSES]

    @staticmethod
    def _same_traces(first: Evaluation, second: Evaluation) -> bool:
        return all(
            np.allclose(first.traces[key], second.traces[key], atol=1e-12, rtol=1e-12)
            for key in TRACE_KEYS
        )

    def _warm_worker(self) -> dict[str, Any]:
        start = time.perf_counter()
        controls = self._controls(self.base_knots)
        with self._session(0) as session:
            t0 = time.perf_counter()
            forward = self.backend.evaluate_multijoint(
                session, MultiJointEvaluationRequest(controls=controls, request_id="warm_forward"),
            )
            forward_s = time.perf_counter() - t0
            positions = self._positions(forward)
            self.engine.predict_nominal(positions)
            # Timing and autograd setup only. The synthetic target and its gradient
            # never enter the benchmark. It uses the shared nominal plan's rollout.
            t0 = time.perf_counter()
            gradient_run = self.backend.evaluate_multijoint(
                session,
                MultiJointEvaluationRequest(
                    controls=controls,
                    need_jacobian=True,
                    mse_target_rad=positions + 0.01,
                    check_directions=True,
                    clip_boundary_gradient=self.clip_mode,
                    request_id="warm_vjp",
                ),
            )
            vjp_s = time.perf_counter() - t0
            if gradient_run.control_vjp is not None:
                self._gradient(self.base_knots, gradient_run.control_vjp)
        return {
            "setup_elapsed_s": time.perf_counter() - start,
            "forward_s": forward_s,
            "checked_vjp_s": vjp_s,
            "warm_vjp_direction_gate_passed": gradient_run.gradient_valid,
        }

    def solve(
        self,
        target_rad: np.ndarray,
        *,
        method: str,
        budget_s: float,
        threshold_deg: float = DEFAULT_THRESHOLD_DEG,
        initial_delta_rad: np.ndarray | None = None,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Score only complete in-budget rollouts, then audit the winner in a new session."""
        if method not in {"model", "direct"}:
            raise ValueError("method must be 'model' or 'direct'")
        budget = require_real_float(budget_s, "budget_s", positive=True)
        threshold = require_real_float(threshold_deg, "threshold_deg", positive=True)
        goal, _ = validate_multijoint_goal(target_rad, n_joints=len(LF_JOINT_NAMES))
        delta = (
            np.zeros(len(LF_JOINT_NAMES), dtype=np.float64)
            if initial_delta_rad is None
            else validate_vector(initial_delta_rad, "initial_delta_rad", length=len(LF_JOINT_NAMES))
        )
        history: list[dict[str, Any]] = []
        expected_s = {
            "forward": self.warmup["forward_s"],
            "checked_vjp": self.warmup["checked_vjp_s"],
        }
        best: Evaluation | None = None
        best_knots: np.ndarray | None = None
        best_controls: np.ndarray | None = None
        best_rmse: float | None = None
        initial_rmse: float | None = None
        time_first_pass: float | None = None
        failure: str | None = None
        stop_reason = "time_budget"
        updates = 0
        s_hist: list[np.ndarray] = []
        y_hist: list[np.ndarray] = []
        previous_knots: np.ndarray | None = None
        previous_gradient: np.ndarray | None = None
        initialization_s: float | None = None
        start = time.perf_counter()

        def elapsed() -> float:
            return time.perf_counter() - start

        def evaluate(
            session: Session, controls: np.ndarray, stage: str, *, need_gradient: bool = False,
        ) -> Evaluation | None:
            kind = "checked_vjp" if need_gradient else "forward"
            # Native evaluations are synchronous. Guard dispatch with the warm and
            # observed timings. A late return is logged and never changes selection.
            began = elapsed()
            reserve = 1.05 * expected_s[kind]
            if began + reserve > budget:
                history.append({
                    "stage": stage,
                    "kind": kind,
                    "elapsed_s": began,
                    "status": "insufficient_time_for_call",
                    "estimated_required_s": reserve,
                })
                return None
            request = MultiJointEvaluationRequest(
                controls=controls,
                initial_qpos_delta_rad=delta,
                need_jacobian=need_gradient,
                mse_target_rad=goal if need_gradient else None,
                check_directions=need_gradient,
                clip_boundary_gradient=self.clip_mode,
                request_id=stage,
            )
            evaluation = self.backend.evaluate_multijoint(session, request)
            rmse = _rmse_deg(self._positions(evaluation) - goal)
            completed = elapsed()
            expected_s[kind] = max(expected_s[kind], completed - began)
            passed_deadline = completed > budget
            history.append({
                "stage": stage,
                "kind": kind,
                "started_s": began,
                "elapsed_s": completed,
                "status": "completed_after_budget" if passed_deadline else "scored",
                "rmse_deg": rmse,
                "gradient_valid": evaluation.gradient_valid,
                "direction_checks": evaluation.direction_checks,
                "accepted": False,
            })
            return None if passed_deadline else evaluation

        def accept(evaluation: Evaluation, knots: np.ndarray, controls: np.ndarray) -> None:
            nonlocal best, best_rmse, best_knots, best_controls, time_first_pass
            best = evaluation
            best_rmse = history[-1]["rmse_deg"]
            best_knots = knots.copy()
            best_controls = controls.copy()
            history[-1]["accepted"] = True
            if time_first_pass is None and best_rmse <= threshold:
                time_first_pass = history[-1]["elapsed_s"]

        try:
            if method == "model":
                knots, _, controls = self.engine.predict_nominal(goal, initial_qpos_delta_rad=delta)
            else:
                knots = self.base_knots.copy()
                controls = self._controls(knots)
            initialization_s = elapsed()

            with self._session(seed) as session:
                first = evaluate(session, controls, "initial")
                if first is not None:
                    accept(first, knots, controls)
                    initial_rmse = best_rmse

                while best is not None:
                    assert best_knots is not None and best_controls is not None and best_rmse is not None
                    derivative = evaluate(session, best_controls, f"vjp_{updates}", need_gradient=True)
                    if derivative is None:
                        break
                    if not self._same_traces(best, derivative):
                        stop_reason = "gradient_trace_mismatch"
                        break
                    if derivative.gradient_valid is not True or derivative.control_vjp is None:
                        stop_reason = "gradient_gate_rejected"
                        break
                    gradient = self._gradient(best_knots, derivative.control_vjp)
                    if previous_knots is not None and previous_gradient is not None:
                        s = (best_knots - previous_knots).ravel()
                        y = (gradient - previous_gradient).ravel()
                        if float(np.dot(y, s)) > 1e-20:
                            s_hist.append(s)
                            y_hist.append(y)
                            del s_hist[:-5]
                            del y_hist[:-5]
                    direction = lbfgs_direction(
                        gradient, s_hist, y_hist,
                        init_step_rms_hz=INITIAL_STEP_RMS_HZ,
                        max_step_rms_hz=MAX_STEP_RMS_HZ,
                    )
                    improved = False
                    for factor in LINE_SEARCH_FACTORS:
                        candidate_knots = np.clip(best_knots + factor * direction, 0.0, self.max_hz)
                        if np.array_equal(candidate_knots, best_knots):
                            continue
                        candidate_controls = self._controls(candidate_knots)
                        candidate = evaluate(session, candidate_controls, f"ls_{updates}_{factor}")
                        if candidate is None:
                            break
                        if history[-1]["rmse_deg"] < best_rmse:
                            previous_knots, previous_gradient = best_knots, gradient
                            accept(candidate, candidate_knots, candidate_controls)
                            updates += 1
                            improved = True
                            break
                    if not improved:
                        if history[-1]["status"] == "scored":
                            stop_reason = "line_search_stalled"
                        break
        except BackendSessionError as exc:
            failure = str(exc)
            stop_reason = "backend_failure"

        search_s = elapsed()
        replay_s = None
        replay_verified = False
        per_joint_rmse = None
        per_joint_max_abs_error = None
        selected_qpos = None
        if best is not None:
            replay_start = time.perf_counter()
            try:
                with self._session(seed) as session:
                    replay = self.backend.evaluate_multijoint(
                        session,
                        MultiJointEvaluationRequest(
                            controls=best.traces["interval_applied_ctrl"][:, LF_ACTUATOR_COLUMNS],
                            initial_qpos_delta_rad=delta,
                            request_id="independent_replay",
                        ),
                    )
                    replay_verified = self._same_traces(best, replay)
                    if not replay_verified:
                        failure = "Independent replay diverged from the selected candidate"
            except BackendSessionError as exc:
                failure = f"Independent replay failed: {exc}"
            replay_s = time.perf_counter() - replay_start
            positions = self._positions(best)
            selected_qpos = positions.tolist()
            error_deg = np.degrees(positions - goal)
            per_joint_rmse = dict(zip(LF_JOINT_NAMES, np.sqrt(np.mean(error_deg**2, axis=0)).tolist()))
            per_joint_max_abs_error = dict(zip(LF_JOINT_NAMES, np.max(np.abs(error_deg), axis=0).tolist()))

        return {
            "method": method,
            "initializer": "conditioned_checkpoint" if method == "model" else "fixed_checkpoint_base_knots",
            "budget_s": budget,
            "initialization_s": initialization_s,
            "search_elapsed_s": search_s,
            "search_overshoot_s": max(0.0, search_s - budget),
            "independent_replay_elapsed_s": replay_s,
            "independent_replay_verified": replay_verified,
            "initial_rmse_deg": initial_rmse,
            "best_rmse_deg": best_rmse,
            "per_joint_rmse_deg": per_joint_rmse,
            "per_joint_max_abs_error_deg": per_joint_max_abs_error,
            "accuracy_threshold_deg": threshold,
            "accuracy_threshold_met": bool(replay_verified and best_rmse is not None and best_rmse <= threshold),
            "time_to_first_pass_s": time_first_pass,
            "optimizer_updates": updates,
            "stop_reason": stop_reason,
            "failure": failure,
            "history": history,
            "selected_knots_hz": best_knots.tolist() if best_knots is not None else None,
            "selected_qpos_rad": selected_qpos,
        }
