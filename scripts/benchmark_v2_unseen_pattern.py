"""Forward-only paired optimizer for the frozen V2 foreleg benchmark."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from sensory_translator_v2.contracts import (
    LF_ACTUATOR_COLUMNS,
    LF_ACTUATOR_NAMES,
    LF_JOINT_NAMES,
    BackendSessionError,
    Evaluation,
    MultiJointEvaluationRequest,
    Session,
    require_real_float,
    validate_vector,
)
from sensory_translator_v2.goals import validate_multijoint_goal

if __package__:
    from .benchmark_v2_unseen_solver import DEFAULT_THRESHOLD_DEG, UnseenBenchmarkSolver, _rmse_deg
else:
    from benchmark_v2_unseen_solver import DEFAULT_THRESHOLD_DEG, UnseenBenchmarkSolver, _rmse_deg


INITIAL_STEP_HZ = 6.0
MINIMUM_STEP_HZ = 0.375


def _directions(shape: tuple[int, int], seed: int) -> list[tuple[str, np.ndarray]]:
    """Use the same smooth a priori basis and order for the paired methods."""
    muscles, samples = shape
    phase = np.pi * (np.arange(samples) + 0.5) / samples
    temporal = np.stack([np.cos(harmonic * phase) for harmonic in range(3)])
    rng = np.random.default_rng(seed)
    spatial = np.concatenate([np.ones((1, muscles)), rng.choice([-1.0, 1.0], (3, muscles))])
    directions = [
        (f"global{index}_cos{harmonic}", np.outer(pattern, waveform))
        for harmonic, waveform in enumerate(temporal)
        for index, pattern in enumerate(spatial)
    ]
    muscle_order = rng.permutation(muscles)
    for harmonic, waveform in enumerate(temporal):
        for muscle in muscle_order:
            direction = np.zeros(shape)
            direction[muscle] = waveform
            directions.append((f"{LF_ACTUATOR_NAMES[muscle]}_cos{harmonic}", direction))
    return directions


class UnseenPatternSolver(UnseenBenchmarkSolver):
    """Run projected pattern search with actual physics at every trial point."""

    def _warm_worker(self) -> dict[str, Any]:
        start = time.perf_counter()
        controls = self._controls(self.base_knots)
        with self._session(0) as session:
            forward_start = time.perf_counter()
            forward = self.backend.evaluate_multijoint(
                session, MultiJointEvaluationRequest(controls=controls, request_id="warm_forward"),
            )
            forward_s = time.perf_counter() - forward_start
            self.engine.predict_nominal(self._positions(forward))
        return {"setup_elapsed_s": time.perf_counter() - start, "forward_s": forward_s}

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
        largest_forward_s = self.warmup["forward_s"]
        best: Evaluation | None = None
        best_knots: np.ndarray | None = None
        best_rmse: float | None = None
        initial_rmse: float | None = None
        time_first_pass: float | None = None
        failure: str | None = None
        updates = 0
        initialization_s: float | None = None
        stop_reason = "time_budget"
        start = time.perf_counter()

        def elapsed() -> float:
            return time.perf_counter() - start

        def evaluate(
            session: Session, controls: np.ndarray, stage: str, knots: np.ndarray,
        ) -> bool:
            nonlocal largest_forward_s, best, best_knots, best_rmse, time_first_pass, updates
            began = elapsed()
            reserve = 1.05 * largest_forward_s
            if began + reserve > budget:
                history.append({
                    "stage": stage, "kind": "forward", "elapsed_s": began,
                    "status": "insufficient_time_for_call", "estimated_required_s": reserve,
                })
                return False
            evaluation = self.backend.evaluate_multijoint(
                session,
                MultiJointEvaluationRequest(
                    controls=controls, initial_qpos_delta_rad=delta, request_id=stage,
                ),
            )
            error = self._positions(evaluation) - goal
            rmse = _rmse_deg(error)
            completed = elapsed()
            largest_forward_s = max(largest_forward_s, completed - began)
            in_budget = completed <= budget
            accepted = in_budget and (best_rmse is None or rmse < best_rmse)
            history.append({
                "stage": stage, "kind": "forward", "started_s": began, "elapsed_s": completed,
                "status": "scored" if in_budget else "completed_after_budget",
                "rmse_deg": rmse, "gradient_valid": None, "direction_checks": None,
                "accepted": accepted,
            })
            if accepted:
                updates += int(best is not None)
                best, best_knots, best_rmse = evaluation, knots.copy(), rmse
                if time_first_pass is None and rmse <= threshold:
                    time_first_pass = completed
            return in_budget

        try:
            if method == "model":
                knots, _, controls = self.engine.predict_nominal(goal, initial_qpos_delta_rad=delta)
            else:
                knots = self.base_knots.copy()
                controls = self._controls(knots)
            directions = _directions(knots.shape, seed)
            initialization_s = elapsed()

            with self._session(seed) as session:
                in_budget = evaluate(session, controls, "initial", knots)
                initial_rmse = best_rmse
                sweep, step_hz = 0, INITIAL_STEP_HZ
                while in_budget and step_hz >= MINIMUM_STEP_HZ:
                    start_updates = updates
                    for label, direction in directions:
                        assert best_knots is not None
                        anchor = best_knots
                        for sign in (1.0, -1.0):
                            proposal = np.clip(anchor + sign * step_hz * direction, 0.0, self.max_hz)
                            if np.array_equal(proposal, anchor):
                                continue
                            candidate_controls = self._controls(proposal)
                            stage = f"sweep_{sweep}_{step_hz:g}hz_{label}_{sign:+g}"
                            in_budget = evaluate(session, candidate_controls, stage, proposal)
                            if not in_budget:
                                break
                        if not in_budget:
                            break
                    if updates == start_updates:
                        step_hz *= 0.5
                    sweep += 1
                if in_budget:
                    stop_reason = "step_floor"
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
            "optimizer": "projected_smooth_pattern_search",
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
