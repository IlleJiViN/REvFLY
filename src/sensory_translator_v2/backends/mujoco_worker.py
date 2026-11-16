"""Self-contained persistent MuJoCo physics worker process (M2 fresh + M3 cached MjModel).

Launch via: `<physics_python> -m sensory_translator_v2.backends.mujoco_worker`
All stdout noise (including C-level warnings) is redirected to stderr before importing
dm_control/mujoco so the binary IPC pipe on the original stdout fd remains uncorrupted.
"""

from __future__ import annotations

import copy
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .worker_protocol import PROTOCOL_VERSION, read_frame_fd, write_frame

# Constants pinned identically to the frozen FlyMimic 420ms LF tibia contract.
SUBSTEPS = 20
CONTROL_STEPS = 210
STATE_STEPS = 211
PHYSICS_TIMESTEP_S = 0.0001
CONTROL_TIMESTEP_S = SUBSTEPS * PHYSICS_TIMESTEP_S
NQ = 14
NV = 14
NU = 15
NA = 15
NSTATE = 43
TARGET_COLUMNS = [13, 14]
TARGET_NAMES = ["LFTibia_flex_93434", "LFTibia_extensor_93932"]
TARGET_JOINT = "joint_LFTibia_pitch"
TARGET_QPOS_ADDRESS = 6
EXPECTED_TEACHER_SHA256 = "285926477784ac9ea1657f2d422642696bc3b4cd5866c7e2235e7fde964a5ebd"

TRACE_KEYS = (
    "qpos",
    "qvel",
    "actuator_activation",
    "state_actuator_ctrl",
    "time_s",
    "interval_applied_ctrl",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_teacher(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    actual_sha = _sha256_file(path)
    if expected_sha256 and actual_sha != expected_sha256:
        raise ValueError(f"Teacher SHA256 mismatch: expected {expected_sha256}, got {actual_sha}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _configure_model_for_profile(dm_mujoco: Any, native_mujoco: Any, model: Any, profile: str) -> None:
    if (model.nq, model.nv, model.nu, model.na) != (NQ, NV, NU, NA):
        raise ValueError(f"Unexpected MuJoCo dimensions: {(model.nq, model.nv, model.nu, model.na)}")
    if float(model.opt.timestep) != PHYSICS_TIMESTEP_S:
        raise ValueError(f"Unexpected timestep: {model.opt.timestep}")
    jid = int(model.name2id(TARGET_JOINT, "joint"))
    if int(model.jnt_qposadr[jid]) != TARGET_QPOS_ADDRESS or int(model.jnt_dofadr[jid]) != TARGET_QPOS_ADDRESS:
        raise ValueError("Unexpected target joint qpos/dof address")
    if [model.id2name(int(i), "actuator") for i in TARGET_COLUMNS] != TARGET_NAMES:
        raise ValueError("Unexpected target actuator ordering")
    if int(model.opt.integrator) != int(native_mujoco.mjtIntegrator.mjINT_EULER):
        raise ValueError("Frozen Euler integrator requirement failed")
    if profile in ("multiccd_v1", "adjacent_static_parent_v1"):
        model.opt.enableflags |= int(native_mujoco.mjtEnableBit.mjENBL_MULTICCD)
    if profile == "adjacent_static_parent_v1" and int(model.nexclude) < 2:
        raise ValueError(f"adjacent_static_parent_v1 expected >=2 contact excludes, got {model.nexclude}")


def _reset_physics_state(physics: Any, teacher: dict[str, np.ndarray], tibia_delta_rad: float) -> None:
    model = physics.model
    if not np.array_equal(np.asarray(model.actuator_actadr), teacher["actuator_actadr"]):
        raise ValueError("Unexpected activation state addresses")
    jid = int(model.name2id(TARGET_JOINT, "joint"))
    target_initial = float(teacher["qpos"][0, TARGET_QPOS_ADDRESS] + tibia_delta_rad)
    limits = np.asarray(model.jnt_range[jid], dtype=np.float64)
    if not limits[0] <= target_initial <= limits[1]:
        raise ValueError(
            f"Initial tibia position {target_initial} outside XML joint limits {limits.tolist()}"
        )
    with physics.reset_context():
        physics.data.qpos[:] = teacher["qpos"][0]
        physics.data.qpos[TARGET_QPOS_ADDRESS] = target_initial
        physics.data.qvel[:] = teacher["qvel"][0]
        physics.data.act[teacher["actuator_actadr"]] = teacher["actuator_activation"][0]
        physics.data.ctrl[:] = teacher["state_actuator_ctrl"][0]
        physics.data.time = float(teacher["time_s"][0])


def _compose_transition(prev_a: np.ndarray, prev_b: np.ndarray, step_a: np.ndarray, step_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return step_a @ prev_a, step_a @ prev_b + step_b


def _input_clip_mask(requested: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return ((requested > bounds[:, 0]) & (requested < bounds[:, 1])).astype(np.float64)


def _apply_input_boundary_gradient(
    requested: np.ndarray,
    raw_gradient: np.ndarray,
    bounds: np.ndarray,
    mode: str = "native",
) -> np.ndarray:
    """Apply native two-sided or one-sided inward-recovery input bound masking."""
    if mode == "native":
        return raw_gradient * _input_clip_mask(requested, bounds)
    if mode != "one_sided":
        raise ValueError(f"Unsupported clip_boundary_gradient mode: {mode!r}")
    out = raw_gradient.copy()
    block_low = (requested <= bounds[:, 0]) & (raw_gradient > 0.0)
    block_high = (requested >= bounds[:, 1]) & (raw_gradient < 0.0)
    out[block_low | block_high] = 0.0
    return out


def _adjoint_control_vjp(
    transitions: np.ndarray,
    control_maps: np.ndarray,
    qpos_cotangent: np.ndarray,
    *,
    input_clip_mask: np.ndarray,
) -> np.ndarray:
    lam = np.zeros(NSTATE, dtype=np.float64)
    lam[TARGET_QPOS_ADDRESS] = float(qpos_cotangent[-1])
    result = np.empty((CONTROL_STEPS, 2), dtype=np.float64)
    for idx in range(CONTROL_STEPS - 1, -1, -1):
        result[idx] = control_maps[idx].T @ lam
        lam = transitions[idx].T @ lam
        lam[TARGET_QPOS_ADDRESS] += float(qpos_cotangent[idx])
    result *= input_clip_mask
    if not np.isfinite(result).all():
        raise ValueError("Non-finite physical adjoint gradient")
    return result


class _PhysicsWorkerState:
    """Holds immutable verified FlyMimic XML/teacher metadata and optional cached MjModel."""

    def __init__(
        self,
        *,
        profile: str,
        xml_path: Path,
        teacher_path: Path,
        expected_teacher_sha256: str,
        cache_mode: str,
    ) -> None:
        from dm_control import mujoco as dm_mujoco
        import mujoco as native_mujoco

        self.dm_mujoco = dm_mujoco
        self.native_mujoco = native_mujoco
        self.profile = profile
        self.xml_path = xml_path
        self.teacher_path = teacher_path
        self.cache_mode = cache_mode
        self.teacher = _load_teacher(teacher_path, expected_teacher_sha256)
        self._cached_model: Any | None = None
        if cache_mode == "cached_model":
            proto_physics = dm_mujoco.Physics.from_xml_path(str(self.xml_path))
            _configure_model_for_profile(dm_mujoco, native_mujoco, proto_physics.model, profile)
            self._cached_model = proto_physics.model

    def instantiate_physics(self, tibia_delta_rad: float) -> Any:
        t0 = time.perf_counter()
        if self._cached_model is not None:
            physics = self.dm_mujoco.Physics.from_model(self._cached_model)
        else:
            physics = self.dm_mujoco.Physics.from_xml_path(str(self.xml_path))
            _configure_model_for_profile(
                self.dm_mujoco, self.native_mujoco, physics.model, self.profile
            )
        try:
            _reset_physics_state(physics, self.teacher, tibia_delta_rad)
            return physics, time.perf_counter() - t0
        except Exception:
            physics.free()
            raise

    def _run_stepped_rollout(
        self,
        tibia_delta_rad: float,
        step_control_fn: Any,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, float]]:
        physics, inst_s = self.instantiate_physics(tibia_delta_rad)
        t_step0 = time.perf_counter()
        try:
            bounds = np.asarray(physics.model.actuator_ctrlrange, dtype=np.float64)[TARGET_COLUMNS].copy()
            applied = self.teacher["interval_actuator_ctrl"].copy()
            qpos = np.empty((STATE_STEPS, NQ), dtype=np.float64)
            qvel = np.empty((STATE_STEPS, NV), dtype=np.float64)
            act = np.empty((STATE_STEPS, NA), dtype=np.float64)
            ctrl_state = np.empty((STATE_STEPS, NU), dtype=np.float64)
            time_s = np.empty((STATE_STEPS,), dtype=np.float64)
            actadr = self.teacher["actuator_actadr"]

            def _obs(i: int) -> None:
                qpos[i] = physics.data.qpos
                qvel[i] = physics.data.qvel
                act[i] = physics.data.act[actadr]
                ctrl_state[i] = physics.data.ctrl
                time_s[i] = physics.data.time

            _obs(0)
            for idx in range(CONTROL_STEPS):
                target_ctrl = step_control_fn(idx, physics, bounds, applied[idx, TARGET_COLUMNS])
                if target_ctrl is not None:
                    applied[idx, TARGET_COLUMNS] = np.clip(target_ctrl, bounds[:, 0], bounds[:, 1])
                physics.set_control(applied[idx])
                physics.step(nstep=SUBSTEPS)
                _obs(idx + 1)

            traces = {
                "qpos": qpos,
                "qvel": qvel,
                "actuator_activation": act,
                "state_actuator_ctrl": ctrl_state,
                "time_s": time_s,
                "interval_applied_ctrl": applied,
            }
            for name, arr in traces.items():
                if not np.isfinite(arr).all():
                    raise ValueError(f"Non-finite state trace encountered: {name}")
            return traces, bounds, {"instantiate_s": inst_s, "step_s": time.perf_counter() - t_step0}
        finally:
            physics.free()

    def rollout_forward(
        self,
        excitation: np.ndarray | None,
        tibia_delta_rad: float,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, float]]:
        return self._run_stepped_rollout(
            tibia_delta_rad,
            lambda idx, _p, _b, _cur: None if excitation is None else excitation[idx],
        )

    def rollout_closed_loop(
        self,
        feedforward_excitation: np.ndarray,
        target_rad: np.ndarray,
        tibia_delta_rad: float,
        *,
        kp: float,
        kd: float,
        ext_to_flex_ratio: float,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Run causal proprioceptive PD reflex atop feedforward excitation inside MuJoCo."""
        target_vel = np.gradient(target_rad, CONTROL_TIMESTEP_S)

        def _closed_loop_step(idx: int, physics: Any, _bounds: np.ndarray, _cur: np.ndarray) -> np.ndarray:
            q_cur = float(physics.data.qpos[TARGET_QPOS_ADDRESS])
            qd_cur = float(physics.data.qvel[TARGET_QPOS_ADDRESS])
            err = float(target_rad[idx + 1]) - q_cur
            derr = float(target_vel[idx]) - qd_cur
            u_fb = kp * err + kd * derr
            base = feedforward_excitation[idx]
            return np.array([base[0] + u_fb, base[1] - ext_to_flex_ratio * u_fb], dtype=np.float64)

        traces, _, timing = self._run_stepped_rollout(tibia_delta_rad, _closed_loop_step)
        return traces, timing

    def rollout_with_vjp(
        self,
        excitation: np.ndarray,
        tibia_delta_rad: float,
        *,
        jacobian_eps: float,
        cotangent_qpos: np.ndarray | None,
        mse_target_rad: np.ndarray | None,
        check_directions: bool,
        direction_seed: int,
        direction_eps: float,
        clip_boundary_gradient: str = "native",
    ) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any] | None, dict[str, float]]:
        physics, inst_s = self.instantiate_physics(tibia_delta_rad)
        t_vjp0 = time.perf_counter()
        try:
            model = physics.model.ptr
            nstate = 2 * model.nv + model.na
            if nstate != NSTATE:
                raise ValueError(f"Expected nstate={NSTATE}, got {nstate}")
            bounds = np.asarray(physics.model.actuator_ctrlrange, dtype=np.float64)[TARGET_COLUMNS].copy()
            applied = self.teacher["interval_actuator_ctrl"].copy()
            applied[:, TARGET_COLUMNS] = np.clip(excitation, bounds[:, 0], bounds[:, 1])

            qpos = np.empty((STATE_STEPS, NQ), dtype=np.float64)
            qvel = np.empty((STATE_STEPS, NV), dtype=np.float64)
            act = np.empty((STATE_STEPS, NA), dtype=np.float64)
            ctrl_state = np.empty((STATE_STEPS, NU), dtype=np.float64)
            time_s = np.empty((STATE_STEPS,), dtype=np.float64)
            actadr = self.teacher["actuator_actadr"]

            native_a = np.empty((nstate, nstate), dtype=np.float64)
            native_b = np.empty((nstate, model.nu), dtype=np.float64)
            transitions = np.empty((CONTROL_STEPS, nstate, nstate), dtype=np.float64)
            control_maps = np.empty((CONTROL_STEPS, nstate, 2), dtype=np.float64)

            def _obs(i: int) -> None:
                qpos[i] = physics.data.qpos
                qvel[i] = physics.data.qvel
                act[i] = physics.data.act[actadr]
                ctrl_state[i] = physics.data.ctrl
                time_s[i] = physics.data.time

            _obs(0)
            for idx, ctrl in enumerate(applied):
                physics.set_control(ctrl)
                interval_a = np.eye(nstate, dtype=np.float64)
                interval_b = np.zeros((nstate, 2), dtype=np.float64)
                for _ in range(SUBSTEPS):
                    scratch = copy.copy(physics.data.ptr)
                    self.native_mujoco.mjd_transitionFD(
                        model, scratch, float(jacobian_eps), 1, native_a, native_b, None, None
                    )
                    interval_a, interval_b = _compose_transition(
                        interval_a, interval_b, native_a, native_b[:, TARGET_COLUMNS]
                    )
                    physics.step(nstep=1)
                transitions[idx] = interval_a
                control_maps[idx] = interval_b
                _obs(idx + 1)

            traces = {
                "qpos": qpos,
                "qvel": qvel,
                "actuator_activation": act,
                "state_actuator_ctrl": ctrl_state,
                "time_s": time_s,
                "interval_applied_ctrl": applied,
            }
            for name, arr in traces.items():
                if not np.isfinite(arr).all():
                    raise ValueError(f"Non-finite state trace encountered: {name}")
            if not (np.isfinite(transitions).all() and np.isfinite(control_maps).all()):
                raise ValueError("Non-finite transition Jacobian encountered")
        finally:
            physics.free()

        if mse_target_rad is not None:
            cot = (2.0 / STATE_STEPS) * (qpos[:, TARGET_QPOS_ADDRESS] - mse_target_rad)
        elif cotangent_qpos is not None:
            cot = cotangent_qpos
        else:
            raise ValueError("Need either cotangent_qpos or mse_target_rad for VJP")

        clip_mask = _input_clip_mask(excitation, bounds)
        raw_gradient = _adjoint_control_vjp(
            transitions, control_maps, cot, input_clip_mask=np.ones_like(clip_mask)
        )
        gradient = _apply_input_boundary_gradient(
            excitation, raw_gradient, bounds, mode=clip_boundary_gradient
        )

        dir_checks = None
        if check_directions:
            dir_checks = self._run_direction_checks(
                excitation=excitation,
                tibia_delta_rad=tibia_delta_rad,
                gradient=gradient,
                bounds=bounds,
                clip_mask=clip_mask,
                cotangent=cot,
                mse_target_rad=mse_target_rad,
                direction_seed=direction_seed,
                max_epsilon=direction_eps,
            )

        return (
            traces,
            gradient,
            dir_checks,
            {"instantiate_s": inst_s, "vjp_step_s": time.perf_counter() - t_vjp0},
        )

    def _run_direction_checks(
        self,
        *,
        excitation: np.ndarray,
        tibia_delta_rad: float,
        gradient: np.ndarray,
        bounds: np.ndarray,
        clip_mask: np.ndarray,
        cotangent: np.ndarray,
        mse_target_rad: np.ndarray | None,
        direction_seed: int,
        max_epsilon: float,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(direction_seed)
        checks: list[dict[str, Any]] = []
        for idx in range(2):
            kind = "gradient" if idx == 0 else "random"
            direction = gradient.copy() if idx == 0 else rng.normal(size=gradient.shape)
            direction *= clip_mask
            norm = float(np.linalg.norm(direction))
            if norm == 0.0 and idx == 0 and float(np.linalg.norm(gradient)) > 0.0:
                # One-sided boundary recovery only: test inward step -direction from bound
                base_exc = np.clip(excitation, bounds[:, 0], bounds[:, 1])
                raw_dir = -gradient / float(np.linalg.norm(gradient))
                # Determine available inward margin
                step_caps = np.where(
                    raw_dir > 0.0,
                    (bounds[:, 1] - base_exc) / np.maximum(raw_dir, 1e-30),
                    np.where(raw_dir < 0.0, (base_exc - bounds[:, 0]) / np.maximum(-raw_dir, 1e-30), np.inf),
                )
                active_b = np.abs(raw_dir) > 0.0
                eps = min(float(max_epsilon), float(np.min(step_caps[active_b])) * 0.25)
                if np.isfinite(eps) and eps > 0.0:
                    def _loss_for(exc_arr: np.ndarray) -> float:
                        tr_b, _, _ = self.rollout_forward(exc_arr, tibia_delta_rad)
                        qb = tr_b["qpos"][:, TARGET_QPOS_ADDRESS]
                        return float(np.mean((qb - mse_target_rad) ** 2)) if mse_target_rad is not None else float(qb @ cotangent)
                    l0 = _loss_for(base_exc)
                    l1 = _loss_for(base_exc + eps * raw_dir)
                    fd = (l1 - l0) / eps
                    adjoint = float(np.sum(gradient * raw_dir))
                    abs_err = abs(fd - adjoint)
                    rel_err = abs_err / max(abs(fd), abs(adjoint), 1e-12)
                    checks.append({
                        "kind": "gradient_one_sided_inward",
                        "status": "evaluated",
                        "epsilon": eps,
                        "adjoint": adjoint,
                        "finite_difference": fd,
                        "absolute_error": abs_err,
                        "relative_error": rel_err,
                        "passed": bool(abs_err <= 1e-6 + 0.005 * max(abs(fd), abs(adjoint))),
                    })
                    continue
            if norm == 0.0:
                checks.append({"kind": kind, "status": "zero_direction", "passed": False})
                continue
            direction /= norm
            active = np.abs(direction) > 0.0
            margins = np.minimum(excitation - bounds[:, 0], bounds[:, 1] - excitation)
            eps = min(float(max_epsilon), float(np.min(margins[active] / np.abs(direction[active]))) * 0.25)
            if not np.isfinite(eps) or eps <= 0.0:
                raise ValueError("No positive interior direction perturbation step available")
            losses: list[float] = []
            for sign in (-1.0, 1.0):
                tr, _, _ = self.rollout_forward(excitation + sign * eps * direction, tibia_delta_rad)
                q = tr["qpos"][:, TARGET_QPOS_ADDRESS]
                if mse_target_rad is not None:
                    losses.append(float(np.mean((q - mse_target_rad) ** 2)))
                else:
                    losses.append(float(q @ cotangent))
            fd = (losses[1] - losses[0]) / (2.0 * eps)
            adjoint = float(np.sum(gradient * direction))
            abs_err = abs(fd - adjoint)
            rel_err = abs_err / max(abs(fd), abs(adjoint), 1e-12)
            passed = bool(abs_err <= 1e-6 + 0.005 * max(abs(fd), abs(adjoint)))
            checks.append(
                {
                    "kind": kind,
                    "status": "evaluated",
                    "epsilon": eps,
                    "adjoint": adjoint,
                    "finite_difference": fd,
                    "absolute_error": abs_err,
                    "relative_error": rel_err,
                    "passed": passed,
                }
            )
        return {"all_passed": all(item["passed"] for item in checks), "checks": checks}

    def verify_reference_teacher_gate(self) -> dict[str, Any]:
        """Run two independent rollouts under teacher controls to prove deterministic reset/physics."""
        tr_a, _, _ = self.rollout_forward(None, 0.0)
        tr_b, _, _ = self.rollout_forward(None, 0.0)
        non_target_cols = [i for i in range(NU) if i not in TARGET_COLUMNS]
        teacher_controls = self.teacher["interval_actuator_ctrl"]
        preserved = bool(
            np.array_equal(tr_a["interval_applied_ctrl"][:, non_target_cols], teacher_controls[:, non_target_cols])
        )
        max_self_q_diff = float(np.max(np.abs(tr_a["qpos"] - tr_b["qpos"])))
        max_self_ctrl_diff = float(np.max(np.abs(tr_a["state_actuator_ctrl"] - tr_b["state_actuator_ctrl"])))
        teacher_q_diff = float(np.max(np.abs(tr_a["qpos"] - self.teacher["qpos"])))
        passed = preserved and max_self_q_diff == 0.0 and max_self_ctrl_diff == 0.0
        if self.profile == "original_v1":
            passed = passed and teacher_q_diff <= 1e-12
        return {
            "passed": passed,
            "profile": self.profile,
            "cache_mode": self.cache_mode,
            "teacher_13_controls_bitwise_equal": preserved,
            "independent_repeat_max_qpos_diff": max_self_q_diff,
            "independent_repeat_max_ctrl_diff": max_self_ctrl_diff,
            "original_teacher_max_qpos_diff_rad": teacher_q_diff,
        }

    def _instantiate_multijoint_physics(
        self,
        target_qpos_indices: list[int],
        initial_qpos_delta_rad: np.ndarray | None = None,
    ) -> tuple[Any, float]:
        physics, inst_s = self.instantiate_physics(0.0)
        if initial_qpos_delta_rad is not None:
            model = physics.model
            with physics.reset_context():
                physics.data.qpos[:] = self.teacher["qpos"][0]
                for pos, adr in enumerate(target_qpos_indices):
                    target_val = float(self.teacher["qpos"][0, adr] + float(initial_qpos_delta_rad[pos]))
                    limits = np.asarray(model.jnt_range[adr], dtype=np.float64)
                    if limits[1] > limits[0] and not (limits[0] <= target_val <= limits[1]):
                        raise ValueError(
                            f"Initial qpos[{adr}]={target_val} outside XML joint limits {limits.tolist()}"
                        )
                    physics.data.qpos[adr] = target_val
                physics.data.qvel[:] = self.teacher["qvel"][0]
                physics.data.act[self.teacher["actuator_actadr"]] = self.teacher["actuator_activation"][0]
                physics.data.ctrl[:] = self.teacher["state_actuator_ctrl"][0]
                physics.data.time = float(self.teacher["time_s"][0])
        return physics, inst_s

    def _assemble_applied_controls(
        self,
        physics: Any,
        controls: np.ndarray,
        control_columns: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        bounds = np.asarray(physics.model.actuator_ctrlrange, dtype=np.float64)[control_columns]
        clipped = np.clip(np.asarray(controls, dtype=np.float64), bounds[:, 0], bounds[:, 1])
        if len(control_columns) == NU and sorted(control_columns) == list(range(NU)):
            applied = np.empty((CONTROL_STEPS, NU), dtype=np.float64)
            applied[:, control_columns] = clipped
        else:
            applied = self.teacher["interval_actuator_ctrl"].copy()
            applied[:, control_columns] = clipped
        return applied, bounds

    @staticmethod
    def _record_trace_buffers(physics: Any) -> dict[str, np.ndarray]:
        qpos = np.empty((STATE_STEPS, NQ), dtype=np.float64)
        qvel = np.empty((STATE_STEPS, NV), dtype=np.float64)
        act = np.empty((STATE_STEPS, NA), dtype=np.float64)
        ctrl_state = np.empty((STATE_STEPS, NU), dtype=np.float64)
        time_s = np.empty((STATE_STEPS,), dtype=np.float64)
        qpos[0] = physics.data.qpos
        qvel[0] = physics.data.qvel
        act[0] = physics.data.act[np.asarray(physics.model.actuator_actadr)]
        ctrl_state[0] = physics.data.ctrl
        time_s[0] = float(physics.data.time)
        return {
            "qpos": qpos,
            "qvel": qvel,
            "actuator_activation": act,
            "state_actuator_ctrl": ctrl_state,
            "time_s": time_s,
        }

    def rollout_forward_multijoint(
        self,
        controls: np.ndarray,
        control_columns: list[int],
        target_qpos_indices: list[int],
        initial_qpos_delta_rad: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, float]]:
        physics, inst_s = self._instantiate_multijoint_physics(
            target_qpos_indices, initial_qpos_delta_rad
        )
        t_step0 = time.perf_counter()
        try:
            applied, bounds = self._assemble_applied_controls(physics, controls, control_columns)
            buf = self._record_trace_buffers(physics)
            actadr = np.asarray(physics.model.actuator_actadr)
            for idx in range(CONTROL_STEPS):
                physics.set_control(applied[idx])
                physics.step(nstep=SUBSTEPS)
                buf["qpos"][idx + 1] = physics.data.qpos
                buf["qvel"][idx + 1] = physics.data.qvel
                buf["actuator_activation"][idx + 1] = physics.data.act[actadr]
                buf["state_actuator_ctrl"][idx + 1] = physics.data.ctrl
                buf["time_s"][idx + 1] = float(physics.data.time)
            buf["interval_applied_ctrl"] = applied.copy()
        finally:
            physics.free()
        return buf, bounds, {"instantiate_s": inst_s, "step_s": time.perf_counter() - t_step0}

    def _compute_local_gain_pinv(
        self,
        physics: Any,
        probe_data: Any,
        current_ctrl: np.ndarray,
        control_columns: list[int],
        target_qpos_indices: list[int],
        damping: float,
    ) -> np.ndarray:
        """Estimate instantaneous weighted pseudoinverse mapping [n_joints] -> [n_ctrl] from MuJoCo FD."""
        native = self.native_mujoco
        m_ptr = physics.model._model
        d_ptr = probe_data
        d_ptr.qpos[:] = physics.data.qpos
        d_ptr.qvel[:] = physics.data.qvel
        d_ptr.act[:] = physics.data.act
        d_ptr.ctrl[:] = current_ctrl
        d_ptr.time = float(physics.data.time)
        native.mj_forward(m_ptr, d_ptr)
        a_tot = np.eye(NSTATE, dtype=np.float64)
        b_tot = np.zeros((NSTATE, len(control_columns)), dtype=np.float64)
        step_a = np.zeros((NSTATE, NSTATE), dtype=np.float64)
        step_b = np.zeros((NSTATE, NU), dtype=np.float64)
        for _ in range(2 * SUBSTEPS):  # 4 ms integration horizon matches muscle activation response
            native.mjd_transitionFD(m_ptr, d_ptr, 1e-6, True, step_a, step_b, None, None)
            a_tot, b_tot = _compose_transition(a_tot, b_tot, step_a, step_b[:, control_columns])
            native.mj_step(m_ptr, d_ptr)
        jq = b_tot[target_qpos_indices, :]
        row_norms = np.maximum(np.linalg.norm(jq, axis=1, keepdims=True), 1e-4)
        j_hat = jq / row_norms
        gram = j_hat @ j_hat.T + float(damping) * np.eye(len(target_qpos_indices), dtype=np.float64)
        return j_hat.T @ np.linalg.inv(gram)

    def rollout_multijoint_closed_loop(
        self,
        feedforward_controls: np.ndarray,
        target_rad: np.ndarray,
        control_columns: list[int],
        target_qpos_indices: list[int],
        initial_qpos_delta_rad: np.ndarray | None,
        *,
        kp: np.ndarray,
        kd: np.ndarray,
        jacobian_refresh_steps: int = 35,
        damping: float = 1e-3,
        feedback_clip: float = 0.8,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        physics, inst_s = self._instantiate_multijoint_physics(
            target_qpos_indices, initial_qpos_delta_rad
        )
        t_step0 = time.perf_counter()
        probe_data = self.native_mujoco.MjData(physics.model._model)
        try:
            base_applied, bounds = self._assemble_applied_controls(
                physics, feedforward_controls, control_columns
            )
            realized = base_applied.copy()
            buf = self._record_trace_buffers(physics)
            actadr = np.asarray(physics.model.actuator_actadr)
            target_vel = np.gradient(target_rad, CONTROL_TIMESTEP_S, axis=0)
            pinv_gain: np.ndarray | None = None
            for idx in range(CONTROL_STEPS):
                ctrl_t = base_applied[idx].copy()
                if pinv_gain is None or (idx % jacobian_refresh_steps == 0):
                    pinv_gain = self._compute_local_gain_pinv(
                        physics,
                        probe_data,
                        ctrl_t,
                        control_columns,
                        target_qpos_indices,
                        damping,
                    )
                q_cur = np.asarray(physics.data.qpos[target_qpos_indices], dtype=np.float64)
                qd_cur = np.asarray(physics.data.qvel[target_qpos_indices], dtype=np.float64)
                q_err = target_rad[idx + 1] - q_cur
                v_err = target_vel[idx] - qd_cur
                joint_cmd = kp * q_err + kd * v_err
                delta_u = np.clip(pinv_gain @ joint_cmd, -feedback_clip, feedback_clip)
                ctrl_cols = np.clip(ctrl_t[control_columns] + delta_u, bounds[:, 0], bounds[:, 1])
                ctrl_t[control_columns] = ctrl_cols
                realized[idx] = ctrl_t
                physics.set_control(ctrl_t)
                physics.step(nstep=SUBSTEPS)
                buf["qpos"][idx + 1] = physics.data.qpos
                buf["qvel"][idx + 1] = physics.data.qvel
                buf["actuator_activation"][idx + 1] = physics.data.act[actadr]
                buf["state_actuator_ctrl"][idx + 1] = physics.data.ctrl
                buf["time_s"][idx + 1] = float(physics.data.time)
            buf["interval_applied_ctrl"] = realized
        finally:
            physics.free()
        return buf, {"instantiate_s": inst_s, "step_s": time.perf_counter() - t_step0}

    def rollout_multijoint_with_vjp(
        self,
        controls: np.ndarray,
        control_columns: list[int],
        target_qpos_indices: list[int],
        initial_qpos_delta_rad: np.ndarray | None = None,
        *,
        jacobian_eps: float = 1e-7,
        cotangent_qpos: np.ndarray | None = None,
        mse_target_rad: np.ndarray | None = None,
        joint_weights: np.ndarray | None = None,
        check_directions: bool = False,
        direction_seed: int = 331,
        direction_eps: float = 1e-6,
        clip_boundary_gradient: str = "one_sided",
    ) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any] | None, dict[str, float]]:
        physics, inst_s = self._instantiate_multijoint_physics(
            target_qpos_indices, initial_qpos_delta_rad
        )
        t_vjp0 = time.perf_counter()
        n_ctrl = len(control_columns)
        n_joints = len(target_qpos_indices)
        try:
            applied, bounds = self._assemble_applied_controls(physics, controls, control_columns)
            buf = self._record_trace_buffers(physics)
            actadr = np.asarray(physics.model.actuator_actadr)
            transitions = np.empty((CONTROL_STEPS, NSTATE, NSTATE), dtype=np.float64)
            control_maps = np.empty((CONTROL_STEPS, NSTATE, n_ctrl), dtype=np.float64)
            step_a = np.zeros((NSTATE, NSTATE), dtype=np.float64)
            step_b = np.zeros((NSTATE, NU), dtype=np.float64)
            m_ptr = physics.model._model
            native = self.native_mujoco

            for idx in range(CONTROL_STEPS):
                physics.set_control(applied[idx])
                interval_a = np.eye(NSTATE, dtype=np.float64)
                interval_b = np.zeros((NSTATE, n_ctrl), dtype=np.float64)
                for _ in range(SUBSTEPS):
                    scratch = copy.copy(physics.data.ptr)
                    native.mjd_transitionFD(m_ptr, scratch, jacobian_eps, True, step_a, step_b, None, None)
                    interval_a, interval_b = _compose_transition(
                        interval_a, interval_b, step_a, step_b[:, control_columns]
                    )
                    physics.step(nstep=1)
                transitions[idx] = interval_a
                control_maps[idx] = interval_b
                buf["qpos"][idx + 1] = physics.data.qpos
                buf["qvel"][idx + 1] = physics.data.qvel
                buf["actuator_activation"][idx + 1] = physics.data.act[actadr]
                buf["state_actuator_ctrl"][idx + 1] = physics.data.ctrl
                buf["time_s"][idx + 1] = float(physics.data.time)
            buf["interval_applied_ctrl"] = applied.copy()
        finally:
            physics.free()

        q_sub = buf["qpos"][:, target_qpos_indices]
        if mse_target_rad is not None:
            weights = (
                np.ones((n_joints,), dtype=np.float64)
                if joint_weights is None
                else np.asarray(joint_weights, dtype=np.float64) / float(np.mean(joint_weights))
            )
            cot = (2.0 / (STATE_STEPS * n_joints)) * weights[None, :] * (q_sub - mse_target_rad)
        elif cotangent_qpos is not None:
            weights = None
            cot = np.asarray(cotangent_qpos, dtype=np.float64)
        else:
            raise ValueError("Need either cotangent_qpos or mse_target_rad for multijoint VJP")

        clip_mask = _input_clip_mask(controls, bounds)
        lam = np.zeros(NSTATE, dtype=np.float64)
        for pos, adr in enumerate(target_qpos_indices):
            lam[adr] = float(cot[-1, pos])
        raw_gradient = np.empty((CONTROL_STEPS, n_ctrl), dtype=np.float64)
        for idx in range(CONTROL_STEPS - 1, -1, -1):
            raw_gradient[idx] = control_maps[idx].T @ lam
            lam = transitions[idx].T @ lam
            for pos, adr in enumerate(target_qpos_indices):
                lam[adr] += float(cot[idx, pos])
        if not np.isfinite(raw_gradient).all():
            raise ValueError("Non-finite multijoint adjoint gradient")
        gradient = _apply_input_boundary_gradient(controls, raw_gradient, bounds, mode=clip_boundary_gradient)

        dir_checks = None
        if check_directions:
            def _loss_for(u_arr: np.ndarray) -> float:
                tr_b, _, _ = self.rollout_forward_multijoint(
                    u_arr, control_columns, target_qpos_indices, initial_qpos_delta_rad
                )
                qb = tr_b["qpos"][:, target_qpos_indices]
                if mse_target_rad is not None:
                    assert weights is not None
                    return float(np.mean(weights[None, :] * (qb - mse_target_rad) ** 2))
                return float(np.sum(qb * cot))

            rng = np.random.default_rng(direction_seed)
            checks: list[dict[str, Any]] = []
            for idx in range(2):
                kind = "gradient" if idx == 0 else "random"
                direction = gradient.copy() if idx == 0 else rng.normal(size=gradient.shape)
                direction *= clip_mask
                norm = float(np.linalg.norm(direction))
                if norm == 0.0 and idx == 0 and float(np.linalg.norm(gradient)) > 0.0:
                    base_u = np.clip(controls, bounds[:, 0], bounds[:, 1])
                    raw_dir = -gradient / float(np.linalg.norm(gradient))
                    step_caps = np.where(
                        raw_dir > 0.0,
                        (bounds[:, 1] - base_u) / np.maximum(raw_dir, 1e-30),
                        np.where(raw_dir < 0.0, (base_u - bounds[:, 0]) / np.maximum(-raw_dir, 1e-30), np.inf),
                    )
                    active_b = np.abs(raw_dir) > 0.0
                    eps = min(float(direction_eps), float(np.min(step_caps[active_b])) * 0.25)
                    l0 = _loss_for(base_u)
                    l1 = _loss_for(base_u + eps * raw_dir)
                    fd = (l1 - l0) / eps
                    adjoint = float(np.sum(gradient * raw_dir))
                    abs_err = abs(fd - adjoint)
                    checks.append({
                        "kind": "gradient_one_sided_inward",
                        "status": "evaluated",
                        "epsilon": eps,
                        "adjoint": adjoint,
                        "finite_difference": fd,
                        "absolute_error": abs_err,
                        "relative_error": abs_err / max(abs(fd), abs(adjoint), 1e-12),
                        "passed": bool(abs_err <= 1e-6 + 0.005 * max(abs(fd), abs(adjoint))),
                    })
                    continue
                if norm == 0.0:
                    checks.append({"kind": kind, "status": "zero_direction", "passed": False})
                    continue
                direction /= norm
                active = np.abs(direction) > 0.0
                margins = np.minimum(controls - bounds[:, 0], bounds[:, 1] - controls)
                eps = min(float(direction_eps), float(np.min(margins[active] / np.abs(direction[active]))) * 0.25)
                fd = (_loss_for(controls + eps * direction) - _loss_for(controls - eps * direction)) / (2.0 * eps)
                adjoint = float(np.sum(gradient * direction))
                abs_err = abs(fd - adjoint)
                checks.append({
                    "kind": kind,
                    "status": "evaluated",
                    "epsilon": eps,
                    "adjoint": adjoint,
                    "finite_difference": fd,
                    "absolute_error": abs_err,
                    "relative_error": abs_err / max(abs(fd), abs(adjoint), 1e-12),
                    "passed": bool(abs_err <= 1e-6 + 0.005 * max(abs(fd), abs(adjoint))),
                })
            dir_checks = {"all_passed": all(c["passed"] for c in checks), "checks": checks}

        return (
            buf,
            gradient,
            dir_checks,
            {"instantiate_s": inst_s, "vjp_step_s": time.perf_counter() - t_vjp0},
        )

    def render_top_view(
        self,
        excitation: np.ndarray,
        tibia_delta_rad: float,
        output_dir: Path,
        stem: str,
    ) -> dict[str, Any]:
        """Render 42 top-view frames at 25 fps with visibility quality gate."""
        physics, _ = self.instantiate_physics(tibia_delta_rad)
        bounds = np.asarray(physics.model.actuator_ctrlrange, dtype=np.float64)[TARGET_COLUMNS]
        applied = self.teacher["interval_actuator_ctrl"].copy()
        applied[:, TARGET_COLUMNS] = np.clip(excitation, bounds[:, 0], bounds[:, 1])
        return self._render_top_view_with_physics(physics, applied, output_dir, stem)

    def render_top_view_multijoint(
        self,
        controls: np.ndarray,
        control_columns: list[int],
        target_qpos_indices: list[int],
        initial_qpos_delta_rad: np.ndarray | None,
        output_dir: Path,
        stem: str,
    ) -> dict[str, Any]:
        """Render 42 top-view frames for arbitrary controlled actuator columns (e.g., all 15 LF muscles)."""
        physics, _ = self._instantiate_multijoint_physics(target_qpos_indices, initial_qpos_delta_rad)
        applied, _ = self._assemble_applied_controls(physics, controls, control_columns)
        return self._render_top_view_with_physics(physics, applied, output_dir, stem)

    def _render_top_view_with_physics(
        self,
        physics: Any,
        applied: np.ndarray,
        output_dir: Path,
        stem: str,
    ) -> dict[str, Any]:
        import imageio.v2 as imageio

        output_dir.mkdir(parents=True, exist_ok=True)
        frames: list[np.ndarray] = []
        try:

            ids = [
                physics.model.name2id(name, "body")
                for name in ("LFCoxa", "LFFemur", "LFTibia", "LFTarsus1")
            ]
            top_center = np.asarray(physics.data.xpos, dtype=np.float64)[ids].mean(axis=0)

            def _style(phys: Any, scene: Any) -> None:
                native = self.native_mujoco
                for flag in (
                    native.mjtRndFlag.mjRND_SHADOW,
                    native.mjtRndFlag.mjRND_REFLECTION,
                    native.mjtRndFlag.mjRND_FOG,
                    native.mjtRndFlag.mjRND_SKYBOX,
                ):
                    scene.flags[flag] = 0
                for idx in range(scene.nlight):
                    light = scene.lights[idx]
                    light.ambient[:] = [0.18, 0.18, 0.18]
                    light.diffuse[:] = [0.35, 0.35, 0.35]
                    light.specular[:] = [0.0, 0.0, 0.0]
                    light.castshadow = False
                for idx in range(scene.ngeom):
                    geom = scene.geoms[idx]
                    if geom.objtype == native.mjtObj.mjOBJ_TENDON:
                        geom.rgba[:] = [0.84, 0.37, 0.12, 0.8]
                    if geom.objtype != native.mjtObj.mjOBJ_GEOM:
                        continue
                    name = phys.model.id2name(geom.objid, "geom") or ""
                    geom.specular = 0.0
                    geom.shininess = 0.0
                    if name == "floor":
                        geom.matid = -1
                        geom.rgba[:] = [0.90, 0.92, 0.94, 1.0]
                        geom.emission = 0.0
                    elif name.startswith(("LFCoxa", "LFTrochanter", "LFFemur", "LFTarsus")):
                        geom.rgba[:] = [0.12, 0.40, 0.55, 1.0]
                        geom.transparent = False
                        geom.emission = 0.0
                    elif name.startswith("LFTibia"):
                        geom.rgba[:] = [0.02, 0.68, 0.62, 1.0]
                        geom.transparent = False
                        geom.emission = 0.0
                    else:
                        geom.rgba[:] = [0.35, 0.42, 0.50, 0.25]
                        geom.transparent = True

            camera = self.dm_mujoco.MovableCamera(
                physics, height=480, width=640, scene_callback=_style
            )
            camera.set_pose(
                distance=3.4,
                lookat=top_center,
                azimuth=90.0,
                elevation=-90.0,
            )

            for idx, ctrl in enumerate(applied):
                physics.set_control(ctrl)
                physics.step(nstep=SUBSTEPS)
                if (idx + 1) % 5 == 0:
                    frames.append(camera.render().copy())
        finally:
            physics.free()

        array = np.asarray(frames, dtype=np.uint8)
        signed = array.astype(np.int16)
        teal = (signed[..., 1] > signed[..., 0] + 20) & (signed[..., 2] > signed[..., 0] + 15) & (signed[..., 1] > 70)
        visible = teal.sum(axis=(1, 2))
        changes = np.max(np.abs(signed - signed[0]), axis=-1)
        moving = int(np.count_nonzero(changes.max(axis=0) > 12))
        passed = bool(array.mean() > 80.0 and int(visible.min()) > 100 and moving > 100)

        mp4_path = output_dir / f"{stem}.mp4"
        webm_path = output_dir / f"{stem}.webm"
        png_path = output_dir / f"{stem}_preview.png"
        imageio.mimsave(mp4_path, frames, fps=25)
        try:
            imageio.mimsave(webm_path, frames, fps=25, codec="libvpx-vp9")
        except Exception:
            webm_path = None  # optional codec
        imageio.imwrite(png_path, frames[min(20, len(frames) - 1)])

        return {
            "mp4_path": str(mp4_path) if mp4_path.is_file() else None,
            "webm_path": str(webm_path) if webm_path and webm_path.is_file() else None,
            "preview_png_path": str(png_path) if png_path.is_file() else None,
            "frames": len(frames),
            "visible": passed,
            "minimum_target_colored_pixels": int(visible.min()),
            "changed_pixels_from_first": moving,
        }


def main() -> int:
    # Preserve binary protocol stdout fd and route any accidental stdout/C prints to stderr.
    os.environ.setdefault("MUJOCO_GL", "egl")
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    proto_out = os.fdopen(proto_fd, "wb", buffering=0)
    in_fd = sys.stdin.fileno()

    state: _PhysicsWorkerState | None = None
    last_gate: dict[str, Any] | None = None

    while True:
        msg = read_frame_fd(in_fd, timeout_s=None)
        if msg is None:
            return 0
        header, arrays = msg
        op = header.get("op")
        try:
            if op == "close":
                write_frame(proto_out, {"ok": True, "op": "close"})
                return 0

            if op == "handshake":
                if int(header.get("protocol_version", -1)) != PROTOCOL_VERSION:
                    raise ValueError(f"Unsupported protocol version: {header.get('protocol_version')}")
                profile = str(header["profile"])
                xml_path = Path(str(header["xml_path"]))
                teacher_path = Path(str(header["teacher_path"]))
                expected_teacher_sha = str(header.get("expected_teacher_sha256", EXPECTED_TEACHER_SHA256))
                cache_mode = str(header.get("cache_mode", "fresh"))
                state = _PhysicsWorkerState(
                    profile=profile,
                    xml_path=xml_path,
                    teacher_path=teacher_path,
                    expected_teacher_sha256=expected_teacher_sha,
                    cache_mode=cache_mode,
                )
                last_gate = state.verify_reference_teacher_gate()
                if not last_gate["passed"]:
                    raise RuntimeError(f"Worker startup reference gate failed: {last_gate}")
                write_frame(
                    proto_out,
                    {
                        "ok": True,
                        "op": "handshake",
                        "protocol_version": PROTOCOL_VERSION,
                        "worker_pid": os.getpid(),
                        "profile": profile,
                        "cache_mode": cache_mode,
                        "runtime_versions": {
                            "python": sys.version.split()[0],
                            "numpy": np.__version__,
                            "mujoco": str(getattr(state.native_mujoco, "__version__", "unknown")),
                            "dm_control": str(getattr(__import__("dm_control"), "__version__", "unknown")),
                        },
                        "independent_teacher_gate": last_gate,
                    },
                )
                continue

            if state is None:
                raise RuntimeError("Worker received request before handshake")

            if op == "evaluate":
                excitation = arrays["controls"]
                delta_rad = float(header["initial_tibia_delta_rad"])
                need_jac = bool(header.get("need_jacobian", False))
                if need_jac:
                    traces, vjp, dir_checks, timing_s = state.rollout_with_vjp(
                        excitation,
                        delta_rad,
                        jacobian_eps=float(header.get("jacobian_eps", 1e-7)),
                        cotangent_qpos=arrays.get("cotangent_qpos"),
                        mse_target_rad=arrays.get("mse_target_rad"),
                        check_directions=bool(header.get("check_directions", False)),
                        direction_seed=int(header.get("direction_seed", 331)),
                        direction_eps=float(header.get("direction_eps", 1e-6)),
                        clip_boundary_gradient=str(header.get("clip_boundary_gradient", "native")),
                    )
                    out_arrays = dict(traces)
                    out_arrays["control_vjp"] = vjp
                else:
                    traces, _, timing_s = state.rollout_forward(excitation, delta_rad)
                    dir_checks = None
                    out_arrays = dict(traces)
                write_frame(
                    proto_out,
                    {
                        "ok": True,
                        "op": "evaluate",
                        "request_id": header.get("request_id", ""),
                        "direction_checks": dir_checks,
                        "independent_teacher_gate": last_gate,
                        "timing_s": timing_s,
                    },
                    out_arrays,
                )
                continue

            if op == "evaluate_closed_loop":
                feedforward = arrays["feedforward_controls"]
                target_rad = arrays["target_rad"]
                delta_rad = float(header["initial_tibia_delta_rad"])
                traces, timing_s = state.rollout_closed_loop(
                    feedforward,
                    target_rad,
                    delta_rad,
                    kp=float(header["kp"]),
                    kd=float(header["kd"]),
                    ext_to_flex_ratio=float(header["ext_to_flex_ratio"]),
                )
                write_frame(
                    proto_out,
                    {
                        "ok": True,
                        "op": "evaluate_closed_loop",
                        "request_id": header.get("request_id", ""),
                        "direction_checks": None,
                        "independent_teacher_gate": last_gate,
                        "timing_s": timing_s,
                    },
                    dict(traces),
                )
                continue

            if op == "render":
                excitation = arrays["controls"]
                delta_rad = float(header["initial_tibia_delta_rad"])
                output_dir = Path(str(header["output_dir"]))
                stem = str(header.get("stem", "simulation_top_view"))
                info = state.render_top_view(excitation, delta_rad, output_dir, stem)
                write_frame(proto_out, {"ok": True, "op": "render", "render": info})
                continue

            if op == "evaluate_multijoint":
                controls = arrays["controls"]
                control_columns = [int(c) for c in header["control_columns"]]
                target_qpos_indices = [int(i) for i in header["target_qpos_indices"]]
                init_delta = arrays.get("initial_qpos_delta_rad")
                need_jac = bool(header.get("need_jacobian", False))
                if need_jac:
                    traces, vjp, dir_checks, timing_s = state.rollout_multijoint_with_vjp(
                        controls,
                        control_columns,
                        target_qpos_indices,
                        init_delta,
                        jacobian_eps=float(header.get("jacobian_eps", 1e-7)),
                        cotangent_qpos=arrays.get("cotangent_qpos"),
                        mse_target_rad=arrays.get("mse_target_rad"),
                        joint_weights=arrays.get("joint_weights"),
                        check_directions=bool(header.get("check_directions", False)),
                        direction_seed=int(header.get("direction_seed", 331)),
                        direction_eps=float(header.get("direction_eps", 1e-6)),
                        clip_boundary_gradient=str(header.get("clip_boundary_gradient", "one_sided")),
                    )
                    out_arrays = dict(traces)
                    out_arrays["control_vjp"] = vjp
                else:
                    traces, _, timing_s = state.rollout_forward_multijoint(
                        controls, control_columns, target_qpos_indices, init_delta
                    )
                    dir_checks = None
                    out_arrays = dict(traces)
                write_frame(
                    proto_out,
                    {
                        "ok": True,
                        "op": "evaluate_multijoint",
                        "request_id": header.get("request_id", ""),
                        "direction_checks": dir_checks,
                        "independent_teacher_gate": last_gate,
                        "timing_s": timing_s,
                    },
                    out_arrays,
                )
                continue

            if op == "evaluate_multijoint_closed_loop":
                feedforward = arrays["feedforward_controls"]
                target_rad = arrays["target_rad"]
                control_columns = [int(c) for c in header["control_columns"]]
                target_qpos_indices = [int(i) for i in header["target_qpos_indices"]]
                init_delta = arrays.get("initial_qpos_delta_rad")
                traces, timing_s = state.rollout_multijoint_closed_loop(
                    feedforward,
                    target_rad,
                    control_columns,
                    target_qpos_indices,
                    init_delta,
                    kp=np.asarray(arrays["kp"], dtype=np.float64),
                    kd=np.asarray(arrays["kd"], dtype=np.float64),
                    jacobian_refresh_steps=int(header.get("jacobian_refresh_steps", 35)),
                    damping=float(header.get("damping", 1e-3)),
                    feedback_clip=float(header.get("feedback_clip", 0.8)),
                )
                write_frame(
                    proto_out,
                    {
                        "ok": True,
                        "op": "evaluate_multijoint_closed_loop",
                        "request_id": header.get("request_id", ""),
                        "direction_checks": None,
                        "independent_teacher_gate": last_gate,
                        "timing_s": timing_s,
                    },
                    dict(traces),
                )
                continue

            if op == "render_multijoint":
                controls = arrays["controls"]
                control_columns = [int(c) for c in header["control_columns"]]
                target_qpos_indices = [int(i) for i in header["target_qpos_indices"]]
                init_delta = arrays.get("initial_qpos_delta_rad")
                output_dir = Path(str(header["output_dir"]))
                stem = str(header.get("stem", "multijoint_top_view"))
                info = state.render_top_view_multijoint(
                    controls, control_columns, target_qpos_indices, init_delta, output_dir, stem
                )
                write_frame(proto_out, {"ok": True, "op": "render_multijoint", "render": info})
                continue

            raise ValueError(f"Unknown worker op: {op!r}")
        except Exception as exc:
            write_frame(
                proto_out,
                {
                    "ok": False,
                    "op": op,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )


if __name__ == "__main__":
    raise SystemExit(main())
