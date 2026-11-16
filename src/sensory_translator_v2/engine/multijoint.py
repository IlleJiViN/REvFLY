"""High-level EngineV2 extension for 15-muscle 7-DOF LF control & bio spike/LIF interfacing."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from ..assets import (
    MULTIJOINT_CHECKPOINT_RELPATH,
    AssetCatalog,
    load_asset_catalog,
    verify_asset_catalog,
)
from ..backends.base import FakeBackend
from ..backends.persistent_mujoco import PersistentMujocoBackend
from ..bio.malecns_pools import premotor_unavailability
from ..contracts import (
    AssetValidationError,
    Backend,
    BackendSessionError,
    CONTROL_STEPS,
    CONTROL_TIMESTEP_S,
    InitialState,
    LF_ACTUATOR_COLUMNS,
    LF_ACTUATOR_NAMES,
    LF_JOINT_GROUPS,
    LF_JOINT_NAMES,
    LF_QPOS_ADDRESSES,
    MultiJointCapableBackend,
    MultiJointClosedLoopRequest,
    MultiJointEvaluationRequest,
    NU,
    RequestValidationError,
    STATE_STEPS,
    SessionSpec,
    TRACE_KEYS,
    require_int,
    require_real_float,
    validate_vector,
)
from ..evidence import utc_now_iso, write_json_atomic, write_npz_atomic
from ..goals import GOAL_CONDITION_SAMPLES, encode_multijoint_condition, validate_multijoint_goal
from ..rate.bridge import (
    LF_TIBIA_POOLS,
    MOTOR_BODY_IDS,
    direct_multichannel_rate_waveform,
    multichannel_rate_to_excitation,
)
from ..rate.model import RatePolicyProtocol, load_rate_checkpoint
from ..rate.optim import lbfgs_direction
from ..bio import (
    LIFBiophysicsParams,
    deconvolve_target_excitation_to_rates,
    load_malecns_premotor_subnetwork,
    optimize_premotor_connectome_drive,
    optimize_rate_spikes_with_physics_vjp,
    realize_direct_lif_mn_drive,
    realize_rate_spikes,
    resolve_motor_pools,
)

DEFAULT_MULTIJOINT_CHECKPOINT_REL = str(MULTIJOINT_CHECKPOINT_RELPATH)
MULTIJOINT_GOAL_FAMILIES: tuple[str, ...] = (
    "swing_scaled",
    "decoupled",
    "rhythmic_double_tap",
    "reach_and_hold",
    "kinematic_teacher_scaled",
)


def _build_primitive_controls(
    teacher_u15: np.ndarray,
    *,
    family: str,
    scale: float,
    decoupled_scales: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    u_mean = teacher_u15.mean(axis=0, keepdims=True)
    du = teacher_u15 - u_mean
    t = np.linspace(0.0, 1.0, CONTROL_STEPS, dtype=np.float64)[:, None]
    if family == "swing_scaled":
        return np.clip(u_mean + scale * du, 0.01, 0.99)
    if family == "decoupled":
        sc, sf, st = decoupled_scales
        out = teacher_u15.copy()
        out[:, 0:7] = np.clip(u_mean[:, 0:7] + sc * du[:, 0:7], 0.01, 0.99)
        out[:, 7:13] = np.clip(u_mean[:, 7:13] + sf * du[:, 7:13], 0.01, 0.99)
        out[:, 13:15] = np.clip(u_mean[:, 13:15] + st * du[:, 13:15], 0.01, 0.99)
        return out
    if family == "rhythmic_double_tap":
        amp = 0.14 * scale
        out = u_mean.repeat(CONTROL_STEPS, axis=0) * 0.8 + 0.1 * teacher_u15
        osc1 = np.sin(2.0 * np.pi * t)
        osc2 = np.sin(4.0 * np.pi * t)
        out[:, [2, 3]] += amp * osc1
        out[:, [5]] -= amp * osc1
        out[:, [7, 12]] += amp * osc2
        out[:, [11]] -= 0.7 * amp * osc2
        out[:, [13]] += amp * osc2
        out[:, [14]] -= 0.8 * amp * osc2
        return np.clip(out, 0.01, 0.99)
    if family == "reach_and_hold":
        hold = 0.16 * scale
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.clip(t / 0.55, 0.0, 1.0)))
        out = u_mean.repeat(CONTROL_STEPS, axis=0)
        out[:, [0, 1, 2]] += hold * ramp
        out[:, [7, 8, 12]] += hold * ramp
        out[:, [13]] += hold * ramp
        out[:, [14]] -= 0.7 * hold * ramp
        return np.clip(out, 0.01, 0.99)
    raise RequestValidationError(
        f"Unknown multi-joint goal family {family!r}; expected one of {MULTIJOINT_GOAL_FAMILIES}"
    )


@dataclass(frozen=True)
class MultiJointEngineConfig:
    """Configuration for 15-muscle (0 teacher-assist) 7-DOF LF controller."""

    backend: str = "persistent_cached"
    profile: str = "adjacent_static_parent_v1"
    checkpoint_path: Path | None = None
    asset_root: str | Path | None = None
    runtime_root: str | Path | None = None
    physics_python: str | Path | None = None
    timeout_s: float = 90.0


class MultiJointEngine:
    """Thread-safe controller operating all 15 LF muscles across Coxa, Femur/Trochanter, and Tibia."""

    def __init__(self, config: MultiJointEngineConfig | None = None) -> None:
        self.config = config or MultiJointEngineConfig()
        self._lock = threading.RLock()
        self._catalog: AssetCatalog | None = None
        self._backend: Backend | None = None
        self._model: RatePolicyProtocol | nn.Module | None = None
        self._model_cfg: dict[str, Any] = {}
        self._model_id: str | None = None
        self._ckpt_path: Path | None = None
        self._ckpt_sha: str | None = None
        self._expected_ckpt_sha: str | None = None
        self._teacher_q7: np.ndarray | None = None
        self._teacher_v0: np.ndarray | None = None
        self._teacher_u15: np.ndarray | None = None
        self._closed = False

    def _ensure_init(self) -> None:
        if self._closed:
            raise BackendSessionError("MultiJointEngine is closed")
        if self._catalog is not None and self._backend is not None:
            return
        cat = load_asset_catalog(
            asset_root=self.config.asset_root,
            runtime_root=self.config.runtime_root,
            physics_python=self.config.physics_python,
            verify_hashes=True,
        )
        model_id = None
        expected_sha = None
        if self.config.checkpoint_path is not None:
            model_id = "custom_multijoint"
            ckpt_path = Path(self.config.checkpoint_path).expanduser().resolve()
            if not ckpt_path.is_file():
                raise AssetValidationError(f"Multi-joint checkpoint does not exist: {ckpt_path}")
        else:
            model_entry = cat.resolve_multijoint_model()
            ckpt_path = model_entry.checkpoint_path if model_entry is not None else None
            if model_entry is not None:
                model_id = model_entry.model_id
                expected_sha = model_entry.checkpoint_sha256
        if ckpt_path is not None:
            sha = expected_sha or hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
            model, mcfg, _ = load_rate_checkpoint(ckpt_path, expected_sha256=sha)
            condition_dim = len(LF_QPOS_ADDRESSES) * (GOAL_CONDITION_SAMPLES + 2) + 1
            if model.n_pools != len(LF_ACTUATOR_COLUMNS) or model.condition_dim != condition_dim:
                raise AssetValidationError(
                    f"Multi-joint checkpoint requires {len(LF_ACTUATOR_COLUMNS)} motor pools "
                    f"and {condition_dim} condition features"
                )
            self._model = model
            self._model_cfg = mcfg
            self._model_id = model_id
            self._ckpt_path = ckpt_path
            self._ckpt_sha = sha
            self._expected_ckpt_sha = expected_sha
        with np.load(cat.teacher_path, allow_pickle=False) as tf:
            self._teacher_q7 = np.asarray(tf["qpos"][:, :7], dtype=np.float64)
            self._teacher_v0 = np.asarray(tf["qvel"][0, :7], dtype=np.float64)
            self._teacher_u15 = np.asarray(tf["interval_actuator_ctrl"], dtype=np.float64)

        if self.config.backend == "fake":
            backend: Backend = FakeBackend(profile=self.config.profile, teacher_joint_rad=self._teacher_q7[:, 6])
        elif self.config.backend in ("persistent_fresh", "persistent_cached"):
            cache_mode = "cached_model" if self.config.backend == "persistent_cached" else "fresh"
            backend = PersistentMujocoBackend(
                cat,
                profile=self.config.profile,
                cache_mode=cache_mode,
                timeout_s=self.config.timeout_s,
            )
        else:
            raise RequestValidationError(
                f"MultiJointEngine requires 'persistent_cached', 'persistent_fresh', or 'fake', got {self.config.backend!r}"
            )
        self._catalog = cat
        self._backend = backend

    def checkpoint_sha256(self) -> str | None:
        self._ensure_init()
        return self._ckpt_sha

    def _checkpoint_identity(self) -> dict[str, Any]:
        return {
            "model_id": self._model_id,
            "sha256": self._ckpt_sha,
            "architecture": (
                self._model_cfg.get("architecture")
                or self._model_cfg.get("model_type")
                or getattr(self._model, "architecture_name", None)
            ),
            "expected_sha256_verified": self._expected_ckpt_sha is not None,
        }

    def reference_target_rad(self) -> np.ndarray:
        self._ensure_init()
        assert self._teacher_q7 is not None
        return self._teacher_q7.copy()

    def verify_assets(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_init()
            assert self._catalog is not None
            return verify_asset_catalog(self._catalog)

    def capabilities(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_init()
            assert self._catalog is not None and self._backend is not None
            bio_modes = ["none", "deterministic_isi", "poisson_refractory", "direct_lif"]
            unavailable_bio_modes: dict[str, str] = {}
            premotor_error = premotor_unavailability(self._catalog.asset_root)
            if premotor_error is None:
                bio_modes.append("premotor_connectome")
            else:
                unavailable_bio_modes["premotor_connectome"] = premotor_error
            return {
                "schema_version": "sensory_translator_v2_multijoint_capabilities_v1",
                "profile": self.config.profile,
                "ready": self._model is not None,
                "model_id": self._model_id,
                "backend": self._backend.describe().name,
                "is_certified_frozen_physics": self._backend.describe().is_certified_frozen_physics,
                "controlled_muscles": len(LF_ACTUATOR_COLUMNS),
                "teacher_assisted_muscles": 0,
                "controlled_joints": list(LF_JOINT_NAMES),
                "joint_groups": {k: list(v) for k, v in LF_JOINT_GROUPS.items()},
                "actuator_names": list(LF_ACTUATOR_NAMES),
                "modes": ["fast", "auto", "refine", "closed_loop"],
                "goal_families": list(MULTIJOINT_GOAL_FAMILIES),
                "bio_modes": bio_modes,
                "unavailable_bio_modes": unavailable_bio_modes,
                "checkpoint": {
                    "path": str(self._ckpt_path) if self._ckpt_path else None,
                    **self._checkpoint_identity(),
                },
            }

    def goal_from_family(
        self,
        *,
        family: str = "swing_scaled",
        scale: float = 0.92,
        initial_delta_deg: np.ndarray | list[float] | tuple[float, ...] | None = None,
        decoupled_scales: tuple[float, float, float] = (1.05, 0.85, 0.95),
    ) -> dict[str, Any]:
        """Build a canonical 7-DOF [211, 7] goal trajectory across supported motion families."""
        if family not in MULTIJOINT_GOAL_FAMILIES:
            raise RequestValidationError(
                f"family must be one of {MULTIJOINT_GOAL_FAMILIES}, got {family!r}"
            )
        scale_val = require_real_float(scale, "scale", positive=True)
        delta_deg_arr = (
            np.zeros((7,), dtype=np.float64)
            if initial_delta_deg is None
            else validate_vector(initial_delta_deg, "initial_delta_deg", length=7)
        )
        with self._lock:
            self._ensure_init()
            assert self._catalog is not None and self._backend is not None
            assert self._teacher_q7 is not None and self._teacher_u15 is not None
            time_s = (np.arange(STATE_STEPS, dtype=np.float64) * CONTROL_TIMESTEP_S).tolist()
            if family == "kinematic_teacher_scaled":
                target = self._teacher_q7[0:1] + scale_val * (self._teacher_q7 - self._teacher_q7[0:1])
                target = target + np.radians(delta_deg_arr)[None, :]
                return {
                    "family": family,
                    "scale": float(scale_val),
                    "reachable_in_profile": False,
                    "joint_names": list(LF_JOINT_NAMES),
                    "time_s": time_s,
                    "target_rad": target.tolist(),
                    "initial_delta_deg": delta_deg_arr.tolist(),
                }
            ctrl = _build_primitive_controls(
                self._teacher_u15,
                family=family,
                scale=scale_val,
                decoupled_scales=decoupled_scales,
            )
            hashes = self._catalog.three_hash_record(self.config.profile)
            spec = SessionSpec(
                profile=self.config.profile,
                asset_manifest_sha256=self._catalog.manifest_sha256,
                teacher_sha256=self._catalog.teacher_sha256,
                raw_xml_sha256=self._catalog.raw_xml_sha256,
                profile_rule_sha256=hashes["profile_rule_sha256"],
                cache_mode=self._backend.describe().cache_mode,
            )
            sess = self._backend.open_session(spec)
            try:
                self._backend.reset(sess, InitialState(0.0), seed=0)
                ev = self._backend.evaluate_multijoint(
                    sess,
                    MultiJointEvaluationRequest(
                        controls=ctrl,
                        control_columns=LF_ACTUATOR_COLUMNS,
                        target_qpos_indices=LF_QPOS_ADDRESSES,
                        initial_qpos_delta_rad=np.radians(delta_deg_arr),
                        need_jacobian=False,
                        request_id=f"goal_{family}",
                    ),
                )
                target = ev.traces["qpos"][:, list(LF_QPOS_ADDRESSES)]
            finally:
                self._backend.close_session(sess)
            return {
                "family": family,
                "scale": float(scale_val),
                "reachable_in_profile": ev.is_certified_frozen_physics,
                "joint_names": list(LF_JOINT_NAMES),
                "time_s": time_s,
                "target_rad": target.tolist(),
                "initial_delta_deg": delta_deg_arr.tolist(),
            }

    def _bridge_params(self) -> tuple[float, float, float, str]:
        bridge = (self._model_cfg or {}).get("bridge", {})
        max_hz = float(self._model.max_rate_hz) if self._model is not None else 200.0
        tau_ms = float(bridge.get("tau_ms", 2.0))
        rate_scale_hz = float(bridge.get("rate_scale_hz", 100.0))
        clip_mode = "one_sided" if self.config.backend != "fake" else "native"
        return max_hz, tau_ms, rate_scale_hz, clip_mode

    def predict_nominal(
        self,
        target_rad: np.ndarray,
        *,
        initial_qpos_delta_rad: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict [15, K], [420, 15], and [210, 15] nominal motor commands (0 teacher-assisted)."""
        self._ensure_init()
        if self._model is None:
            raise RequestValidationError(
                "No multi-joint checkpoint loaded; provide --checkpoint or train with train-multijoint"
            )
        assert self._teacher_q7 is not None and self._teacher_v0 is not None
        targ, _ = validate_multijoint_goal(target_rad, n_joints=7)
        delta = (
            np.zeros((7,), dtype=np.float64)
            if initial_qpos_delta_rad is None
            else validate_vector(initial_qpos_delta_rad, "initial_qpos_delta_rad", length=7)
        )
        q0 = self._teacher_q7[0] + delta
        max_hz, tau_ms, rate_scale_hz, clip_mode = self._bridge_params()
        with torch.no_grad():
            cond = encode_multijoint_condition(
                torch.from_numpy(targ).unsqueeze(0),
                torch.from_numpy(q0).unsqueeze(0),
                torch.from_numpy(self._teacher_v0).unsqueeze(0),
            )
            knots = self._model(cond).clamp(0.0, max_hz)
            wf = direct_multichannel_rate_waveform(knots, max_rate_hz=max_hz)
            exc = multichannel_rate_to_excitation(
                wf,
                max_rate_hz=max_hz,
                rate_scale_hz=rate_scale_hz,
                tau_ms=tau_ms,
                clip_boundary_gradient=clip_mode,
            )
        return knots[0].cpu().numpy(), wf[0].cpu().numpy(), exc[0].cpu().numpy()

    def run(
        self,
        target_rad: Any,
        *,
        output: str | Path | None = None,
        mode: str = "auto",
        threshold_deg: float = 5.729577951308233,
        correction_steps: int = 4,
        feedback_kp: float | np.ndarray = 8.0,
        feedback_kd: float | np.ndarray = 0.04,
        initial_qpos_delta_deg: np.ndarray | None = None,
        initial_qpos_delta_rad: np.ndarray | None = None,
        render: bool = False,
        bio_mode: str = "none",
        bio_seed: int = 0,
        bio_physics_vjp_steps: int = 0,
        bio_population_scale: int = 1,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run 15-muscle (0 teacher-assist) 7-DOF LF simulation, optional L-BFGS / PD, and optional bio spike/LIF realization."""
        with self._lock:
            return self._run_unlocked(
                target_rad,
                output=output,
                mode=mode,
                threshold_deg=threshold_deg,
                correction_steps=correction_steps,
                feedback_kp=feedback_kp,
                feedback_kd=feedback_kd,
                initial_qpos_delta_deg=initial_qpos_delta_deg,
                initial_qpos_delta_rad=initial_qpos_delta_rad,
                render=render,
                bio_mode=bio_mode,
                bio_seed=bio_seed,
                bio_physics_vjp_steps=bio_physics_vjp_steps,
                bio_population_scale=bio_population_scale,
                progress_callback=progress_callback,
            )

    def _run_unlocked(
        self,
        target_rad: Any,
        *,
        output: str | Path | None = None,
        mode: str = "auto",
        threshold_deg: float = 5.729577951308233,
        correction_steps: int = 4,
        feedback_kp: float | np.ndarray = 8.0,
        feedback_kd: float | np.ndarray = 0.04,
        initial_qpos_delta_deg: np.ndarray | None = None,
        initial_qpos_delta_rad: np.ndarray | None = None,
        render: bool = False,
        bio_mode: str = "none",
        bio_seed: int = 0,
        bio_physics_vjp_steps: int = 0,
        bio_population_scale: int = 1,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if mode not in ("fast", "auto", "refine", "closed_loop"):
            raise RequestValidationError("mode must be fast, auto, refine, or closed_loop")
        if bio_mode not in ("none", "deterministic_isi", "poisson_refractory", "direct_lif", "premotor_connectome"):
            raise RequestValidationError(f"Unsupported bio_mode: {bio_mode!r}")
        steps = require_int(correction_steps, "correction_steps", minimum=0, maximum=20)
        threshold = require_real_float(threshold_deg, "threshold_deg", positive=True)
        targ, _ = validate_multijoint_goal(target_rad, n_joints=7)

        if initial_qpos_delta_rad is not None and initial_qpos_delta_deg is not None:
            raise RequestValidationError("Specify at most one of initial_qpos_delta_rad or initial_qpos_delta_deg")
        if initial_qpos_delta_rad is not None:
            delta_rad = validate_vector(initial_qpos_delta_rad, "initial_qpos_delta_rad", length=7)
        elif initial_qpos_delta_deg is not None:
            delta_rad = np.radians(validate_vector(initial_qpos_delta_deg, "initial_qpos_delta_deg", length=7))
        else:
            delta_rad = np.zeros((7,), dtype=np.float64)

        self._ensure_init()
        assert self._catalog is not None and self._backend is not None
        assert isinstance(self._backend, MultiJointCapableBackend)

        out_dir: Path | None = None
        if output is not None:
            out_dir = Path(output).expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

        hashes = self._catalog.three_hash_record(self.config.profile)
        spec = SessionSpec(
            profile=self.config.profile,
            asset_manifest_sha256=self._catalog.manifest_sha256,
            teacher_sha256=self._catalog.teacher_sha256,
            raw_xml_sha256=self._catalog.raw_xml_sha256,
            profile_rule_sha256=hashes["profile_rule_sha256"],
            cache_mode=self._backend.describe().cache_mode,
        )

        t0 = time.perf_counter()
        knots_np, wf_np, exc_np = self.predict_nominal(targ, initial_qpos_delta_rad=delta_rad)
        max_hz, tau_ms, rate_scale_hz, clip_mode = self._bridge_params()

        sess = self._backend.open_session(spec)
        try:
            self._backend.reset(sess, InitialState(0.0), seed=bio_seed)
            ev = self._backend.evaluate_multijoint(
                sess,
                MultiJointEvaluationRequest(
                    controls=exc_np,
                    control_columns=LF_ACTUATOR_COLUMNS,
                    target_qpos_indices=LF_QPOS_ADDRESSES,
                    initial_qpos_delta_rad=delta_rad,
                    need_jacobian=False,
                    request_id="mj_nominal",
                ),
            )
            best_ev = ev
            best_knots = knots_np.copy()
            best_wf = wf_np.copy()
            best_exc = exc_np.copy()
            nominal_rmse = _mj_overall_rmse_deg(ev, targ)
            best_rmse = nominal_rmse
            steps_taken = 0

            if mode == "closed_loop":
                best_ev = self._backend.evaluate_multijoint_closed_loop(
                    sess,
                    MultiJointClosedLoopRequest(
                        feedforward_controls=best_exc,
                        target_rad=targ,
                        control_columns=LF_ACTUATOR_COLUMNS,
                        target_qpos_indices=LF_QPOS_ADDRESSES,
                        initial_qpos_delta_rad=delta_rad,
                        kp=feedback_kp,
                        kd=feedback_kd,
                        request_id="mj_closed_loop",
                    ),
                )
                best_rmse = _mj_overall_rmse_deg(best_ev, targ)
            elif mode in ("auto", "refine") and steps > 0 and (mode == "refine" or best_rmse > threshold):
                x_cur = best_knots.copy()
                s_hist: list[np.ndarray] = []
                y_hist: list[np.ndarray] = []
                prev_x: np.ndarray | None = None
                prev_g: np.ndarray | None = None
                for st in range(steps):
                    if mode == "auto" and best_rmse <= threshold:
                        break
                    ev_j = self._backend.evaluate_multijoint(
                        sess,
                        MultiJointEvaluationRequest(
                            controls=best_exc,
                            control_columns=LF_ACTUATOR_COLUMNS,
                            target_qpos_indices=LF_QPOS_ADDRESSES,
                            initial_qpos_delta_rad=delta_rad,
                            need_jacobian=True,
                            mse_target_rad=targ,
                            clip_boundary_gradient=clip_mode,
                            request_id=f"mj_vjp_{st}",
                        ),
                    )
                    assert ev_j.control_vjp is not None
                    kt = torch.tensor(x_cur[None, :, :], dtype=torch.float64, requires_grad=True)
                    wf_t = direct_multichannel_rate_waveform(kt.clamp(0.0, max_hz), max_rate_hz=max_hz)
                    exc_t = multichannel_rate_to_excitation(
                        wf_t,
                        max_rate_hz=max_hz,
                        rate_scale_hz=rate_scale_hz,
                        tau_ms=tau_ms,
                        clip_boundary_gradient=clip_mode,
                    )
                    exc_t.backward(torch.from_numpy(ev_j.control_vjp[None, :, :]))
                    assert kt.grad is not None
                    g_np = kt.grad[0].detach().cpu().numpy().copy()
                    if prev_x is not None and prev_g is not None:
                        sk = x_cur.reshape(-1) - prev_x
                        yk = g_np.reshape(-1) - prev_g
                        if float(np.dot(yk, sk)) > 1e-20:
                            s_hist.append(sk)
                            y_hist.append(yk)
                            if len(s_hist) > 5:
                                s_hist.pop(0)
                                y_hist.pop(0)
                    direction = lbfgs_direction(g_np, s_hist, y_hist, init_step_rms_hz=0.05, max_step_rms_hz=12.0)
                    improved = False
                    for alpha in (2.0, 1.0, 0.5, 0.25, 0.1):
                        cand_k = np.clip(x_cur + alpha * direction, 0.0, max_hz)
                        with torch.no_grad():
                            wf_c = direct_multichannel_rate_waveform(
                                torch.from_numpy(cand_k[None, :, :]), max_rate_hz=max_hz
                            )
                            exc_c = multichannel_rate_to_excitation(
                                wf_c,
                                max_rate_hz=max_hz,
                                rate_scale_hz=rate_scale_hz,
                                tau_ms=tau_ms,
                                clip_boundary_gradient=clip_mode,
                            )
                        ev_c = self._backend.evaluate_multijoint(
                            sess,
                            MultiJointEvaluationRequest(
                                controls=exc_c[0].cpu().numpy(),
                                control_columns=LF_ACTUATOR_COLUMNS,
                                target_qpos_indices=LF_QPOS_ADDRESSES,
                                initial_qpos_delta_rad=delta_rad,
                                request_id=f"mj_ls_{st}_{alpha}",
                            ),
                        )
                        rmse_c = _mj_overall_rmse_deg(ev_c, targ)
                        if rmse_c < best_rmse:
                            prev_x = x_cur.reshape(-1).copy()
                            prev_g = g_np.reshape(-1).copy()
                            x_cur = cand_k
                            best_knots = cand_k
                            best_wf = wf_c[0].cpu().numpy()
                            best_exc = exc_c[0].cpu().numpy()
                            best_ev = ev_c
                            best_rmse = rmse_c
                            steps_taken += 1
                            improved = True
                            break
                    if not improved:
                        break

            applied_ctrl = best_ev.traces["interval_applied_ctrl"][:, list(LF_ACTUATOR_COLUMNS)]
            self._backend.reset(sess, InitialState(0.0), seed=bio_seed)
            replay_ev = self._backend.evaluate_multijoint(
                sess,
                MultiJointEvaluationRequest(
                    controls=applied_ctrl,
                    control_columns=LF_ACTUATOR_COLUMNS,
                    target_qpos_indices=LF_QPOS_ADDRESSES,
                    initial_qpos_delta_rad=delta_rad,
                    request_id="mj_independent_replay",
                ),
            )
            for key in TRACE_KEYS:
                if not np.allclose(replay_ev.traces[key], best_ev.traces[key], rtol=1e-12, atol=1e-12):
                    raise BackendSessionError(f"Multi-joint independent replay diverged in {key!r}")

            # Biological realization evaluates the feedforward plan in a separate rollout.
            bio_summary: dict[str, Any] | None = None
            if bio_mode != "none":
                if progress_callback is not None:
                    progress_callback({"stage": "bio_realization", "bio_mode": bio_mode})
                bio_summary = evaluate_bio_realization(
                    backend=self._backend,
                    session=sess,
                    catalog=self._catalog,
                    scope="lf15",
                    command_rates_hz=best_wf,
                    target_excitation=best_exc,
                    target_rad=targ,
                    bio_mode=bio_mode,
                    tau_bridge_ms=tau_ms,
                    rate_scale_hz=rate_scale_hz,
                    initial_qpos_delta_rad=delta_rad,
                    seed=bio_seed,
                    population_scale=bio_population_scale,
                    physics_vjp_steps=bio_physics_vjp_steps,
                    output_dir=out_dir,
                )

            render_info = None
            if render and out_dir is not None and isinstance(self._backend, PersistentMujocoBackend):
                if progress_callback is not None:
                    progress_callback({"stage": "rendering"})
                rres = self._backend.render_multijoint_rollout(
                    sess,
                    applied_ctrl,
                    out_dir,
                    initial_qpos_delta_rad=delta_rad,
                    stem="multijoint_top_view",
                )
                render_info = rres.report
        finally:
            self._backend.close_session(sess)

        per_j_deg = np.degrees(
            np.sqrt(np.mean((best_ev.traces["qpos"][:, :7] - targ) ** 2, axis=0))
        )
        artifacts: dict[str, str] = {}
        result = {
            "schema_version": "sensory_translator_v2_multijoint_run_v1",
            "created_utc": utc_now_iso(),
            "profile": self.config.profile,
            "mode": mode,
            "model_id": self._model_id,
            "backend": self._backend.describe().name,
            "is_certified_frozen_physics": best_ev.is_certified_frozen_physics,
            "independent_replay_verified": True,
            "asset_hashes": {**hashes, **sess.identity.asset_hashes},
            "initial_qpos_delta_rad": delta_rad.tolist(),
            "model_checkpoint": {
                "filename": self._ckpt_path.name if self._ckpt_path else None,
                **self._checkpoint_identity(),
            },
            "scope": {
                "controlled_muscles": 15,
                "teacher_assisted_muscles": 0,
                "controlled_joints": list(LF_JOINT_NAMES),
                "joint_groups": {k: list(v) for k, v in LF_JOINT_GROUPS.items()},
                "actuator_names": list(LF_ACTUATOR_NAMES),
            },
            "nominal_rmse_deg": float(nominal_rmse),
            "best_rmse_deg": float(best_rmse),
            "accuracy_threshold_deg": float(threshold),
            "accuracy_threshold_met": bool(best_rmse <= threshold),
            "correction_steps_taken": int(steps_taken),
            "group_rmse_deg": {
                "coxa_rmse_deg": float(np.sqrt(np.mean(per_j_deg[list(LF_JOINT_GROUPS["coxa"])] ** 2))),
                "femur_rmse_deg": float(np.sqrt(np.mean(per_j_deg[list(LF_JOINT_GROUPS["femur_trochanter"])] ** 2))),
                "tibia_rmse_deg": float(per_j_deg[6]),
            },
            "per_joint_rmse_deg": {j: float(v) for j, v in zip(LF_JOINT_NAMES, per_j_deg)},
            "bio_interface": bio_summary,
            "render": render_info,
            "artifacts": artifacts,
            "elapsed_s": time.perf_counter() - t0,
        }
        if out_dir is not None:
            traces_npz = {key: best_ev.traces[key] for key in TRACE_KEYS}
            traces_npz["target_rad"] = targ
            traces_npz["rate_knots_hz"] = best_knots
            traces_npz["command_pool_rates_hz"] = best_wf
            traces_npz["feedforward_excitation"] = best_exc
            traces_npz["initial_qpos_delta_rad"] = delta_rad
            traces_npz["bridge_tau_ms"] = np.array(tau_ms, dtype=np.float64)
            write_npz_atomic(out_dir / "multijoint_traces.npz", traces_npz)
            artifacts["manifest"] = "multijoint_manifest.json"
            artifacts["traces"] = "multijoint_traces.npz"
            if isinstance(render_info, dict):
                for k, src_key in (
                    ("video_mp4", "mp4_path"),
                    ("video_webm", "webm_path"),
                    ("preview_png", "preview_png_path"),
                ):
                    raw = render_info.get(src_key)
                    if isinstance(raw, str) and Path(raw).is_file():
                        artifacts[k] = Path(raw).name
            if isinstance(bio_summary, dict) and isinstance(bio_summary.get("artifacts"), dict):
                for bk, bv in bio_summary["artifacts"].items():
                    artifacts[f"bio_{bk}"] = str(bv)
            result["artifacts"] = artifacts
            write_json_atomic(out_dir / "multijoint_manifest.json", result)
        if progress_callback is not None:
            progress_callback({"stage": "completed", "best_rmse_deg": result["best_rmse_deg"]})
        return result

    def bio_realize(
        self,
        command_rates_hz: np.ndarray,
        target_excitation: np.ndarray,
        target_rad: np.ndarray,
        *,
        bio_mode: str = "deterministic_isi",
        scope: str = "lf15",
        initial_qpos_delta_rad: np.ndarray | None = None,
        seed: int = 0,
        tau_bridge_ms: float | None = None,
        population_scale: int = 1,
        premotor_opt_steps: int = 40,
        physics_vjp_steps: int = 0,
        output: str | Path | None = None,
    ) -> dict[str, Any]:
        """Execute standalone biological spike / LIF / MaleCNS premotor realization."""
        with self._lock:
            self._ensure_init()
            assert self._catalog is not None and self._backend is not None
            _, default_tau_ms, rate_scale_hz, _ = self._bridge_params()
            tau_ms = (
                require_real_float(tau_bridge_ms, "tau_bridge_ms", positive=True)
                if tau_bridge_ms is not None
                else (20.0 if scope == "tibia" else default_tau_ms)
            )
            hashes = self._catalog.three_hash_record(self.config.profile)
            spec = SessionSpec(
                profile=self.config.profile,
                asset_manifest_sha256=self._catalog.manifest_sha256,
                teacher_sha256=self._catalog.teacher_sha256,
                raw_xml_sha256=self._catalog.raw_xml_sha256,
                profile_rule_sha256=hashes["profile_rule_sha256"],
                cache_mode=self._backend.describe().cache_mode,
            )
            sess = self._backend.open_session(spec)
            try:
                self._backend.reset(sess, InitialState(0.0), seed=int(seed))
                return evaluate_bio_realization(
                    backend=self._backend,
                    session=sess,
                    catalog=self._catalog,
                    scope=scope,
                    command_rates_hz=np.asarray(command_rates_hz, dtype=np.float64),
                    target_excitation=np.asarray(target_excitation, dtype=np.float64),
                    target_rad=np.asarray(target_rad, dtype=np.float64),
                    bio_mode=bio_mode,
                    initial_qpos_delta_rad=initial_qpos_delta_rad,
                    seed=int(seed),
                    tau_bridge_ms=tau_ms,
                    rate_scale_hz=rate_scale_hz,
                    population_scale=int(population_scale),
                    premotor_opt_steps=int(premotor_opt_steps),
                    physics_vjp_steps=int(physics_vjp_steps),
                    output_dir=output,
                )
            finally:
                self._backend.close_session(sess)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._backend is not None:
                self._backend.close()
                self._backend = None


def _mj_overall_rmse_deg(ev: Evaluation, target_rad: np.ndarray) -> float:
    q_sub = ev.traces["qpos"][:, list(LF_QPOS_ADDRESSES)]
    return float(np.degrees(np.sqrt(np.mean((q_sub - target_rad) ** 2))))


def evaluate_bio_realization(
    *,
    backend: MultiJointCapableBackend,
    session: Any,
    catalog: AssetCatalog,
    scope: str,
    command_rates_hz: np.ndarray,
    target_excitation: np.ndarray,
    target_rad: np.ndarray,
    bio_mode: str,
    initial_qpos_delta_rad: np.ndarray | None = None,
    seed: int = 0,
    tau_bridge_ms: float = 20.0,
    rate_scale_hz: float = 100.0,
    population_scale: int = 1,
    premotor_opt_steps: int = 40,
    physics_vjp_steps: int = 12,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Convert K=105 continuous rates -> spikes/LIF/MaleCNS connectome -> verified FlyMimic rollout."""
    if scope not in ("tibia", "lf15"):
        raise RequestValidationError(f"scope must be 'tibia' or 'lf15', got {scope!r}")

    pools = LF_TIBIA_POOLS if scope == "tibia" else resolve_motor_pools("lf15", disjoint_subbranches=True)
    mn_ids = MOTOR_BODY_IDS if scope == "tibia" else tuple(sorted({b for p in pools for b in p.body_ids}))
    cols = (13, 14) if scope == "tibia" else LF_ACTUATOR_COLUMNS
    qidxs = (6,) if scope == "tibia" else LF_QPOS_ADDRESSES
    t_arr = np.asarray(target_rad, dtype=np.float64).reshape(STATE_STEPS, len(qidxs))
    init_delta = initial_qpos_delta_rad if scope == "lf15" else None

    def _physics_vjp_step(exc_cand: np.ndarray) -> tuple[float, np.ndarray]:
        ev_vjp = backend.evaluate_multijoint(
            session,
            MultiJointEvaluationRequest(
                controls=np.clip(exc_cand, 0.0, 1.0),
                control_columns=cols,
                target_qpos_indices=qidxs,
                initial_qpos_delta_rad=init_delta,
                need_jacobian=True,
                mse_target_rad=t_arr,
                clip_boundary_gradient="one_sided",
                request_id=f"bio_{scope}_{bio_mode}_vjp",
            ),
        )
        q_cur = ev_vjp.traces["qpos"][:, list(qidxs)]
        rmse_deg = float(np.degrees(np.sqrt(np.mean((q_cur - t_arr) ** 2))))
        assert ev_vjp.control_vjp is not None
        return rmse_deg, ev_vjp.control_vjp

    # Express fast target_excitation on biological tau_bridge_ms (20ms) motor-neuron timescale
    rates_bio = deconvolve_target_excitation_to_rates(
        target_excitation, tau_ms=tau_bridge_ms, rate_scale_hz=rate_scale_hz
    )

    mn_spikes_arr: np.ndarray | None = None
    mn_u_arr: np.ndarray | None = None
    pre_spikes_arr: np.ndarray | None = None
    pre_u_arr: np.ndarray | None = None
    if bio_mode in ("deterministic_isi", "poisson_refractory"):
        if physics_vjp_steps > 0:
            res, vjp_hist = optimize_rate_spikes_with_physics_vjp(
                rates_bio,
                pools,
                mn_ids,
                _physics_vjp_step,
                method=bio_mode,
                population_scale=population_scale,
                tau_ms=tau_bridge_ms,
                rate_scale_hz=rate_scale_hz,
                steps=physics_vjp_steps,
                seed=seed,
            )
            meta = {**res.metadata(), "vjp_rmse_history_deg": list(vjp_hist)}
        else:
            res = realize_rate_spikes(
                rates_bio,
                pools,
                mn_ids,
                method=bio_mode,
                population_scale=population_scale,
                tau_ms=tau_bridge_ms,
                rate_scale_hz=rate_scale_hz,
                seed=seed,
            )
            meta = res.metadata()
        ctrl = res.excitation
        mn_spikes_arr = res.spike_counts
    elif bio_mode == "direct_lif":
        res_lif = realize_direct_lif_mn_drive(
            rates_bio,
            pools,
            mn_ids,
            tau_bridge_ms=tau_bridge_ms,
            rate_scale_hz=rate_scale_hz,
            calibrate_steps=30,
            physics_vjp_fn=_physics_vjp_step if physics_vjp_steps > 0 else None,
            physics_vjp_steps=physics_vjp_steps,
        )
        ctrl = res_lif.excitation
        mn_spikes_arr = res_lif.mn_spikes_1ms
        mn_u_arr = res_lif.mn_current_u_mv
        meta = res_lif.metadata()
    elif bio_mode == "premotor_connectome":
        conn = load_malecns_premotor_subnetwork(catalog.asset_root, scope=scope, disjoint_subbranches=True)
        res_pre = optimize_premotor_connectome_drive(
            target_excitation,
            conn,
            tau_bridge_ms=tau_bridge_ms,
            rate_scale_hz=rate_scale_hz,
            opt_steps=premotor_opt_steps,
            allow_hyperpolarizing_u=False,
            physics_vjp_fn=_physics_vjp_step if physics_vjp_steps > 0 else None,
            physics_vjp_steps=physics_vjp_steps,
        )
        ctrl = res_pre.excitation
        mn_spikes_arr = res_pre.mn_spikes_1ms
        mn_u_arr = res_pre.mn_current_u_mv
        pre_spikes_arr = res_pre.premotor_spikes_1ms
        pre_u_arr = res_pre.premotor_current_u_mv
        meta = {**res_pre.metadata(), "connectome": conn.summary()}
    else:
        raise RequestValidationError(f"Unsupported bio_mode: {bio_mode!r}")

    backend.reset(session, InitialState(0.0), seed=seed)
    ev_bio = backend.evaluate_multijoint(
        session,
        MultiJointEvaluationRequest(
            controls=ctrl,
            control_columns=cols,
            target_qpos_indices=qidxs,
            initial_qpos_delta_rad=init_delta,
            request_id=f"bio_{scope}_{bio_mode}",
        ),
    )
    q_bio = ev_bio.traces["qpos"][:, list(qidxs)]
    per_j_deg = np.degrees(np.sqrt(np.mean((q_bio - t_arr) ** 2, axis=0)))
    overall_deg = float(np.degrees(np.sqrt(np.mean((q_bio - t_arr) ** 2))))
    exc_rmse = float(np.sqrt(np.mean((ctrl - target_excitation) ** 2)))
    bio_artifacts: dict[str, str] = {}
    out_summary: dict[str, Any] = {
        "schema_version": "sensory_translator_v2_bio_realization_v1",
        "bio_mode": bio_mode,
        "interface_kind": meta.get("interface_kind"),
        "scope": scope,
        "tau_bridge_ms": float(tau_bridge_ms),
        "population_scale": int(population_scale),
        "is_certified_frozen_physics": ev_bio.is_certified_frozen_physics,
        "controlled_muscles": len(cols),
        "teacher_assisted_muscles": NU - len(cols),
        "excitation_rmse": exc_rmse,
        "overall_rmse_deg": overall_deg,
        "per_joint_rmse_deg": (
            {"joint_LFTibia_pitch": float(per_j_deg[0])}
            if scope == "tibia"
            else {j: float(v) for j, v in zip(LF_JOINT_NAMES, per_j_deg)}
        ),
        "details": meta,
        "artifacts": bio_artifacts,
    }
    if output_dir is not None:
        bio_dir = Path(output_dir).expanduser().resolve()
        bio_dir.mkdir(parents=True, exist_ok=True)
        npz_payload: dict[str, np.ndarray] = {
            "command_pool_rates_hz": np.asarray(command_rates_hz, dtype=np.float32),
            "target_excitation": np.asarray(target_excitation, dtype=np.float32),
            "realized_excitation": np.asarray(ctrl, dtype=np.float32),
            "target_rad": np.asarray(t_arr, dtype=np.float64),
            "actual_rad": np.asarray(q_bio, dtype=np.float64),
        }
        if mn_spikes_arr is not None:
            npz_payload["mn_spikes_1ms"] = np.asarray(mn_spikes_arr, dtype=np.float32)
        if mn_u_arr is not None:
            npz_payload["mn_current_u_mv"] = np.asarray(mn_u_arr, dtype=np.float32)
        if pre_spikes_arr is not None:
            npz_payload["premotor_spikes_1ms"] = np.asarray(pre_spikes_arr, dtype=np.float32)
        if pre_u_arr is not None:
            npz_payload["premotor_current_u_mv"] = np.asarray(pre_u_arr, dtype=np.float32)
        write_npz_atomic(bio_dir / "bio_traces.npz", npz_payload)
        bio_artifacts["manifest"] = "bio_realization.json"
        bio_artifacts["traces"] = "bio_traces.npz"
        write_json_atomic(bio_dir / "bio_realization.json", out_summary)
    return out_summary
