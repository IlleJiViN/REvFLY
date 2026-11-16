"""Generate held-out motion families from witnessed commands in the frozen V2 plant."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass

import numpy as np
import torch

from sensory_translator_v2.assets import AssetCatalog
from sensory_translator_v2.backends import PersistentMujocoBackend
from sensory_translator_v2.contracts import (
    CONTROL_STEPS,
    CONTROL_TIMESTEP_S,
    LF_QPOS_ADDRESSES,
    TRACE_KEYS,
    InitialState,
    MultiJointEvaluationRequest,
    SessionSpec,
)
from sensory_translator_v2.rate.bridge import (
    direct_multichannel_rate_waveform,
    multichannel_rate_to_excitation,
)


TRAINING_FAMILIES = (
    "swing_scaled",
    "decoupled_group_swing",
    "rhythmic_double_tap",
    "reach_and_hold",
)
FAMILY_DESCRIPTIONS = {
    "sequential_group_pulses": "Separate nonperiodic bursts fire coxa, femur, then tibia.",
    "chirped_antagonists": "Antagonistic drives accelerate on a smooth envelope with intergroup phase offsets.",
    "coactivation_release": "Muscles coactivate briefly, release below baseline, and rebound asymmetrically.",
}

# Signed actuator perturbation directions grouped in the frozen LF actuator order.
COXA_PATTERN = np.array([0.8, 0.55, -0.9, 0.9, 0.5, -0.8, -0.55])
FEMUR_PATTERN = np.array([0.9, -0.75, -0.35, 0.6, -0.95, 0.75])
TIBIA_PATTERN = np.array([0.9, -0.8])
GROUP_PATTERNS = (COXA_PATTERN, FEMUR_PATTERN, TIBIA_PATTERN)


@dataclass(frozen=True)
class ReachableTarget:
    name: str
    family: str
    target_rad: np.ndarray
    witness_knots_hz: np.ndarray
    witness_controls: np.ndarray
    initial_delta_rad: np.ndarray
    parameters: dict[str, float]


def _pulse(t: np.ndarray, center: float, half_width: float) -> np.ndarray:
    distance = np.abs(t - center) / half_width
    return 0.5 * (1.0 + np.cos(np.pi * np.minimum(distance, 1.0)))


def _sequence_pulses(t: np.ndarray, variant: int) -> tuple[np.ndarray, dict[str, float]]:
    variants = (
        ((0.18, 0.47, 0.76), (0.22, -0.19, 0.23), 0.14),
        ((0.23, 0.50, 0.80), (-0.18, 0.24, -0.19), 0.16),
        ((0.14, 0.40, 0.70), (0.26, 0.22, -0.24), 0.13),
    )
    centers, amplitudes, half_width = variants[variant]
    control = np.concatenate(
        [
            (amplitude * _pulse(t, center, half_width))[:, None] * pattern
            for center, amplitude, pattern in zip(centers, amplitudes, GROUP_PATTERNS)
        ],
        axis=1,
    )
    parameters = {"half_width_fraction": half_width}
    for label, center, amplitude in zip(("coxa", "femur", "tibia"), centers, amplitudes):
        parameters[f"{label}_center_fraction"] = center
        parameters[f"{label}_amplitude_fraction"] = amplitude
    return control, parameters


def _chirped_antagonists(t: np.ndarray, variant: int) -> tuple[np.ndarray, dict[str, float]]:
    variants = ((0.55, 5.10, 0.19), (1.20, 6.50, 0.21), (0.80, 4.20, 0.25))
    start, end, amplitude = variants[variant]
    phase = 2.0 * np.pi * (start * t + 0.5 * (end - start) * t**2)
    envelope = np.sin(np.pi * t) ** 2
    control = np.concatenate(
        [
            (amplitude * envelope * np.sin(phase + phase_offset))[:, None] * pattern
            for phase_offset, pattern in zip((0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0), GROUP_PATTERNS)
        ],
        axis=1,
    )
    return control, {"start_cycles_per_trial": start, "end_cycles_per_trial": end, "amplitude_fraction": amplitude}


def _coactivation_release(t: np.ndarray, variant: int) -> tuple[np.ndarray, dict[str, float]]:
    variants = ((0.27, 0.59, 0.24), (0.35, 0.71, 0.29), (0.47, 0.79, 0.32))
    squeeze_center, recoil_center, amplitude = variants[variant]
    coactivation = np.array([0.8, 0.75, 0.4, 0.8, 0.7, 0.7, 0.65, 0.8, 0.8, 0.5, 0.8, 0.55, 0.8, 1.0, 1.0])
    release = _pulse(t, squeeze_center + 0.19, 0.105)
    squeeze = _pulse(t, squeeze_center, 0.18)
    recoil = _pulse(t, recoil_center, 0.15)
    recoil_direction = np.concatenate((-COXA_PATTERN, FEMUR_PATTERN, -TIBIA_PATTERN))
    control = amplitude * (
        (squeeze - 0.60 * release)[:, None] * coactivation
        + 0.60 * recoil[:, None] * recoil_direction
    )
    return control, {
        "squeeze_center_fraction": squeeze_center,
        "recoil_center_fraction": recoil_center,
        "amplitude_fraction": amplitude,
    }


def generate_targets(
    catalog: AssetCatalog,
    *,
    profile: str = "adjacent_static_parent_v1",
    seed: int = 0,
    knots_per_pool: int = 105,
    max_rate_hz: float = 200.0,
    rate_scale_hz: float = 100.0,
    tau_ms: float = 2.0,
    control_bin_ms: int = 2,
) -> list[ReachableTarget]:
    """Generate nine targets from stock rate knots and verify the same physics replay exactly."""
    with np.load(catalog.teacher_path, allow_pickle=False) as teacher:
        teacher_controls = np.asarray(teacher["interval_actuator_ctrl"], dtype=np.float64)

    # Use just the teacher's temporal mean as a bias. The time-varying commands are new.
    baseline = 0.08 + 0.74 * teacher_controls.mean(axis=0)
    t = np.linspace(0.0, 1.0, knots_per_pool, dtype=np.float64)
    families = (
        ("sequential_group_pulses", _sequence_pulses),
        ("chirped_antagonists", _chirped_antagonists),
        ("coactivation_release", _coactivation_release),
    )
    targets: list[ReachableTarget] = []
    initial_delta = np.zeros(len(LF_QPOS_ADDRESSES), dtype=np.float64)
    duration_ms = round(CONTROL_STEPS * CONTROL_TIMESTEP_S * 1000)
    hashes = catalog.three_hash_record(profile)
    with closing(PersistentMujocoBackend(catalog, profile=profile)) as backend:
        session = backend.open_session(SessionSpec(
            profile=profile,
            asset_manifest_sha256=catalog.manifest_sha256,
            teacher_sha256=catalog.teacher_sha256,
            raw_xml_sha256=catalog.raw_xml_sha256,
            profile_rule_sha256=hashes["profile_rule_sha256"],
            cache_mode=backend.describe().cache_mode,
        ))
        try:
            for family, build in families:
                for variant in range(3):
                    name = f"{family}_{variant + 1}"
                    modulation, parameters = build(t, variant)
                    knots_hz = np.clip(rate_scale_hz * (baseline + modulation), 0.0, max_rate_hz).T.copy()
                    rates = direct_multichannel_rate_waveform(
                        torch.from_numpy(knots_hz[None]), duration_ms=duration_ms, max_rate_hz=max_rate_hz,
                    )
                    excitation = multichannel_rate_to_excitation(
                        rates,
                        rate_scale_hz=rate_scale_hz,
                        tau_ms=tau_ms,
                        control_bin_ms=control_bin_ms,
                        max_rate_hz=max_rate_hz,
                    )[0].numpy()
                    request = MultiJointEvaluationRequest(
                        controls=excitation, initial_qpos_delta_rad=initial_delta, request_id=name,
                    )
                    backend.reset(session, InitialState(0.0), seed=seed)
                    original = backend.evaluate_multijoint(session, request)
                    backend.reset(session, InitialState(0.0), seed=seed)
                    repeated = backend.evaluate_multijoint(session, request)
                    if not original.is_certified_frozen_physics or not repeated.is_certified_frozen_physics:
                        raise RuntimeError(f"Target {name} requires certified frozen physics")
                    for trace in TRACE_KEYS:
                        if not np.array_equal(original.traces[trace], repeated.traces[trace]):
                            raise RuntimeError(f"Witness replay for {name} differs in {trace}")
                    goal = original.traces["qpos"][:, LF_QPOS_ADDRESSES].copy()
                    if not np.isfinite(goal).all() or np.count_nonzero(np.ptp(np.rad2deg(goal), axis=0) >= 5.0) < 3:
                        raise RuntimeError(f"Target {name} must move at least three joints by 5 degrees")
                    targets.append(ReachableTarget(name, family, goal, knots_hz, excitation, initial_delta.copy(), parameters))
        finally:
            backend.close_session(session)
    return targets
