"""Standalone continuous motor-rate -> causal neuromuscular excitation bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from ..contracts import RateBoundaryError, require_int, require_real_float

MOTOR_BODY_IDS: tuple[int, ...] = (807165, 809912, 818057, 819384, 909831, 800636, 815344)
POOL_NAMES: tuple[str, str] = ("LFTibia_flex_93434", "LFTibia_extensor_93932")


@dataclass(frozen=True)
class MotorPool:
    """Single target muscle pool and its motor neuron body IDs."""

    name: str
    body_ids: tuple[int, ...]


LF_TIBIA_POOLS: tuple[MotorPool, MotorPool] = (
    MotorPool(name=POOL_NAMES[0], body_ids=MOTOR_BODY_IDS[:5]),
    MotorPool(name=POOL_NAMES[1], body_ids=MOTOR_BODY_IDS[5:]),
)

# Canonical 15-muscle FlyMimic LF motor pools ordered identically to MuJoCo actuators [0..14].
LF_15_POOLS: tuple[MotorPool, ...] = (
    MotorPool("LFC_tergopleural_promotor_a", (800061, 801173, 807281, 934091)),
    MotorPool("LFC_tergopleural_promotor_b", (800061, 801173, 807281, 934091)),
    MotorPool("LFC_pleural_remotor_and_abductor", (802509, 803803)),
    MotorPool("LFC_pleural_promotor", (800061, 801173, 807281, 934091)),
    MotorPool("LFC_sternal_anterior_rotator", (806923, 919509)),
    MotorPool("LFC_sternal_posterior_rotator", (801915, 804851, 808640, 1050335829)),
    MotorPool("LFC_sternal_adductor", (813480,)),
    MotorPool("LFF_trochanter_flexor_b", (810639, 814489, 818389, 823168, 823281, 828939, 1051367796, 1051670893)),
    MotorPool("LFF_sterno-tergo-trochanter_extensor_a", (801685, 802361, 802552, 803237, 804465, 805741)),
    MotorPool("LFF_sterno-tergo-trochanter_extensor_b", (801685, 802361, 802552, 803237, 804465, 805741)),
    MotorPool("LFF_accesory_trochanter_flexor", (821291, 909618, 911134)),
    MotorPool("LFF_trochanter_extensor", (803412, 807906)),
    MotorPool("LFF_trochanter_flexor_a", (810639, 814489, 818389, 823168, 823281, 828939, 1051367796, 1051670893)),
    MotorPool("LFTibia_flex_93434", MOTOR_BODY_IDS[:5]),
    MotorPool("LFTibia_extensor_93932", MOTOR_BODY_IDS[5:]),
)
LF_ALL_MOTOR_BODY_IDS: tuple[int, ...] = tuple(
    sorted({bid for pool in LF_15_POOLS for bid in pool.body_ids})
)


class _OneSidedExcitationClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normalized: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(normalized)
        return normalized.clamp(0.0, 1.0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (normalized,) = ctx.saved_tensors
        block_low = (normalized < 0.0) & (grad_output > 0.0)
        block_high = (normalized > 1.0) & (grad_output < 0.0)
        return grad_output.masked_fill(block_low | block_high, 0.0)


def _validate_pool_tensor(
    values: torch.Tensor,
    name: str,
    last_dim: int,
    *,
    max_val: float | None = None,
) -> torch.Tensor:
    if (
        not isinstance(values, torch.Tensor)
        or not values.is_floating_point()
        or values.ndim != 3
        or values.shape[0] < 1
        or values.shape[1] < 1
        or values.shape[2] != last_dim
        or not bool(torch.isfinite(values).all())
        or bool((values < 0.0).any())
    ):
        raise RateBoundaryError(
            f"{name} must be finite nonnegative floating tensor [B>=1,T>=1,{last_dim}]"
        )
    if max_val is not None and bool((values > max_val).any()):
        raise RateBoundaryError(f"{name} must lie inside [0, {max_val}]")
    return values.to(dtype=torch.float64)


def _interpolate_rate_knots(
    knots_hz: torch.Tensor,
    *,
    duration_ms: int = 420,
    max_rate_hz: float = 200.0,
    expected_pools: int | None = None,
) -> torch.Tensor:
    duration = require_int(duration_ms, "duration_ms", minimum=2)
    max_hz = require_real_float(max_rate_hz, "max_rate_hz", positive=True)
    if (
        not isinstance(knots_hz, torch.Tensor)
        or not knots_hz.is_floating_point()
        or knots_hz.ndim != 3
        or knots_hz.shape[0] < 1
        or knots_hz.shape[1] < 1
        or (expected_pools is not None and knots_hz.shape[1] != expected_pools)
        or knots_hz.shape[2] < 2
        or not bool(torch.isfinite(knots_hz).all())
    ):
        dim_desc = str(expected_pools) if expected_pools is not None else "P>=1"
        raise RateBoundaryError(f"knots_hz must be finite floating [B>=1, {dim_desc}, K>=2]")
    knots = knots_hz.to(dtype=torch.float64)
    if bool((knots < 0.0).any()) or bool((knots > max_hz).any()):
        raise RateBoundaryError(f"knots_hz must lie inside [0, {max_hz}]")
    knot_count = knots.shape[2]
    knot_times = np.linspace(0.0, float(duration), num=knot_count)
    sample_times = np.arange(duration, dtype=np.float64)
    right = np.clip(np.searchsorted(knot_times, sample_times, side="right"), 1, knot_count - 1)
    left = right - 1
    denom = knot_times[right] - knot_times[left]
    alpha_np = (sample_times - knot_times[left]) / denom
    left_idx = torch.as_tensor(left, device=knots.device, dtype=torch.long)
    right_idx = torch.as_tensor(right, device=knots.device, dtype=torch.long)
    alpha = torch.as_tensor(alpha_np, device=knots.device, dtype=torch.float64).view(1, duration, 1)
    y0 = knots.permute(0, 2, 1)[:, left_idx, :]
    y1 = knots.permute(0, 2, 1)[:, right_idx, :]
    return (y0 + alpha * (y1 - y0)).clamp(0.0, max_hz)


def direct_pool_rate_waveform(
    knots_hz: torch.Tensor,
    *,
    duration_ms: int = 420,
    max_rate_hz: float = 200.0,
) -> torch.Tensor:
    """Interpolate [B, 2, K>=2] rate knots at exact 1ms integer positions 0..duration_ms-1."""
    return _interpolate_rate_knots(
        knots_hz,
        duration_ms=duration_ms,
        max_rate_hz=max_rate_hz,
        expected_pools=2,
    )


def direct_multichannel_rate_waveform(
    knots_hz: torch.Tensor,
    *,
    duration_ms: int = 420,
    max_rate_hz: float = 200.0,
) -> torch.Tensor:
    """Interpolate [B, P>=1, K>=2] rate knots at exact 1ms integer positions 0..duration_ms-1."""
    return _interpolate_rate_knots(
        knots_hz,
        duration_ms=duration_ms,
        max_rate_hz=max_rate_hz,
        expected_pools=None,
    )


def rates_to_motor_counts(
    pool_rates_hz: torch.Tensor,
    *,
    max_rate_hz: float = 200.0,
    motor_body_ids: Sequence[int] = MOTOR_BODY_IDS,
) -> torch.Tensor:
    """Map [B, T_ms, 2] flexor/extensor rates (Hz) to expected 1ms motor counts [B, T_ms, 7]."""
    max_hz = require_real_float(max_rate_hz, "max_rate_hz", positive=True)
    ids = tuple(motor_body_ids)
    if ids != MOTOR_BODY_IDS:
        raise RateBoundaryError(f"motor_body_ids must equal frozen {MOTOR_BODY_IDS}")
    rates = _validate_pool_tensor(pool_rates_hz, "pool_rates_hz", 2, max_val=max_hz)
    # Pools map columns: first 5 motors -> pool 0, last 2 motors -> pool 1
    col_indices = torch.tensor([0, 0, 0, 0, 0, 1, 1], device=rates.device, dtype=torch.long)
    return rates[:, :, col_indices] / 1000.0


def _causal_filter_to_excitation(
    counts: torch.Tensor,
    *,
    rate_scale_hz: float,
    tau_ms: float,
    control_bin_ms: int,
    initial_rate_hz: np.ndarray | torch.Tensor | None,
    clip_boundary_gradient: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = require_real_float(rate_scale_hz, "rate_scale_hz", positive=True)
    tau = require_real_float(tau_ms, "tau_ms", positive=True)
    width = require_int(control_bin_ms, "control_bin_ms", minimum=1)
    if clip_boundary_gradient not in ("native", "one_sided"):
        raise RateBoundaryError("clip_boundary_gradient must be 'native' or 'one_sided'")
    batch, duration, width_c = counts.shape
    if duration % width != 0:
        raise RateBoundaryError("Neural duration must align to complete control intervals")

    if initial_rate_hz is None:
        state = counts.new_zeros((batch, width_c))
    else:
        raw = torch.as_tensor(initial_rate_hz, device=counts.device, dtype=torch.float64)
        if raw.shape != (batch, width_c) or not bool(torch.isfinite(raw).all()) or bool((raw < 0).any()):
            raise RateBoundaryError(f"initial_rate_hz must be finite nonnegative [B, {width_c}]")
        state = raw

    decay = float(np.exp(-1.0 / tau))
    increment_hz = (1.0 - decay) * 1000.0
    boundaries = [state]
    for idx in range(duration):
        state = decay * state + increment_hz * counts[:, idx]
        boundaries.append(state)
    rates = torch.stack(boundaries, dim=1)
    return rates, counts.new_tensor(scale)


def _normalize_pool_excitation(
    pool_rates: torch.Tensor,
    *,
    rate_scale_hz: float,
    control_bin_ms: int,
    clip_boundary_gradient: str,
) -> torch.Tensor:
    duration = pool_rates.shape[1] - 1
    normalized = pool_rates / float(rate_scale_hz)
    if clip_boundary_gradient == "one_sided":
        pool_excitation = _OneSidedExcitationClip.apply(normalized)
    else:
        pool_excitation = normalized.clamp(0.0, 1.0)
    return pool_excitation[:, :duration:control_bin_ms]


def differentiable_bridge_motor_spikes(
    spike_counts: torch.Tensor,
    *,
    rate_scale_hz: float = 100.0,
    tau_ms: float = 20.0,
    control_bin_ms: int = 2,
    initial_rate_hz: np.ndarray | torch.Tensor | None = None,
    motor_body_ids: Sequence[int] = MOTOR_BODY_IDS,
    return_rates: bool = False,
    clip_boundary_gradient: str = "native",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Differentiable fixed causal filter mapping [B, T_ms, 7] counts to [B, T_ms//2, 2]."""
    ids = tuple(motor_body_ids)
    if ids != MOTOR_BODY_IDS:
        raise RateBoundaryError(f"motor_body_ids must equal frozen {MOTOR_BODY_IDS}")
    counts = _validate_pool_tensor(spike_counts, "spike_counts", len(ids))
    rates, _ = _causal_filter_to_excitation(
        counts,
        rate_scale_hz=rate_scale_hz,
        tau_ms=tau_ms,
        control_bin_ms=control_bin_ms,
        initial_rate_hz=initial_rate_hz,
        clip_boundary_gradient=clip_boundary_gradient,
    )
    pool_rates = torch.stack(
        [
            rates[..., [ids.index(bid) for bid in pool.body_ids]].mean(dim=-1)
            for pool in LF_TIBIA_POOLS
        ],
        dim=-1,
    )
    excitation = _normalize_pool_excitation(
        pool_rates,
        rate_scale_hz=rate_scale_hz,
        control_bin_ms=control_bin_ms,
        clip_boundary_gradient=clip_boundary_gradient,
    )
    return (excitation, rates) if return_rates else excitation


def differentiable_bridge_pool_spikes(
    spike_counts: torch.Tensor,
    pools: Sequence[MotorPool],
    motor_body_ids: Sequence[int],
    *,
    rate_scale_hz: float = 100.0,
    tau_ms: float = 20.0,
    control_bin_ms: int = 2,
    initial_rate_hz: np.ndarray | torch.Tensor | None = None,
    return_rates: bool = False,
    clip_boundary_gradient: str = "one_sided",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Differentiable causal filter mapping [B, T_ms, N_mn] spike counts to [B, T_ms//bin, len(pools)]."""
    if not pools:
        raise RateBoundaryError("pools must be non-empty")
    ids = tuple(motor_body_ids)
    id_to_col = {bid: i for i, bid in enumerate(ids)}
    counts = _validate_pool_tensor(spike_counts, "spike_counts", len(ids))
    rates, _ = _causal_filter_to_excitation(
        counts,
        rate_scale_hz=rate_scale_hz,
        tau_ms=tau_ms,
        control_bin_ms=control_bin_ms,
        initial_rate_hz=initial_rate_hz,
        clip_boundary_gradient=clip_boundary_gradient,
    )
    pool_rates = torch.stack(
        [
            rates[..., [id_to_col[bid] for bid in pool.body_ids]].mean(dim=-1)
            for pool in pools
        ],
        dim=-1,
    )
    excitation = _normalize_pool_excitation(
        pool_rates,
        rate_scale_hz=rate_scale_hz,
        control_bin_ms=control_bin_ms,
        clip_boundary_gradient=clip_boundary_gradient,
    )
    return (excitation, rates) if return_rates else excitation


def rate_to_excitation(
    pool_rates_hz: torch.Tensor,
    *,
    max_rate_hz: float = 200.0,
    rate_scale_hz: float = 100.0,
    tau_ms: float = 20.0,
    control_bin_ms: int = 2,
    clip_boundary_gradient: str = "native",
) -> torch.Tensor:
    """Convert [B, T_ms, 2] pool rates directly into [B, T_ms//2, 2] actuator excitation."""
    counts = rates_to_motor_counts(pool_rates_hz, max_rate_hz=max_rate_hz)
    return differentiable_bridge_motor_spikes(
        counts,
        rate_scale_hz=rate_scale_hz,
        tau_ms=tau_ms,
        control_bin_ms=control_bin_ms,
        clip_boundary_gradient=clip_boundary_gradient,
    )


def multichannel_rate_to_excitation(
    pool_rates_hz: torch.Tensor,
    *,
    max_rate_hz: float = 200.0,
    rate_scale_hz: float = 100.0,
    tau_ms: float = 2.0,
    control_bin_ms: int = 2,
    clip_boundary_gradient: str = "one_sided",
    return_rates: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Convert [B, T_ms, P] motor pool rates directly into [B, T_ms//2, P] excitation."""
    max_hz = require_real_float(max_rate_hz, "max_rate_hz", positive=True)
    if not isinstance(pool_rates_hz, torch.Tensor) or pool_rates_hz.ndim != 3 or pool_rates_hz.shape[2] < 1:
        raise RateBoundaryError("pool_rates_hz must have shape [B>=1, T>=1, P>=1]")
    rates = _validate_pool_tensor(pool_rates_hz, "pool_rates_hz", int(pool_rates_hz.shape[2]), max_val=max_hz)
    counts = rates / 1000.0
    filtered_rates, _ = _causal_filter_to_excitation(
        counts,
        rate_scale_hz=rate_scale_hz,
        tau_ms=tau_ms,
        control_bin_ms=control_bin_ms,
        initial_rate_hz=None,
        clip_boundary_gradient=clip_boundary_gradient,
    )
    excitation = _normalize_pool_excitation(
        filtered_rates,
        rate_scale_hz=rate_scale_hz,
        control_bin_ms=control_bin_ms,
        clip_boundary_gradient=clip_boundary_gradient,
    )
    return (excitation, filtered_rates) if return_rates else excitation


@dataclass
class MotorRateProtocol:
    """Structured continuous motor-rate protocol metadata for saved artifacts."""

    goal_id: str
    command_pool_rates_hz: np.ndarray
    behavior_simulation_evaluated: bool = False
    joint_rmse_deg: float | None = None
    accuracy_threshold_deg: float = 5.729577951308233
    max_rate_hz: float = 200.0
    tau_ms: float = 20.0
    rate_scale_hz: float = 100.0
    clip_boundary: str = "native"
    protocol_kind: str = "continuous_pool_rate_waveform"
    extra_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        rates = np.asarray(self.command_pool_rates_hz, dtype=np.float64)
        payload = {
            "schema_version": "motor_rate_protocol_v1",
            "protocol_kind": self.protocol_kind,
            "goal_id": self.goal_id,
            "duration_ms": int(rates.shape[0]),
            "rate_unit": "Hz",
            "max_command_rate_hz": float(self.max_rate_hz),
            "rate_bound_status": (
                "within_bounds"
                if bool((rates >= 0.0).all() and (rates <= self.max_rate_hz).all())
                else "out_of_bounds"
            ),
            "rate_interpretation": "continuous flexor/extensor motor pool command rate",
            "command_pool_rates_hz": rates.tolist(),
            "motor_body_ids": list(MOTOR_BODY_IDS),
            "pool_names": list(POOL_NAMES),
            "pool_members": {pool.name: list(pool.body_ids) for pool in LF_TIBIA_POOLS},
            "motor_profile_representation": "pool_uniform_expected_counts",
            "individual_motor_profiles_identified": False,
            "non_unique_solution": True,
            "expected_counts_are_discrete_spikes": False,
            "bridge": {
                "tau_ms": float(self.tau_ms),
                "rate_scale_hz": float(self.rate_scale_hz),
                "control_bin_ms": 2,
                "clip_boundary": self.clip_boundary,
            },
            "behavior_simulation_evaluated": bool(self.behavior_simulation_evaluated),
            "joint_rmse_deg": self.joint_rmse_deg,
            "accuracy_threshold_deg": float(self.accuracy_threshold_deg),
            "joint_accuracy_threshold_met": (
                None
                if self.joint_rmse_deg is None
                else bool(self.joint_rmse_deg <= self.accuracy_threshold_deg)
            ),
            "spiking_feasibility_validated": False,
            "physical_current_calibrated": False,
            "biological_motor_unit_calibrated": False,
        }
        if self.extra_metadata:
            payload.update(self.extra_metadata)
        return payload
