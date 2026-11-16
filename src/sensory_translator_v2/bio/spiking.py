"""Discrete Poisson & refractory spike-train realization from continuous K=105 motor rates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..contracts import RateBoundaryError, require_int, require_real_float
from ..rate.bridge import MotorPool, differentiable_bridge_pool_spikes


@dataclass(frozen=True)
class SpikeRealizationResult:
    """Discrete 1ms motor-neuron spike trains and bridged FlyMimic excitations."""

    method: str
    mn_body_ids: tuple[int, ...]
    pools: tuple[MotorPool, ...]
    spike_counts: np.ndarray  # [T_ms, N_mn]
    excitation: np.ndarray    # [T_ms // control_bin_ms, len(pools)]
    pool_realized_rates_hz: np.ndarray  # [T_ms, len(pools)]
    total_spikes: int
    min_observed_isi_ms: float | None
    population_scale: int
    tau_ms: float
    rate_scale_hz: float

    def metadata(self) -> dict[str, Any]:
        per_pool_spikes = []
        id_to_col = {bid: i for i, bid in enumerate(self.mn_body_ids)}
        for pool in self.pools:
            cols = [id_to_col[bid] for bid in pool.body_ids]
            per_pool_spikes.append(int(self.spike_counts[:, cols].sum()))
        return {
            "method": self.method,
            "num_motor_neurons": len(self.mn_body_ids),
            "num_pools": len(self.pools),
            "total_spikes": self.total_spikes,
            "per_pool_spikes": per_pool_spikes,
            "min_observed_isi_ms": self.min_observed_isi_ms,
            "population_scale": self.population_scale,
            "tau_ms": self.tau_ms,
            "rate_scale_hz": self.rate_scale_hz,
        }


def realize_rate_spikes(
    pool_rates_hz: np.ndarray,
    pools: Sequence[MotorPool],
    mn_body_ids: Sequence[int],
    *,
    method: str = "deterministic_isi",
    min_isi_ms: float = 2.2,
    population_scale: int = 1,
    tau_ms: float = 20.0,
    rate_scale_hz: float = 100.0,
    control_bin_ms: int = 2,
    clip_boundary_gradient: str = "one_sided",
    seed: int = 0,
) -> SpikeRealizationResult:
    """Convert [T_ms, P] continuous rate commands into [T_ms, N_mn] spike counts + excitations."""
    if method not in ("deterministic_isi", "poisson_refractory"):
        raise RateBoundaryError(f"method must be 'deterministic_isi' or 'poisson_refractory', got {method!r}")
    rates = np.asarray(pool_rates_hz, dtype=np.float64)
    if rates.ndim != 2 or rates.shape[1] != len(pools) or not bool(np.isfinite(rates).all()) or bool((rates < 0.0).any()):
        raise RateBoundaryError(f"pool_rates_hz must be finite nonnegative [T_ms, {len(pools)}]")
    duration_ms = int(rates.shape[0])
    min_isi = require_real_float(min_isi_ms, "min_isi_ms", nonnegative=True)
    pop_scale = require_int(population_scale, "population_scale", minimum=1)
    tau = require_real_float(tau_ms, "tau_ms", positive=True)
    scale_hz = require_real_float(rate_scale_hz, "rate_scale_hz", positive=True)
    bin_ms = require_int(control_bin_ms, "control_bin_ms", minimum=1)
    seed_int = require_int(seed, "seed", minimum=0)

    ids = tuple(int(b) for b in mn_body_ids)
    id_to_col = {bid: idx for idx, bid in enumerate(ids)}
    col_to_pool: dict[int, int] = {}
    col_sub_idx: dict[int, int] = {}
    pool_sizes: list[int] = [len(p.body_ids) for p in pools]
    for p_idx, pool in enumerate(pools):
        for local_i, bid in enumerate(pool.body_ids):
            col = id_to_col[bid]
            if col not in col_to_pool:
                col_to_pool[col] = p_idx
                col_sub_idx[col] = local_i

    n_mn = len(ids)
    spikes = np.zeros((duration_ms, n_mn), dtype=np.float64)
    min_bins_between = max(1, int(np.ceil(min_isi))) if pop_scale == 1 else 1
    rng = np.random.default_rng(seed_int)

    if method == "deterministic_isi":
        for c in range(n_mn):
            p_idx = col_to_pool[c]
            local_i = col_sub_idx[c]
            n_members = max(1, pool_sizes[p_idx])
            r_col = rates[:, p_idx]
            for replica in range(pop_scale):
                phase = ((local_i + replica / float(pop_scale)) / float(n_members)) % 1.0
                last_spike = -10_000
                for t in range(duration_ms):
                    phase += r_col[t] * 0.001
                    if phase >= 1.0 and (t - last_spike) >= min_bins_between:
                        spikes[t, c] += 1.0
                        phase -= 1.0
                        last_spike = t
    else:
        for c in range(n_mn):
            p_idx = col_to_pool[c]
            r_col = rates[:, p_idx]
            probs = np.clip(r_col * 0.001, 0.0, 1.0)
            if pop_scale == 1:
                last_spike = -10_000
                draws = rng.random(duration_ms)
                for t in range(duration_ms):
                    if (t - last_spike) >= min_bins_between and draws[t] < probs[t]:
                        spikes[t, c] = 1.0
                        last_spike = t
            else:
                spikes[:, c] = rng.binomial(pop_scale, probs).astype(np.float64)

    scaled_counts = spikes / float(pop_scale)
    spikes_t = torch.from_numpy(scaled_counts).unsqueeze(0)
    with torch.no_grad():
        exc_t, filtered_rates_t = differentiable_bridge_pool_spikes(
            spikes_t,
            pools,
            ids,
            rate_scale_hz=scale_hz,
            tau_ms=tau,
            control_bin_ms=bin_ms,
            return_rates=True,
            clip_boundary_gradient=clip_boundary_gradient,
        )
    filtered_np = filtered_rates_t[0].cpu().numpy()
    pool_realized = np.stack(
        [
            filtered_np[:-1, [id_to_col[bid] for bid in p.body_ids]].mean(axis=-1)
            for p in pools
        ],
        axis=-1,
    )
    min_gap: float | None = None
    if pop_scale == 1:
        for c in range(n_mn):
            ts = np.flatnonzero(spikes[:, c] > 0.5)
            if ts.size >= 2:
                g = float(np.diff(ts).min())
                min_gap = g if min_gap is None else min(min_gap, g)

    return SpikeRealizationResult(
        method=method,
        mn_body_ids=ids,
        pools=tuple(pools),
        spike_counts=spikes,
        excitation=exc_t[0].cpu().numpy(),
        pool_realized_rates_hz=pool_realized,
        total_spikes=int(spikes.sum()),
        min_observed_isi_ms=min_gap,
        population_scale=pop_scale,
        tau_ms=tau,
        rate_scale_hz=scale_hz,
    )


def deconvolve_target_excitation_to_rates(
    target_excitation: np.ndarray,
    *,
    tau_ms: float = 20.0,
    rate_scale_hz: float = 100.0,
    max_rate_hz: float = 200.0,
    control_bin_ms: int = 2,
) -> np.ndarray:
    """Invert target [210, P] excitation into [420, P] target command rates for arbitrary tau_ms.

    Wait: useful when bridging fast tau=2ms plans into slower biological tau=20ms NMJ filters!
    Because x_{t+1} = d*x_t + (1-d)*(r_t / scale), solving forward/least-squares recovers
    causal command rates r_t in [0, max_rate_hz] whose slower tau_ms filtered output matches target_excitation.
    """
    exc = np.asarray(target_excitation, dtype=np.float64)
    n_bins, n_pools = exc.shape
    duration_ms = n_bins * control_bin_ms
    # Upsample desired excitation smoothly to 1ms steps and apply inverse first-order filter with Tikhonov smoothing
    exc_1ms = np.repeat(exc, control_bin_ms, axis=0)
    exc_next = np.vstack([exc_1ms[1:], exc_1ms[-1:]])
    d = float(np.exp(-1.0 / tau_ms))
    raw_rates = rate_scale_hz * (exc_next - d * exc_1ms) / (1.0 - d)
    # Mild causal binomial smoothing removes 2ms ZOH staircasing before LIF/Poisson generation
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    padded = np.pad(raw_rates, ((1, 1), (0, 0)), mode="edge")
    smoothed = kernel[0] * padded[:-2] + kernel[1] * padded[1:-1] + kernel[2] * padded[2:]
    return np.clip(smoothed, 0.0, max_rate_hz)


def optimize_rate_spikes_with_physics_vjp(
    initial_rates_hz: np.ndarray,
    pools: Sequence[MotorPool],
    mn_body_ids: Sequence[int],
    physics_vjp_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
    *,
    method: str = "deterministic_isi",
    min_isi_ms: float = 2.2,
    population_scale: int = 1,
    tau_ms: float = 20.0,
    rate_scale_hz: float = 100.0,
    max_rate_hz: float = 250.0,
    control_bin_ms: int = 2,
    steps: int = 12,
    lr: float = 10.0,
    seed: int = 0,
) -> tuple[SpikeRealizationResult, tuple[float, ...]]:
    """Refine K=105 rate knots directly against MuJoCo control_vjp using hard-spike forward + smooth STE."""
    init_np = np.clip(np.asarray(initial_rates_hz, dtype=np.float64), 0.0, max_rate_hz)
    duration_ms, n_pools = init_np.shape
    ids = tuple(int(b) for b in mn_body_ids)
    id_to_col = {bid: i for i, bid in enumerate(ids)}
    pool_map_idx = torch.zeros((len(ids),), dtype=torch.long)
    for p_idx, pool in enumerate(pools):
        for bid in pool.body_ids:
            pool_map_idx[id_to_col[int(bid)]] = p_idx

    n_knots = 105
    knots_init = F.interpolate(
        torch.from_numpy(init_np.T[None, :, :]),
        size=n_knots,
        mode="linear",
        align_corners=True,
    )[0]  # [P, 105]
    rate_knots = knots_init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([rate_knots], lr=lr)

    best_res: SpikeRealizationResult | None = None
    best_rmse = float("inf")
    rmse_hist: list[float] = []

    for _ in range(max(1, steps)):
        rates_1ms_t = F.interpolate(
            rate_knots.clamp(0.0, max_rate_hz).unsqueeze(0),
            size=duration_ms,
            mode="linear",
            align_corners=True,
        )[0].T  # [T_ms, P]
        res = realize_rate_spikes(
            rates_1ms_t.detach().cpu().numpy(),
            pools,
            ids,
            method=method,
            min_isi_ms=min_isi_ms,
            population_scale=population_scale,
            tau_ms=tau_ms,
            rate_scale_hz=rate_scale_hz,
            control_bin_ms=control_bin_ms,
            clip_boundary_gradient="one_sided",
            seed=seed,
        )
        rmse_deg, vjp_np = physics_vjp_fn(res.excitation)
        rmse_hist.append(float(rmse_deg))
        if float(rmse_deg) < best_rmse:
            best_rmse = float(rmse_deg)
            best_res = res

        opt.zero_grad()
        mn_rates_t = (rates_1ms_t / 1000.0)[:, pool_map_idx].unsqueeze(0)
        exc_smooth = differentiable_bridge_pool_spikes(
            mn_rates_t,
            pools,
            ids,
            rate_scale_hz=rate_scale_hz,
            tau_ms=tau_ms,
            control_bin_ms=control_bin_ms,
            clip_boundary_gradient="one_sided",
        )
        smooth_loss = 2e-7 * (rate_knots[:, 1:] - rate_knots[:, :-1]).pow(2).mean()
        exc_smooth.backward(torch.from_numpy(np.asarray(vjp_np, dtype=np.float64)[None, :, :]))
        smooth_loss.backward()
        torch.nn.utils.clip_grad_norm_([rate_knots], 20.0)
        opt.step()
        with torch.no_grad():
            rate_knots.clamp_(0.0, max_rate_hz)

    # Final hard check after last update
    with torch.no_grad():
        final_rates = F.interpolate(
            rate_knots.clamp(0.0, max_rate_hz).unsqueeze(0),
            size=duration_ms,
            mode="linear",
            align_corners=True,
        )[0].T.cpu().numpy()
    final_res = realize_rate_spikes(
        final_rates,
        pools,
        ids,
        method=method,
        min_isi_ms=min_isi_ms,
        population_scale=population_scale,
        tau_ms=tau_ms,
        rate_scale_hz=rate_scale_hz,
        control_bin_ms=control_bin_ms,
        clip_boundary_gradient="one_sided",
        seed=seed,
    )
    final_rmse, _ = physics_vjp_fn(final_res.excitation)
    rmse_hist.append(float(final_rmse))
    if float(final_rmse) < best_rmse or best_res is None:
        best_res = final_res
    return best_res, tuple(rmse_hist)
