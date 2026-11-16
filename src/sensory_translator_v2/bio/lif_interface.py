"""Biological LIF motor-neuron & MaleCNS premotor connectome drive interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..contracts import RateBoundaryError
from ..rate.bridge import MotorPool, differentiable_bridge_pool_spikes
from .malecns_pools import PremotorConnectome


@dataclass(frozen=True)
class LIFBiophysicsParams:
    """Frozen biophysical LIF & synaptic parameters matching Drosophila connectome kinetics."""

    dt_ms: float = 0.2
    v_rest_mv: float = -52.0
    v_reset_mv: float = -52.0
    v_thresh_mv: float = -45.0
    tau_mem_ms: float = 20.0
    tau_syn_ms: float = 5.0
    refractory_ms: float = 2.2
    delay_ms: float = 1.8
    w_syn: float = 0.125


def _surrogate_spike(v_diff: torch.Tensor, beta: float = 2.5) -> torch.Tensor:
    """Heaviside forward spike emission with SuperSpike fast-sigmoid surrogate gradient."""
    forward = (v_diff >= 0.0).to(dtype=v_diff.dtype)
    # Fast-sigmoid surrogate derivative sigma'(x) = 1 / (1 + beta * |x|)^2
    surr = v_diff / (1.0 + beta * torch.abs(v_diff))
    # Normalized so integral matches smooth transition; STE passes exact binary spikes in forward
    return forward + (surr - surr.detach())


def analytical_lif_current_mv(
    rate_hz: np.ndarray | torch.Tensor,
    params: LIFBiophysicsParams = LIFBiophysicsParams(),
    *,
    max_current_mv: float = 60.0,
) -> np.ndarray | torch.Tensor:
    """Invert passive LIF steady-state f-I curve: desired rate (Hz) -> u = R_m I_ext (mV)."""
    is_torch = isinstance(rate_hz, torch.Tensor)
    dv_th = params.v_thresh_mv - params.v_rest_mv
    if is_torch:
        r = rate_hz.to(dtype=torch.float64).clamp_min(0.0)
        r_safe = r.clamp_min(1e-6)
        t_int = (1000.0 / r_safe) - params.refractory_ms
        t_int = t_int.clamp_min(params.dt_ms)
        denom = 1.0 - torch.exp(-t_int / params.tau_mem_ms)
        u = dv_th / denom.clamp_min(1e-6)
        # Smooth sub-threshold fade below 1 Hz so 0 Hz produces sub-threshold rest (0 mV)
        gate = torch.clamp(r / 2.0, 0.0, 1.0)
        return (u * gate).clamp(0.0, max_current_mv)

    r_np = np.maximum(np.asarray(rate_hz, dtype=np.float64), 0.0)
    r_safe = np.maximum(r_np, 1e-6)
    t_int = np.maximum((1000.0 / r_safe) - params.refractory_ms, params.dt_ms)
    denom = np.maximum(1.0 - np.exp(-t_int / params.tau_mem_ms), 1e-6)
    u_np = dv_th / denom
    gate_np = np.clip(r_np / 2.0, 0.0, 1.0)
    return np.clip(u_np * gate_np, 0.0, max_current_mv)


def _upsample_1ms_to_substeps(x_1ms: torch.Tensor, substeps_per_ms: int) -> torch.Tensor:
    """Convert [T_ms, N] (or [B, T_ms, N]) smoothly or ZOH to [T_sub, N]."""
    return torch.repeat_interleave(x_1ms, substeps_per_ms, dim=-2)


def simulate_lif_substep_trace(
    u_ext_sub: torch.Tensor,
    *,
    params: LIFBiophysicsParams = LIFBiophysicsParams(),
    w_rec: torch.Tensor | None = None,
    g_ext_sub: torch.Tensor | None = None,
    v_init: torch.Tensor | None = None,
    surrogate_grad: bool = False,
    beta: float = 2.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run exact dt=0.2ms LIF dynamics over [T_sub, N] substeps and return (spikes_1ms, v_sub)."""
    t_sub, n_neurons = u_ext_sub.shape
    sub_per_ms = max(1, int(round(1.0 / params.dt_ms)))
    if t_sub % sub_per_ms != 0:
        raise RateBoundaryError(f"u_ext_sub length {t_sub} must be a multiple of {sub_per_ms}")
    duration_ms = t_sub // sub_per_ms

    dtype = u_ext_sub.dtype
    device = u_ext_sub.device
    v = (
        torch.full((n_neurons,), params.v_rest_mv, dtype=dtype, device=device)
        if v_init is None
        else v_init.to(dtype=dtype, device=device).clone()
    )
    g = torch.zeros((n_neurons,), dtype=dtype, device=device)
    ref_steps = torch.zeros((n_neurons,), dtype=torch.int64, device=device)
    rfc_count = max(1, int(round(params.refractory_ms / params.dt_ms)))
    delay_steps = max(0, int(round(params.delay_ms / params.dt_ms)))

    decay_syn = float(np.exp(-params.dt_ms / params.tau_syn_ms))
    alpha_mem = params.dt_ms / params.tau_mem_ms

    v_reset_t = torch.full((n_neurons,), params.v_reset_mv, dtype=dtype, device=device)
    rfc_full = torch.full((n_neurons,), rfc_count, dtype=torch.int64, device=device)
    zero_spk = torch.zeros_like(v)

    delay_buf: list[torch.Tensor] = (
        [torch.zeros((n_neurons,), dtype=dtype, device=device) for _ in range(delay_steps + 1)]
        if (w_rec is not None and delay_steps > 0)
        else []
    )
    spikes_sub: list[torch.Tensor] = []
    v_hist: list[torch.Tensor] = []
    v_drive_sub = params.v_rest_mv + u_ext_sub

    for step in range(t_sub):
        if w_rec is not None:
            delayed = (
                delay_buf[step % len(delay_buf)]
                if delay_steps > 0
                else (spikes_sub[-1] if spikes_sub else zero_spk)
            )
            g = decay_syn * g + params.w_syn * (w_rec @ delayed)
        if g_ext_sub is not None:
            g = decay_syn * g + params.w_syn * g_ext_sub[step]

        active_mask = (ref_steps <= 0).to(dtype=dtype)
        dv = alpha_mem * (v_drive_sub[step] + g - v) * active_mask
        v_cand = v + dv

        v_diff = v_cand - params.v_thresh_mv
        fired_bool = (v_diff >= 0.0) & (ref_steps <= 0)
        if surrogate_grad:
            spk = _surrogate_spike(v_diff, beta=beta) * active_mask
        else:
            spk = fired_bool.to(dtype=dtype)
        # Detach reset path from surrogate derivative to prevent (-dV * sigma')^T gradient blowup
        v = torch.where(fired_bool, v_reset_t, v_cand)

        ref_steps = torch.where(fired_bool, rfc_full, torch.clamp_min(ref_steps - 1, 0))
        if w_rec is not None and delay_steps > 0:
            delay_buf[(step + delay_steps) % len(delay_buf)] = spk
        spikes_sub.append(spk)
        if step % sub_per_ms == 0:
            v_hist.append(v)

    spikes_tensor = torch.stack(spikes_sub, dim=0)  # [T_sub, N]
    spikes_1ms = spikes_tensor.reshape(duration_ms, sub_per_ms, n_neurons).sum(dim=1)
    v_1ms = torch.stack(v_hist, dim=0)
    return spikes_1ms, v_1ms


@dataclass(frozen=True)
class LIFDriveResult:
    """Verified discrete LIF simulation output and FlyMimic excitation."""

    interface_kind: str
    mn_body_ids: tuple[int, ...]
    pools: tuple[MotorPool, ...]
    mn_spikes_1ms: np.ndarray          # [T_ms, N_mn] discrete binary/count spikes from hard threshold
    excitation: np.ndarray             # [T_ms // 2, P] FlyMimic excitations
    mn_current_u_mv: np.ndarray | None
    premotor_current_u_mv: np.ndarray | None
    premotor_spikes_1ms: np.ndarray | None
    mean_mn_rate_hz: float
    total_mn_spikes: int
    total_premotor_spikes: int
    calibration_loss_history: tuple[float, ...]
    tau_bridge_ms: float
    rate_scale_hz: float
    params: LIFBiophysicsParams
    synaptic_gain_scale: float | None = None
    vjp_rmse_history_deg: tuple[float, ...] = ()

    def metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "interface_kind": self.interface_kind,
            "num_motor_neurons": len(self.mn_body_ids),
            "num_pools": len(self.pools),
            "mean_mn_rate_hz": self.mean_mn_rate_hz,
            "total_mn_spikes": self.total_mn_spikes,
            "total_premotor_spikes": self.total_premotor_spikes,
            "tau_bridge_ms": self.tau_bridge_ms,
            "rate_scale_hz": self.rate_scale_hz,
            "lif_params": {
                "dt_ms": self.params.dt_ms,
                "v_rest_mv": self.params.v_rest_mv,
                "v_thresh_mv": self.params.v_thresh_mv,
                "tau_mem_ms": self.params.tau_mem_ms,
                "tau_syn_ms": self.params.tau_syn_ms,
                "refractory_ms": self.params.refractory_ms,
                "delay_ms": self.params.delay_ms,
                "w_syn": self.params.w_syn,
            },
            "calibration_loss_history": list(self.calibration_loss_history),
        }
        if self.synaptic_gain_scale is not None:
            meta["synaptic_gain_scale"] = float(self.synaptic_gain_scale)
        if self.vjp_rmse_history_deg:
            meta["vjp_rmse_history_deg"] = list(self.vjp_rmse_history_deg)
        return meta


def _staggered_v_init(
    pools: Sequence[MotorPool],
    mn_body_ids: Sequence[int],
    params: LIFBiophysicsParams,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    id_to_col = {int(b): i for i, b in enumerate(mn_body_ids)}
    v_init = torch.full((len(mn_body_ids),), params.v_reset_mv, dtype=dtype)
    span = params.v_thresh_mv - params.v_reset_mv
    assigned = set()
    for pool in pools:
        m = max(1, len(pool.body_ids))
        for local_i, bid in enumerate(pool.body_ids):
            c = id_to_col[int(bid)]
            if c not in assigned:
                v_init[c] = params.v_reset_mv + (float(local_i) / float(m)) * 0.85 * span
                assigned.add(c)
    return v_init


def realize_direct_lif_mn_drive(
    pool_rates_hz: np.ndarray,
    pools: Sequence[MotorPool],
    mn_body_ids: Sequence[int],
    *,
    params: LIFBiophysicsParams = LIFBiophysicsParams(),
    tau_bridge_ms: float = 20.0,
    rate_scale_hz: float = 100.0,
    calibrate_steps: int = 30,
    max_current_mv: float = 60.0,
    stagger_v_init: bool = True,
    physics_vjp_fn: Callable[[np.ndarray], tuple[float, np.ndarray]] | None = None,
    physics_vjp_steps: int = 12,
    physics_vjp_lr: float = 0.2,
) -> LIFDriveResult:
    """Convert K=105 continuous pool rate commands into LIF MN intracellular currents + hard spikes."""
    rates = np.asarray(pool_rates_hz, dtype=np.float64)
    duration_ms, n_pools = rates.shape
    if n_pools != len(pools):
        raise RateBoundaryError(f"Expected {len(pools)} rate pools, got {n_pools}")
    sub_per_ms = max(1, int(round(1.0 / params.dt_ms)))
    ids = tuple(int(b) for b in mn_body_ids)
    id_to_col = {bid: idx for idx, bid in enumerate(ids)}
    n_mn = len(ids)

    mn_rates = np.zeros((duration_ms, n_mn), dtype=np.float64)
    col_assigned = [False] * n_mn
    for p_idx, pool in enumerate(pools):
        for bid in pool.body_ids:
            c = id_to_col[bid]
            if not col_assigned[c]:
                mn_rates[:, c] = rates[:, p_idx]
                col_assigned[c] = True

    dtype = torch.float32
    u_init_1ms = analytical_lif_current_mv(mn_rates, params=params, max_current_mv=max_current_mv)
    v_init = (
        _staggered_v_init(pools, ids, params, dtype=dtype)
        if stagger_v_init
        else None
    )

    with torch.no_grad():
        target_exc_t = differentiable_bridge_pool_spikes(
            torch.from_numpy(mn_rates / 1000.0).unsqueeze(0),
            pools,
            ids,
            rate_scale_hz=rate_scale_hz,
            tau_ms=tau_bridge_ms,
            control_bin_ms=2,
            clip_boundary_gradient="one_sided",
        ).to(dtype=dtype)

    loss_hist: list[float] = []
    n_knots = max(2, duration_ms // 2)
    knot_init = np.zeros((n_knots, n_pools), dtype=np.float32)
    for p_idx, pool in enumerate(pools):
        c0 = id_to_col[pool.body_ids[0]]
        down = u_init_1ms[: n_knots * 2, c0].reshape(n_knots, 2).mean(axis=1)
        knot_init[:, p_idx] = down.astype(np.float32)

    pool_map_idx = torch.zeros((n_mn,), dtype=torch.long)
    for c in range(n_mn):
        for p_idx, pool in enumerate(pools):
            if ids[c] in pool.body_ids:
                pool_map_idx[c] = p_idx
                break

    def _run_mn_knots(knots_t: torch.Tensor, *, surr: bool) -> tuple[torch.Tensor, torch.Tensor]:
        u_pool_sub = torch.repeat_interleave(knots_t.clamp(0.0, max_current_mv), 2 * sub_per_ms, dim=0)
        u_mn_sub = u_pool_sub[:, pool_map_idx]
        spikes_1ms, _ = simulate_lif_substep_trace(
            u_mn_sub, params=params, v_init=v_init, surrogate_grad=surr, beta=2.0
        )
        exc = differentiable_bridge_pool_spikes(
            spikes_1ms.unsqueeze(0).to(dtype=torch.float64),
            pools,
            ids,
            rate_scale_hz=rate_scale_hz,
            tau_ms=tau_bridge_ms,
            control_bin_ms=2,
            clip_boundary_gradient="one_sided",
        ).to(dtype=dtype)
        return exc, spikes_1ms

    u_knots = torch.tensor(knot_init, dtype=dtype, requires_grad=True)
    if calibrate_steps > 0:
        opt = torch.optim.Adam([u_knots], lr=0.35)
        best_s1 = float("inf")
        best_s1_knots = u_knots.detach().clone()
        for _ in range(calibrate_steps):
            opt.zero_grad()
            exc, _ = _run_mn_knots(u_knots, surr=True)
            mse = ((exc - target_exc_t) ** 2).mean()
            loss = mse + 1e-4 * (u_knots[1:] - u_knots[:-1]).pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([u_knots], 2.0)
            opt.step()
            with torch.no_grad():
                u_knots.clamp_(0.0, max_current_mv)
                lv = float(mse.item())
                if lv < best_s1:
                    best_s1 = lv
                    best_s1_knots = u_knots.detach().clone()
            loss_hist.append(lv)
        with torch.no_grad():
            u_knots.copy_(best_s1_knots)

    vjp_rmse_hist: list[float] = []
    if physics_vjp_fn is not None and physics_vjp_steps > 0:
        opt_vjp = torch.optim.Adam([u_knots], lr=physics_vjp_lr)
        best_rmse = float("inf")
        best_knots = u_knots.detach().clone()
        for _ in range(physics_vjp_steps):
            with torch.no_grad():
                hard_exc_t, _ = _run_mn_knots(u_knots.detach(), surr=False)
                hard_exc_np = hard_exc_t[0].cpu().numpy().astype(np.float64)
            rmse_deg, vjp_np = physics_vjp_fn(hard_exc_np)
            vjp_rmse_hist.append(float(rmse_deg))
            if float(rmse_deg) < best_rmse:
                best_rmse = float(rmse_deg)
                best_knots = u_knots.detach().clone()

            opt_vjp.zero_grad()
            surr_exc, _ = _run_mn_knots(u_knots, surr=True)
            smooth_reg = 2e-5 * (u_knots[1:] - u_knots[:-1]).pow(2).mean()
            surr_exc.backward(torch.from_numpy(np.asarray(vjp_np, dtype=np.float32)[None, :, :]))
            smooth_reg.backward()
            torch.nn.utils.clip_grad_norm_([u_knots], 3.0)
            opt_vjp.step()
            with torch.no_grad():
                u_knots.clamp_(0.0, max_current_mv)

        with torch.no_grad():
            cand_exc_t, _ = _run_mn_knots(u_knots.detach(), surr=False)
            cand_rmse, _ = physics_vjp_fn(cand_exc_t[0].cpu().numpy().astype(np.float64))
            vjp_rmse_hist.append(float(cand_rmse))
            if float(cand_rmse) < best_rmse:
                best_knots = u_knots.detach().clone()
            u_knots.copy_(best_knots)

    with torch.no_grad():
        u_pool_1ms = torch.repeat_interleave(u_knots.detach().clamp(0.0, max_current_mv), 2, dim=0)
        u_mn_1ms = u_pool_1ms[:, pool_map_idx]
        hard_exc, hard_spikes_1ms = _run_mn_knots(u_knots.detach(), surr=False)

    sp_np = hard_spikes_1ms.cpu().numpy().astype(np.float64)
    mean_rate = float(sp_np.sum() / (max(1, n_mn) * (duration_ms / 1000.0)))
    return LIFDriveResult(
        interface_kind="direct_lif_motor_pool_current",
        mn_body_ids=ids,
        pools=tuple(pools),
        mn_spikes_1ms=sp_np,
        excitation=hard_exc[0].cpu().numpy().astype(np.float64),
        mn_current_u_mv=u_mn_1ms.cpu().numpy().astype(np.float64),
        premotor_current_u_mv=None,
        premotor_spikes_1ms=None,
        mean_mn_rate_hz=mean_rate,
        total_mn_spikes=int(round(float(sp_np.sum()))),
        total_premotor_spikes=0,
        calibration_loss_history=tuple(loss_hist),
        tau_bridge_ms=float(tau_bridge_ms),
        rate_scale_hz=float(rate_scale_hz),
        params=params,
        vjp_rmse_history_deg=tuple(vjp_rmse_hist),
    )


def compute_auto_synaptic_gain_scale(
    connectome: PremotorConnectome,
    params: LIFBiophysicsParams = LIFBiophysicsParams(),
) -> float:
    """Compute a single global scalar gain keeping MaleCNS weights in the linear LIF regime."""
    pos_row_sum = np.maximum(connectome.weight_pre_to_mn, 0.0).sum(axis=1)
    pos_active = pos_row_sum[pos_row_sum > 0.0]
    ref_pos = float(np.percentile(pos_active, 75)) if pos_active.size > 0 else 100.0
    denom = max(params.tau_syn_ms * params.w_syn * 0.05 * max(ref_pos, 1.0), 1.0)
    return float(np.clip(25.0 / denom, 0.05, 1.0))


def optimize_premotor_connectome_drive(
    target_excitation: np.ndarray,
    connectome: PremotorConnectome,
    *,
    params: LIFBiophysicsParams = LIFBiophysicsParams(),
    tau_bridge_ms: float = 20.0,
    rate_scale_hz: float = 100.0,
    opt_steps: int = 45,
    lr: float = 0.6,
    max_current_mv: float = 45.0,
    allow_hyperpolarizing_u: bool = False,
    synaptic_gain_scale: float | None = None,
    physics_vjp_fn: Callable[[np.ndarray], tuple[float, np.ndarray]] | None = None,
    physics_vjp_steps: int = 12,
    physics_vjp_lr: float = 0.6,
) -> LIFDriveResult:
    """Optimize upstream MaleCNS premotor stimulation current u_pre(t) with u_MN=0.

    Motor neurons receive zero external current; all MN firing originates strictly from
    delayed weighted spikes passing through real MaleCNS synapses (graph_min5.npz) into
    coupled LIF MN dynamics, bridged via CausalMotorExcitation to FlyMimic controls.
    """
    exc_target_np = np.asarray(target_excitation, dtype=np.float32)
    n_bins, n_pools = exc_target_np.shape
    duration_ms = n_bins * 2
    sub_per_ms = max(1, int(round(1.0 / params.dt_ms)))
    delay_steps = max(0, int(round(params.delay_ms / params.dt_ms)))

    pools = connectome.pools
    if len(pools) != n_pools:
        raise RateBoundaryError(f"target_excitation has {n_pools} pools, connectome has {len(pools)}")

    n_mn, n_pre = connectome.weight_pre_to_mn.shape
    if n_pre == 0:
        raise RateBoundaryError("PremotorConnectome contains zero presynaptic neurons")

    gain_scale = (
        compute_auto_synaptic_gain_scale(connectome, params)
        if synaptic_gain_scale is None
        else float(synaptic_gain_scale)
    )

    device = torch.device("cuda" if (torch.cuda.is_available() and n_pre >= 700) else "cpu")
    dtype = torch.float32
    w_pre_mn = torch.tensor(
        connectome.weight_pre_to_mn * gain_scale, dtype=dtype, device=device
    )
    w_mn_mn = torch.tensor(
        connectome.weight_mn_to_mn * gain_scale, dtype=dtype, device=device
    )
    has_w_rec = float(w_mn_mn.abs().sum().item()) > 0.0

    # Initialize premotor drive near threshold so gradients stay alive without tonic saturation
    min_u = -float(max_current_mv) if allow_hyperpolarizing_u else 0.0
    dv_th = params.v_thresh_mv - params.v_rest_mv
    net_exc = connectome.weight_pre_to_mn.mean(axis=0) > 0.0
    u_init = np.full((105, n_pre), 0.78 * dv_th, dtype=np.float32)
    u_init[:, net_exc] = 0.95 * dv_th
    u_pre_knots = torch.tensor(u_init, dtype=dtype, device=device, requires_grad=True)

    target_t = torch.from_numpy(exc_target_np).to(device=device).unsqueeze(0)
    v_init_mn = _staggered_v_init(pools, connectome.mn_body_ids, params, dtype=dtype).to(device=device)
    v_init_pre = (params.v_reset_mv + (torch.arange(n_pre, dtype=dtype, device=device) % 7) * (0.12 * dv_th))

    t_sub = duration_ms * sub_per_ms
    rfc_count = max(1, int(round(params.refractory_ms / params.dt_ms)))
    alpha_mem = params.dt_ms / params.tau_mem_ms
    v_reset_pre_t = torch.full((n_pre,), params.v_reset_mv, dtype=dtype, device=device)
    rfc_pre_full = torch.full((n_pre,), rfc_count, dtype=torch.int64, device=device)
    u_mn_zero = torch.zeros((t_sub, n_mn), dtype=dtype, device=device)

    def _forward_two_layer(surr: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u_pre_sub = F.interpolate(
            u_pre_knots.clamp(min_u, max_current_mv).T.unsqueeze(0),
            size=t_sub,
            mode="linear",
            align_corners=True,
        )[0].T
        v_drive_pre = params.v_rest_mv + u_pre_sub

        v_p = v_init_pre.clone()
        rfc_pre = torch.zeros((n_pre,), dtype=torch.int64, device=device)
        pre_spk_sub: list[torch.Tensor] = []
        for s in range(t_sub):
            active = (rfc_pre <= 0).to(dtype=dtype)
            dv = alpha_mem * (v_drive_pre[s] - v_p) * active
            vc = v_p + dv
            vd = vc - params.v_thresh_mv
            fired = (vd >= 0.0) & (rfc_pre <= 0)
            if surr:
                spk = _surrogate_spike(vd, beta=2.5) * active
            else:
                spk = fired.to(dtype=dtype)
            v_p = torch.where(fired, v_reset_pre_t, vc)
            rfc_pre = torch.where(fired, rfc_pre_full, torch.clamp_min(rfc_pre - 1, 0))
            pre_spk_sub.append(spk)
        pre_spk_mat = torch.stack(pre_spk_sub, dim=0)

        if delay_steps > 0:
            delayed_pre = torch.cat(
                [torch.zeros((delay_steps, n_pre), dtype=dtype, device=device), pre_spk_mat[:-delay_steps]],
                dim=0,
            )
        else:
            delayed_pre = pre_spk_mat
        g_ext_mn = delayed_pre @ w_pre_mn.T

        mn_spikes_1ms, _ = simulate_lif_substep_trace(
            u_mn_zero,
            params=params,
            w_rec=w_mn_mn if has_w_rec else None,
            g_ext_sub=g_ext_mn,
            v_init=v_init_mn,
            surrogate_grad=surr,
            beta=2.5,
        )
        exc_out = differentiable_bridge_pool_spikes(
            mn_spikes_1ms.unsqueeze(0).to(device="cpu", dtype=torch.float64),
            pools,
            connectome.mn_body_ids,
            rate_scale_hz=rate_scale_hz,
            tau_ms=tau_bridge_ms,
            control_bin_ms=2,
            clip_boundary_gradient="one_sided",
        ).to(device=device, dtype=dtype)
        pre_1ms = pre_spk_mat.reshape(duration_ms, sub_per_ms, n_pre).sum(dim=1)
        return exc_out, mn_spikes_1ms, pre_1ms

    loss_hist: list[float] = []
    best_knots = u_pre_knots.detach().clone()
    if opt_steps > 0:
        opt = torch.optim.Adam([u_pre_knots], lr=lr)
        best_loss = float("inf")
        for _ in range(opt_steps):
            opt.zero_grad()
            exc_pred, _, _ = _forward_two_layer(surr=True)
            mse = ((exc_pred - target_t) ** 2).mean()
            smooth = 2e-5 * (u_pre_knots[1:] - u_pre_knots[:-1]).pow(2).mean()
            u_reg = 1e-5 * (u_pre_knots ** 2).mean()
            loss = mse + smooth + u_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_([u_pre_knots], 2.0)
            opt.step()
            with torch.no_grad():
                u_pre_knots.clamp_(min_u, max_current_mv)
                lval = float(mse.item())
                loss_hist.append(lval)
                if lval < best_loss:
                    best_loss = lval
                    best_knots = u_pre_knots.detach().clone()
        with torch.no_grad():
            u_pre_knots.copy_(best_knots)

    vjp_rmse_hist: list[float] = []
    if physics_vjp_fn is not None and physics_vjp_steps > 0:
        opt_vjp = torch.optim.Adam([u_pre_knots], lr=physics_vjp_lr)
        best_rmse = float("inf")
        for _ in range(physics_vjp_steps):
            with torch.no_grad():
                hard_exc_t, _, _ = _forward_two_layer(surr=False)
                hard_exc_np = hard_exc_t[0].detach().cpu().numpy().astype(np.float64)
            rmse_deg, vjp_np = physics_vjp_fn(hard_exc_np)
            vjp_rmse_hist.append(float(rmse_deg))
            if float(rmse_deg) < best_rmse:
                best_rmse = float(rmse_deg)
                best_knots = u_pre_knots.detach().clone()

            opt_vjp.zero_grad()
            surr_exc, _, _ = _forward_two_layer(surr=True)
            reg_loss = 2e-5 * (u_pre_knots[1:] - u_pre_knots[:-1]).pow(2).mean() + 1e-5 * (u_pre_knots ** 2).mean()
            surr_exc.backward(torch.from_numpy(np.asarray(vjp_np, dtype=np.float32)[None, :, :]).to(device=device))
            reg_loss.backward()
            torch.nn.utils.clip_grad_norm_([u_pre_knots], 2.5)
            opt_vjp.step()
            with torch.no_grad():
                u_pre_knots.clamp_(min_u, max_current_mv)

        with torch.no_grad():
            cand_exc_t, _, _ = _forward_two_layer(surr=False)
            cand_rmse, _ = physics_vjp_fn(cand_exc_t[0].detach().cpu().numpy().astype(np.float64))
            vjp_rmse_hist.append(float(cand_rmse))
            if float(cand_rmse) < best_rmse:
                best_knots = u_pre_knots.detach().clone()

    with torch.no_grad():
        u_pre_knots.copy_(best_knots)
        hard_exc, hard_mn_spikes, hard_pre_spikes = _forward_two_layer(surr=False)
        u_pre_1ms = F.interpolate(
            best_knots.T.unsqueeze(0),
            size=duration_ms,
            mode="linear",
            align_corners=True,
        )[0].T

    sp_mn_np = hard_mn_spikes.detach().cpu().numpy().astype(np.float64)
    sp_pre_np = hard_pre_spikes.detach().cpu().numpy().astype(np.float64)
    mean_rate = float(sp_mn_np.sum() / (max(1, n_mn) * (duration_ms / 1000.0)))
    return LIFDriveResult(
        interface_kind="malecns_premotor_connectome_lif_drive",
        mn_body_ids=connectome.mn_body_ids,
        pools=tuple(pools),
        mn_spikes_1ms=sp_mn_np,
        excitation=hard_exc[0].detach().cpu().numpy().astype(np.float64),
        mn_current_u_mv=None,
        premotor_current_u_mv=u_pre_1ms.detach().cpu().numpy().astype(np.float64),
        premotor_spikes_1ms=sp_pre_np,
        mean_mn_rate_hz=mean_rate,
        total_mn_spikes=int(round(float(sp_mn_np.sum()))),
        total_premotor_spikes=int(round(float(sp_pre_np.sum()))),
        calibration_loss_history=tuple(loss_hist),
        tau_bridge_ms=float(tau_bridge_ms),
        rate_scale_hz=float(rate_scale_hz),
        params=params,
        synaptic_gain_scale=gain_scale,
        vjp_rmse_history_deg=tuple(vjp_rmse_hist),
    )
