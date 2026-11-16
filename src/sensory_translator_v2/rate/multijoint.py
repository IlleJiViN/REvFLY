"""Multi-joint (Coxa/Femur/Tibia, 15-muscle zero teacher-assist) & multi-motion-family trainer."""

from __future__ import annotations

import copy
import math
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from ..assets import AssetCatalog, load_asset_catalog
from ..backends.persistent_mujoco import PersistentMujocoBackend
from ..contracts import (
    Backend,
    CONTROL_STEPS,
    Evaluation,
    LF_ACTUATOR_COLUMNS,
    LF_JOINT_GROUPS,
    LF_JOINT_NAMES,
    LF_QPOS_ADDRESSES,
    MultiJointEvaluationRequest,
    NU,
    STATE_STEPS,
    Session,
    SessionSpec,
    InitialState,
)
from ..evidence import utc_now_iso, write_json_atomic
from ..goals import encode_multijoint_condition
from .bridge import direct_multichannel_rate_waveform, multichannel_rate_to_excitation
from .model import build_rate_policy_from_config, save_rate_checkpoint
from .optim import lbfgs_direction


@dataclass(frozen=True)
class MultiJointTrainingConfig:
    """Configuration for 15-muscle (0 teacher-assist), 7-DOF LF multi-family policy training."""

    output_dir: Path
    profile: str = "adjacent_static_parent_v1"
    architecture: str = "GoalStateConditionedContinuousRateFFN"
    knots_per_pool: int = 105
    hidden_dim: int = 256
    num_layers: int = 2
    max_rate_hz: float = 200.0
    tau_ms: float = 2.0
    rate_scale_hz: float = 100.0
    n_workers: int = 8
    anchor_lbfgs_steps: int = 8
    e2e_epochs: int = 14
    e2e_lr: float = 3e-4
    energy_weight: float = 1.5e-4
    sat_weight: float = 2e-6
    smooth_weight: float = 2e-7
    distill_steps: int = 700
    family_weights: dict[str, float] | None = None
    seed: int = 42
    timeout_s: float = 120.0
    asset_root: Path | None = None
    runtime_root: Path | None = None
    physics_python: Path | None = None


@dataclass(frozen=True)
class MultiJointTask:
    """Single 7-joint LF trajectory task across a named motion family."""

    name: str
    family: str
    split: str
    delta_rad: np.ndarray       # [7] initial joint offset from teacher qpos[0]
    target_rad: np.ndarray      # [211, 7]
    condition: torch.Tensor     # [127]
    warm_exc: np.ndarray | None # [210, 15] optional physical forward primitive or None


class MultiJointWorkerPool:
    """Thread-safe pool of persistent MuJoCo (or fake) backends for 15-muscle 7-joint VJP."""

    def __init__(
        self,
        catalog: AssetCatalog,
        profile: str,
        n_workers: int,
        *,
        timeout_s: float = 120.0,
        backend_factory: Callable[[AssetCatalog, str], Backend] | None = None,
    ) -> None:
        self._slots: queue.Queue[tuple[Backend, Session]] = queue.Queue()
        self._all: list[tuple[Backend, Session]] = []
        hashes = catalog.three_hash_record(profile)
        spec = SessionSpec(
            profile=profile,
            asset_manifest_sha256=catalog.manifest_sha256,
            teacher_sha256=catalog.teacher_sha256,
            raw_xml_sha256=catalog.raw_xml_sha256,
            profile_rule_sha256=hashes["profile_rule_sha256"],
            cache_mode="fake" if backend_factory is not None else "cached_model",
        )
        try:
            for _ in range(max(1, n_workers)):
                backend = (
                    backend_factory(catalog, profile)
                    if backend_factory is not None
                    else PersistentMujocoBackend(
                        catalog,
                        profile=profile,
                        cache_mode="cached_model",
                        timeout_s=timeout_s,
                    )
                )
                session = backend.open_session(spec)
                pair = (backend, session)
                self._all.append(pair)
                self._slots.put(pair)
        except Exception:
            self.close()
            raise

    @property
    def size(self) -> int:
        return len(self._all)

    def eval_batch(
        self,
        tasks: list[MultiJointTask],
        controls_list: list[np.ndarray],
        *,
        need_jacobian: bool,
        clip_boundary_gradient: str = "one_sided",
        seed_base: int = 1000,
    ) -> list[Evaluation]:
        def _run_one(item: tuple[int, MultiJointTask, np.ndarray]) -> Evaluation:
            idx, task, ctrl = item
            backend, session = self._slots.get()
            try:
                backend.reset(session, InitialState(0.0), seed=seed_base + idx)
                req = MultiJointEvaluationRequest(
                    controls=ctrl,
                    control_columns=LF_ACTUATOR_COLUMNS,
                    target_qpos_indices=LF_QPOS_ADDRESSES,
                    initial_qpos_delta_rad=task.delta_rad,
                    need_jacobian=need_jacobian,
                    mse_target_rad=task.target_rad if need_jacobian else None,
                    clip_boundary_gradient=clip_boundary_gradient,
                    request_id=f"mj_train_{task.name}_{idx}",
                )
                return backend.evaluate_multijoint(session, req)  # type: ignore[attr-defined]
            finally:
                self._slots.put((backend, session))

        if len(tasks) <= 1 or len(self._all) <= 1:
            return [_run_one((i, t, c)) for i, (t, c) in enumerate(zip(tasks, controls_list))]
        with ThreadPoolExecutor(max_workers=min(len(tasks), len(self._all))) as ex:
            return list(ex.map(_run_one, [(i, t, c) for i, (t, c) in enumerate(zip(tasks, controls_list))]))

    def close(self) -> None:
        for backend, session in self._all:
            try:
                backend.close_session(session)
            except Exception:
                pass
            try:
                backend.close()
            except Exception:
                pass
        self._all.clear()


def _invert_exc_to_knots_batch(
    exc_list: list[np.ndarray],
    cfg: MultiJointTrainingConfig,
    *,
    steps: int = 240,
) -> list[np.ndarray]:
    if not exc_list:
        return []
    exc_batch = np.stack([np.asarray(e, dtype=np.float64) for e in exc_list], axis=0)  # [B, 210, 15]
    target_t = torch.from_numpy(exc_batch)
    inits = []
    for exc_arr in exc_list:
        init = np.clip(exc_arr[: CONTROL_STEPS : 2].T * cfg.rate_scale_hz, 0.5, cfg.max_rate_hz - 5.0)
        if init.shape[1] != cfg.knots_per_pool:
            init = np.full((NU, cfg.knots_per_pool), 12.0, dtype=np.float64)
        inits.append(init)
    knots = torch.tensor(np.stack(inits, axis=0), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([knots], lr=2.5)
    for _ in range(steps):
        opt.zero_grad()
        clamped = knots.clamp(0.0, cfg.max_rate_hz)
        wf = direct_multichannel_rate_waveform(clamped, max_rate_hz=cfg.max_rate_hz)
        exc = multichannel_rate_to_excitation(
            wf,
            max_rate_hz=cfg.max_rate_hz,
            rate_scale_hz=cfg.rate_scale_hz,
            tau_ms=cfg.tau_ms,
            clip_boundary_gradient="one_sided",
        )
        loss = (
            ((exc - target_t) ** 2).mean()
            + 2e-5 * (knots[:, :, 1:] - knots[:, :, :-1]).pow(2).mean()
            + 5e-7 * torch.relu(wf - 94.0).pow(2).mean()
        )
        loss.backward()
        opt.step()
        with torch.no_grad():
            knots.clamp_(0.2, min(180.0, cfg.max_rate_hz - 5.0))
    arr = knots.detach().cpu().numpy()
    return [arr[i].copy() for i in range(len(exc_list))]


def _invert_exc_to_knots(
    exc_arr: np.ndarray,
    cfg: MultiJointTrainingConfig,
    *,
    steps: int = 300,
) -> np.ndarray:
    return _invert_exc_to_knots_batch([exc_arr], cfg, steps=steps)[0]


def _knots_to_wf_exc(
    knots: torch.Tensor,
    cfg: MultiJointTrainingConfig,
    *,
    clip_mode: str = "one_sided",
) -> tuple[torch.Tensor, torch.Tensor]:
    clamped = knots.clamp(0.0, cfg.max_rate_hz)
    wf = direct_multichannel_rate_waveform(clamped, max_rate_hz=cfg.max_rate_hz)
    exc = multichannel_rate_to_excitation(
        wf,
        max_rate_hz=cfg.max_rate_hz,
        rate_scale_hz=cfg.rate_scale_hz,
        tau_ms=cfg.tau_ms,
        clip_boundary_gradient=clip_mode,
    )
    return wf, exc


def _reg_terms(
    knots: torch.Tensor,
    wf: torch.Tensor,
    exc: torch.Tensor,
    cfg: MultiJointTrainingConfig,
) -> torch.Tensor:
    energy = cfg.energy_weight * (exc ** 2).mean(dim=(1, 2))
    sat = cfg.sat_weight * torch.relu(wf - 92.0).pow(2).mean(dim=(1, 2))
    smooth = cfg.smooth_weight * (knots[:, :, 1:] - knots[:, :, :-1]).pow(2).mean(dim=(1, 2))
    return energy + sat + smooth


def _make_task_from_target(
    name: str,
    family: str,
    split: str,
    target_rad: np.ndarray,
    teacher_q0: np.ndarray,
    teacher_v0: np.ndarray,
    delta_rad: np.ndarray,
    warm_exc: np.ndarray | None = None,
) -> MultiJointTask:
    q0 = teacher_q0 + delta_rad
    cond = encode_multijoint_condition(
        torch.from_numpy(target_rad).unsqueeze(0),
        torch.from_numpy(q0).unsqueeze(0),
        torch.from_numpy(teacher_v0).unsqueeze(0),
    )[0]
    return MultiJointTask(
        name=name,
        family=family,
        split=split,
        delta_rad=delta_rad.astype(np.float64),
        target_rad=target_rad.astype(np.float64),
        condition=cond,
        warm_exc=warm_exc,
    )


def build_multifamily_tasks(
    pool: MultiJointWorkerPool,
    teacher_q7: np.ndarray,
    teacher_v0: np.ndarray,
    teacher_u15: np.ndarray,
) -> list[MultiJointTask]:
    """Build 4 physically reachable multi-joint motion families (train + held-out eval splits).

    Families:
      1. `swing_scaled`: whole-leg swing scaled & shifted across initial poses (Coxa/Femur/Tibia)
      2. `decoupled_group_swing`: independent Coxa / Femur / Tibia amplitude combinations
      3. `rhythmic_double_tap`: 2x-frequency stepping/tapping motor primitives verified on MuJoCo
      4. `reach_and_hold`: sigmoid/cosine reach-and-hold posture transitions across Coxa/Femur/Tibia
    """
    q0 = teacher_q7[0]
    t = np.linspace(0.0, 1.0, CONTROL_STEPS, dtype=np.float64)[:, None]

    # Define control primitives for physical rollouts so Families 2/3/4 are 100% dynamically realizable
    u_mean = teacher_u15.mean(axis=0, keepdims=True)
    du = teacher_u15 - u_mean
    coxa_cols = list(range(0, 7))
    femur_cols = list(range(7, 13))
    tibia_cols = [13, 14]

    primitive_specs: list[tuple[str, str, str, np.ndarray, np.ndarray]] = []
    # 1) Swing scaled + perturbed initial poses across ALL 3 leg groups
    zero_d = np.zeros((7,), dtype=np.float64)
    for split, items in (
        ("train", [(0.75, zero_d), (0.85, zero_d), (0.95, zero_d), (1.00, zero_d), (1.08, zero_d), (1.15, zero_d),
                   (0.95, np.radians([2.0, -4.0, 1.5, -2.5, 4.0, 2.0, -5.0])),
                   (0.95, np.radians([-2.0, 4.0, -1.5, 2.5, -4.0, -2.0, 5.0]))]),
        ("val", [(0.80, zero_d), (0.90, zero_d), (1.03, zero_d),
                 (0.95, np.radians([1.5, -3.0, 1.0, -2.0, 3.0, 1.5, -4.0]))]),
    ):
        for idx, (s, d7) in enumerate(items):
            u_cand = np.clip(u_mean + s * du, 0.01, 0.99)
            primitive_specs.append((f"swing_{split}_{idx}_s{s:.2f}", "swing_scaled", split, d7, u_cand))

    # 2) Decoupled group amplitudes (Coxa scale, Femur scale, Tibia scale independently varied)
    for split, triplets in (
        ("train", [(1.25, 0.70, 0.85), (0.65, 1.25, 0.95), (0.85, 0.80, 1.20), (1.15, 1.10, 0.65)]),
        ("val", [(1.10, 0.75, 1.05), (0.75, 1.15, 0.75)]),
    ):
        for idx, (sc, sf, st) in enumerate(triplets):
            u_cand = teacher_u15.copy()
            u_cand[:, coxa_cols] = np.clip(u_mean[:, coxa_cols] + sc * du[:, coxa_cols], 0.01, 0.99)
            u_cand[:, femur_cols] = np.clip(u_mean[:, femur_cols] + sf * du[:, femur_cols], 0.01, 0.99)
            u_cand[:, tibia_cols] = np.clip(u_mean[:, tibia_cols] + st * du[:, tibia_cols], 0.01, 0.99)
            primitive_specs.append((f"decoupled_{split}_{idx}", "decoupled_group_swing", split, zero_d, u_cand))

    # 3) Rhythmic 2x frequency double-tap modulation
    for split, amps in (
        ("train", [(0.14, 0.16, 0.18), (-0.12, 0.14, 0.14), (0.10, -0.14, 0.16), (0.16, 0.10, -0.14)]),
        ("val", [(0.12, 0.12, 0.15), (-0.10, 0.12, 0.12)]),
    ):
        for idx, (ac, af, at) in enumerate(amps):
            u_cand = u_mean.repeat(CONTROL_STEPS, axis=0) * 0.8 + 0.1 * teacher_u15
            osc1 = np.sin(2.0 * np.pi * t)
            osc2 = np.sin(4.0 * np.pi * t)
            u_cand[:, [2, 3]] += ac * osc1
            u_cand[:, [5]] -= ac * osc1
            u_cand[:, [7, 12]] += af * osc2
            u_cand[:, [11]] -= af * 0.7 * osc2
            u_cand[:, [13]] += at * osc2
            u_cand[:, [14]] -= at * 0.8 * osc2
            u_cand = np.clip(u_cand, 0.01, 0.99)
            primitive_specs.append((f"tap2x_{split}_{idx}", "rhythmic_double_tap", split, zero_d, u_cand))

    # 4) Smooth reach-and-hold posture transitions across Coxa/Femur/Tibia
    for split, holds in (
        ("train", [(0.20, 0.18, 0.22), (-0.18, 0.14, -0.14), (0.14, -0.18, 0.18), (-0.14, -0.14, 0.20)]),
        ("val", [(0.17, 0.15, 0.19), (-0.16, 0.12, 0.14)]),
    ):
        for idx, (hc, hf, ht) in enumerate(holds):
            ramp = 0.5 * (1.0 - np.cos(np.pi * np.clip(t / 0.55, 0.0, 1.0)))
            u_cand = u_mean.repeat(CONTROL_STEPS, axis=0)
            u_cand[:, [0, 1, 2]] += hc * ramp
            u_cand[:, [7, 8, 12]] += hf * ramp
            u_cand[:, [13]] += ht * ramp
            u_cand[:, [14]] -= 0.7 * ht * ramp
            u_cand = np.clip(u_cand, 0.01, 0.99)
            primitive_specs.append((f"reach_{split}_{idx}", "reach_and_hold", split, zero_d, u_cand))

    # Generate true dynamically reachable target qpos[:, :7] by rolling every primitive on zero_d
    # Wait: for perturbed delta_rad tasks, keep the nominal zero_d target so the policy learns pose-error recovery!
    zero_tasks_for_rollout = [
        MultiJointTask(
            name=name,
            family=fam,
            split=split,
            delta_rad=zero_d,
            target_rad=teacher_q7,
            condition=torch.zeros((127,), dtype=torch.float64),
            warm_exc=u_cand,
        )
        for name, fam, split, _, u_cand in primitive_specs
    ]
    rollouts = pool.eval_batch(
        zero_tasks_for_rollout,
        [u for _, _, _, _, u in primitive_specs],
        need_jacobian=False,
        seed_base=100,
    )
    tasks: list[MultiJointTask] = []
    for (name, fam, split, d7, u_cand), ev in zip(primitive_specs, rollouts):
        q_targ = ev.traces["qpos"][:, :7].copy()
        tasks.append(
            _make_task_from_target(
                name=name,
                family=fam,
                split=split,
                target_rad=q_targ,
                teacher_q0=q0,
                teacher_v0=teacher_v0,
                delta_rad=d7,
                warm_exc=u_cand,
            )
        )
    return tasks


def _solve_lbfgs_batch(
    pool: MultiJointWorkerPool,
    tasks: list[MultiJointTask],
    init_knots: list[np.ndarray],
    cfg: MultiJointTrainingConfig,
    *,
    clip_mode: str,
) -> list[np.ndarray]:
    n = len(tasks)
    xs = [k.copy() for k in init_knots]
    s_hists: list[list[np.ndarray]] = [[] for _ in range(n)]
    y_hists: list[list[np.ndarray]] = [[] for _ in range(n)]
    prev_xs: list[np.ndarray | None] = [None] * n
    prev_gs: list[np.ndarray | None] = [None] * n
    alphas = (2.0, 1.0, 0.5, 0.25, 0.1)

    for step in range(cfg.anchor_lbfgs_steps):
        excs: list[np.ndarray] = []
        with torch.no_grad():
            for x in xs:
                _, e = _knots_to_wf_exc(torch.from_numpy(x[None, :, :]), cfg, clip_mode=clip_mode)
                excs.append(e[0].cpu().numpy())
        evals = pool.eval_batch(
            tasks,
            excs,
            need_jacobian=True,
            clip_boundary_gradient=clip_mode,
            seed_base=5000 + step * 100,
        )
        curr_losses: list[float] = []
        dirs: list[np.ndarray] = []
        grads_np: list[np.ndarray] = []
        for idx, (task, x_np, ev) in enumerate(zip(tasks, xs, evals)):
            q_sub = ev.traces["qpos"][:, list(LF_QPOS_ADDRESSES)]
            mse = float(np.mean((q_sub - task.target_rad) ** 2))
            kt = torch.tensor(x_np[None, :, :], dtype=torch.float64, requires_grad=True)
            wf, exc = _knots_to_wf_exc(kt, cfg, clip_mode=clip_mode)
            reg = _reg_terms(kt, wf, exc, cfg)[0]
            curr_losses.append(mse + float(reg.item()))
            assert ev.control_vjp is not None
            surr = torch.sum(exc[0] * torch.from_numpy(ev.control_vjp)) + reg
            surr.backward()
            assert kt.grad is not None
            g_np = kt.grad[0].detach().cpu().numpy().copy()
            grads_np.append(g_np)
            if prev_xs[idx] is not None and prev_gs[idx] is not None:
                sk = x_np.reshape(-1) - prev_xs[idx]  # type: ignore[operator]
                yk = g_np.reshape(-1) - prev_gs[idx]  # type: ignore[operator]
                if float(np.dot(yk, sk)) > 1e-20:
                    s_hists[idx].append(sk)
                    y_hists[idx].append(yk)
                    if len(s_hists[idx]) > 5:
                        s_hists[idx].pop(0)
                        y_hists[idx].pop(0)
            dirs.append(
                lbfgs_direction(
                    g_np, s_hists[idx], y_hists[idx], init_step_rms_hz=0.05, max_step_rms_hz=12.0
                )
            )

        best_cands = list(xs)
        best_losses = list(curr_losses)
        improved = [False] * n
        for j, alpha in enumerate(alphas):
            trial_xs = [np.clip(xs[i] + alpha * dirs[i], 0.0, cfg.max_rate_hz) for i in range(n)]
            trial_excs: list[np.ndarray] = []
            with torch.no_grad():
                for tx in trial_xs:
                    _, te = _knots_to_wf_exc(torch.from_numpy(tx[None, :, :]), cfg, clip_mode=clip_mode)
                    trial_excs.append(te[0].cpu().numpy())
            trial_evals = pool.eval_batch(
                tasks,
                trial_excs,
                need_jacobian=False,
                seed_base=6000 + step * 100 + j * 10,
            )
            with torch.no_grad():
                for i in range(n):
                    q_t = trial_evals[i].traces["qpos"][:, list(LF_QPOS_ADDRESSES)]
                    mse_t = float(np.mean((q_t - tasks[i].target_rad) ** 2))
                    kt = torch.from_numpy(trial_xs[i][None, :, :])
                    wf_t, exc_t = _knots_to_wf_exc(kt, cfg, clip_mode=clip_mode)
                    loss_t = mse_t + float(_reg_terms(kt, wf_t, exc_t, cfg)[0].item())
                    if loss_t < best_losses[i]:
                        best_losses[i] = loss_t
                        best_cands[i] = trial_xs[i]
                        improved[i] = True
        for i in range(n):
            if improved[i]:
                prev_xs[i] = xs[i].reshape(-1).copy()
                prev_gs[i] = grads_np[i].reshape(-1).copy()
                xs[i] = best_cands[i]
    return xs


def evaluate_multijoint_tasks(
    pool: MultiJointWorkerPool,
    model: nn.Module,
    tasks: list[MultiJointTask],
    cfg: MultiJointTrainingConfig,
    *,
    clip_mode: str,
) -> dict[str, Any]:
    conds = torch.stack([t.condition for t in tasks], dim=0)
    with torch.no_grad():
        knots = model(conds)
        wf, exc = _knots_to_wf_exc(knots, cfg, clip_mode=clip_mode)
        excs = [exc[i].cpu().numpy() for i in range(len(tasks))]
    evals = pool.eval_batch(tasks, excs, need_jacobian=False, seed_base=9000)
    per_task: dict[str, dict[str, Any]] = {}
    all_joint_rmses = np.zeros((len(tasks), 7), dtype=np.float64)
    for i, (ev, t) in enumerate(zip(evals, tasks)):
        q_sub = ev.traces["qpos"][:, list(LF_QPOS_ADDRESSES)]
        per_j_deg = np.degrees(np.sqrt(np.mean((q_sub - t.target_rad) ** 2, axis=0)))
        all_joint_rmses[i] = per_j_deg
        overall_deg = float(np.degrees(np.sqrt(np.mean((q_sub - t.target_rad) ** 2))))
        per_task[t.name] = {
            "family": t.family,
            "overall_rmse_deg": overall_deg,
            "coxa_rmse_deg": float(np.sqrt(np.mean(per_j_deg[list(LF_JOINT_GROUPS["coxa"])] ** 2))),
            "femur_rmse_deg": float(np.sqrt(np.mean(per_j_deg[list(LF_JOINT_GROUPS["femur_trochanter"])] ** 2))),
            "tibia_rmse_deg": float(per_j_deg[6]),
            "per_joint_rmse_deg": {
                jname: float(val) for jname, val in zip(LF_JOINT_NAMES, per_j_deg)
            },
            "controlled_muscles": 15,
            "teacher_assisted_muscles": 0,
        }
    group_summary = {
        "coxa_mean_rmse_deg": float(np.mean([v["coxa_rmse_deg"] for v in per_task.values()])),
        "femur_mean_rmse_deg": float(np.mean([v["femur_rmse_deg"] for v in per_task.values()])),
        "tibia_mean_rmse_deg": float(np.mean([v["tibia_rmse_deg"] for v in per_task.values()])),
    }
    by_family: dict[str, float] = {}
    for fam in sorted({t.family for t in tasks}):
        vals = [v["overall_rmse_deg"] for v in per_task.values() if v["family"] == fam]
        by_family[fam] = float(np.mean(vals))
    return {
        "mean_overall_rmse_deg": float(np.mean([v["overall_rmse_deg"] for v in per_task.values()])),
        "max_overall_rmse_deg": float(np.max([v["overall_rmse_deg"] for v in per_task.values()])),
        "mean_mse_rad2": float(
            np.mean([(math.radians(v["overall_rmse_deg"])) ** 2 for v in per_task.values()])
        ),
        "max_mse_rad2": float(
            np.max([(math.radians(v["overall_rmse_deg"])) ** 2 for v in per_task.values()])
        ),
        "group_summary": group_summary,
        "family_mean_rmse_deg": by_family,
        "per_joint_mean_rmse_deg": {
            jname: float(val) for jname, val in zip(LF_JOINT_NAMES, all_joint_rmses.mean(axis=0))
        },
        "per_task": per_task,
    }


def train_multijoint_rate_model(
    cfg: MultiJointTrainingConfig,
    *,
    backend_factory: Callable[[AssetCatalog, str], Backend] | None = None,
    model_factory: Callable[[MultiJointTrainingConfig, np.ndarray, torch.Tensor, torch.Tensor], nn.Module] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train 15-muscle (0 teacher-assist), 7-DOF LF multi-family RatePolicyProtocol model."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    out_dir = Path(cfg.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    catalog = load_asset_catalog(
        asset_root=cfg.asset_root,
        runtime_root=cfg.runtime_root,
        physics_python=cfg.physics_python,
        verify_hashes=True,
    )
    with np.load(catalog.teacher_path, allow_pickle=False) as tf:
        teacher_q7 = np.asarray(tf["qpos"][:, :7], dtype=np.float64)
        teacher_v0 = np.asarray(tf["qvel"][0, :7], dtype=np.float64)
        teacher_u15 = np.asarray(tf["interval_actuator_ctrl"], dtype=np.float64)

    pool = MultiJointWorkerPool(
        catalog,
        cfg.profile,
        cfg.n_workers,
        timeout_s=cfg.timeout_s,
        backend_factory=backend_factory,
    )
    try:
        sample_backend = pool._all[0][0]
        details = sample_backend.describe().details or {}
        clip_mode = (
            "one_sided" if bool(details.get("supports_one_sided_boundary_gradient", False)) else "native"
        )

        tasks = build_multifamily_tasks(pool, teacher_q7, teacher_v0, teacher_u15)
        train_tasks = [t for t in tasks if t.split == "train"]
        val_tasks = [t for t in tasks if t.split == "val"]

        all_conds = torch.stack([t.condition for t in train_tasks], dim=0)
        cond_mean = all_conds.mean(dim=0)
        cond_std = all_conds.std(dim=0).clamp_min(1e-6)

        if progress_callback:
            progress_callback(
                {
                    "phase": "invert_primitives",
                    "train_tasks": len(train_tasks),
                    "val_tasks": len(val_tasks),
                }
            )
        # Invert all train warm_exc to K=105 knots in one vectorized batch, and refine perturbed delta tasks via parallel MuJoCo L-BFGS
        train_knots = _invert_exc_to_knots_batch(
            [t.warm_exc if t.warm_exc is not None else teacher_u15 for t in train_tasks],
            cfg,
        )
        if cfg.anchor_lbfgs_steps > 0:
            pert_indices = [i for i, t in enumerate(train_tasks) if float(np.max(np.abs(t.delta_rad))) > 1e-9]
            if pert_indices:
                if progress_callback:
                    progress_callback(
                        {"phase": "lbfgs_refine_perturbed", "tasks": len(pert_indices), "steps": cfg.anchor_lbfgs_steps}
                    )
                solved_pert = _solve_lbfgs_batch(
                    pool,
                    [train_tasks[i] for i in pert_indices],
                    [train_knots[i] for i in pert_indices],
                    cfg,
                    clip_mode=clip_mode,
                )
                for idx, sol in zip(pert_indices, solved_pert):
                    train_knots[idx] = sol

        center_idx = next((i for i, t in enumerate(train_tasks) if "s1.00" in t.name), 0)
        if model_factory is not None:
            model = model_factory(cfg, train_knots[center_idx], cond_mean, cond_std)
        else:
            model = build_rate_policy_from_config(
                {
                    "architecture": cfg.architecture,
                    "condition_dim": 127,
                    "hidden_dim": cfg.hidden_dim,
                    "knots_per_pool": cfg.knots_per_pool,
                    "max_rate_hz": cfg.max_rate_hz,
                    "n_pools": NU,
                    "num_layers": cfg.num_layers,
                }
            )
            with torch.no_grad():
                if hasattr(model, "base_knots_hz"):
                    getattr(model, "base_knots_hz").copy_(torch.as_tensor(train_knots[center_idx], dtype=torch.float64))
                if hasattr(model, "condition_mean"):
                    getattr(model, "condition_mean").copy_(cond_mean.to(dtype=torch.float64))
                if hasattr(model, "condition_std"):
                    getattr(model, "condition_std").copy_(cond_std.to(dtype=torch.float64))

        anchor_conds = torch.stack([t.condition for t in train_tasks], dim=0)
        anchor_targets = torch.stack([torch.from_numpy(k) for k in train_knots], dim=0)
        opt_d = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        for _ in range(cfg.distill_steps):
            opt_d.zero_grad()
            pred = model(anchor_conds)
            loss_d = ((pred - anchor_targets) ** 2).mean()
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_d.step()

        best_val = evaluate_multijoint_tasks(pool, model, val_tasks, cfg, clip_mode=clip_mode)
        best_score = best_val["mean_mse_rad2"] + 0.25 * best_val["max_mse_rad2"]
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        history: list[dict[str, Any]] = [
            {"epoch": 0, "phase": "distill", "val": best_val, "score": best_score}
        ]
        if progress_callback:
            progress_callback({"epoch": 0, "val_mean_rmse_deg": best_val["mean_overall_rmse_deg"]})

        opt_e2e = torch.optim.AdamW(model.parameters(), lr=cfg.e2e_lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_e2e, T_max=max(1, cfg.e2e_epochs), eta_min=cfg.e2e_lr * 0.1
        )
        batch_size = min(pool.size, len(train_tasks))
        for epoch in range(1, cfg.e2e_epochs + 1):
            order = rng.permutation(len(train_tasks))
            train_rmses: list[float] = []
            for start in range(0, len(train_tasks), batch_size):
                batch_idx = order[start : start + batch_size]
                batch_tasks = [train_tasks[int(i)] for i in batch_idx]
                cur_bs = len(batch_tasks)
                conds_b = torch.stack([t.condition for t in batch_tasks], dim=0)
                opt_e2e.zero_grad()
                knots_b = model(conds_b)
                wf_b, exc_b = _knots_to_wf_exc(knots_b, cfg, clip_mode=clip_mode)
                excs_np = [exc_b[i].detach().cpu().numpy() for i in range(cur_bs)]
                evals = pool.eval_batch(
                    batch_tasks,
                    excs_np,
                    need_jacobian=True,
                    clip_boundary_gradient=clip_mode,
                    seed_base=10_000 + epoch * 100 + start,
                )
                regs_b = _reg_terms(knots_b, wf_b, exc_b, cfg)
                surr = torch.zeros((), dtype=torch.float64)
                fw = cfg.family_weights or {}
                w_sum = 0.0
                for i, (t, ev) in enumerate(zip(batch_tasks, evals)):
                    assert ev.control_vjp is not None
                    q_sub = ev.traces["qpos"][:, list(LF_QPOS_ADDRESSES)]
                    train_rmses.append(float(np.degrees(math.sqrt(np.mean((q_sub - t.target_rad) ** 2)))))
                    tw = float(fw.get(t.family, 1.0))
                    w_sum += tw
                    surr = surr + tw * (torch.sum(exc_b[i] * torch.from_numpy(ev.control_vjp)) + regs_b[i])
                (surr / max(w_sum, 1e-12)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_e2e.step()
            scheduler.step()

            val_m = evaluate_multijoint_tasks(pool, model, val_tasks, cfg, clip_mode=clip_mode)
            score = val_m["mean_mse_rad2"] + 0.25 * val_m["max_mse_rad2"]
            accepted = score < best_score
            if accepted:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                best_val = val_m
                best_epoch = epoch
            rec = {
                "epoch": epoch,
                "phase": "e2e_vjp",
                "train_batch_rmse_deg": float(np.mean(train_rmses)),
                "val_mean_rmse_deg": val_m["mean_overall_rmse_deg"],
                "val_max_rmse_deg": val_m["max_overall_rmse_deg"],
                "val_groups": val_m["group_summary"],
                "val_families": val_m["family_mean_rmse_deg"],
                "accepted": accepted,
            }
            history.append({"epoch": epoch, "phase": "e2e_vjp", "val": val_m, "score": score, "accepted": accepted})
            if progress_callback:
                progress_callback(rec)

        model.load_state_dict(best_state)
        train_eval = evaluate_multijoint_tasks(pool, model, train_tasks, cfg, clip_mode=clip_mode)
    finally:
        pool.close()

    ckpt_path = out_dir / "best_model.npz"
    arch_name = str(getattr(model, "architecture_name", cfg.architecture))
    model_config = {
        "architecture": arch_name,
        "model_type": type(model).__name__,
        "scope": "lf_multijoint_15muscles_7joints",
        "condition_dim": 127,
        "hidden_dim": cfg.hidden_dim,
        "num_layers": cfg.num_layers,
        "n_pools": NU,
        "knots_per_pool": cfg.knots_per_pool,
        "max_rate_hz": cfg.max_rate_hz,
        "controlled_muscles": NU,
        "teacher_assisted_muscles": 0,
        "target_joints": list(LF_JOINT_NAMES),
        "target_qpos_indices": list(LF_QPOS_ADDRESSES),
        "bridge": {
            "tau_ms": cfg.tau_ms,
            "rate_scale_hz": cfg.rate_scale_hz,
            "control_bin_ms": 2,
        },
        "regularization": {
            "energy_weight": cfg.energy_weight,
            "sat_weight": cfg.sat_weight,
            "smooth_weight": cfg.smooth_weight,
        },
    }
    metadata = {
        "schema_version": "sensory_translator_v2_multijoint_model_v1",
        "created_utc": utc_now_iso(),
        "profile": cfg.profile,
        "best_epoch": int(best_epoch),
        "controlled_muscles": 15,
        "teacher_assisted_muscles": 0,
        "val_mean_overall_rmse_deg": float(best_val["mean_overall_rmse_deg"]),
        "val_max_overall_rmse_deg": float(best_val["max_overall_rmse_deg"]),
        "val_group_summary": best_val["group_summary"],
        "val_family_mean_rmse_deg": best_val["family_mean_rmse_deg"],
        "train_mean_overall_rmse_deg": float(train_eval["mean_overall_rmse_deg"]),
    }
    save_rate_checkpoint(model, ckpt_path, config=model_config, metadata=metadata)
    summary = {
        "schema_version": "sensory_translator_v2_multijoint_training_report_v1",
        "created_utc": utc_now_iso(),
        "architecture": arch_name,
        "checkpoint_path": str(ckpt_path),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "best_epoch": int(best_epoch),
        "best_val": best_val,
        "train_eval": train_eval,
        "elapsed_s": time.perf_counter() - t_start,
        "history": history,
    }
    write_json_atomic(out_dir / "training_report.json", summary)
    return summary
