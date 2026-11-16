"""Low-energy fast-bridge ConditionalRateFFN training pipeline (v2 non-v1 assets)."""

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

from ..assets import AssetCatalog, load_asset_catalog
from ..backends.persistent_mujoco import PersistentMujocoBackend
from ..contracts import (
    Backend,
    Evaluation,
    EvaluationRequest,
    InitialState,
    STATE_STEPS,
    Session,
    SessionSpec,
    TARGET_QPOS_ADDRESS,
)
from ..evidence import utc_now_iso, write_json_atomic
from ..goals import encode_joint_condition, goal_from_scale
from .bridge import direct_pool_rate_waveform, rate_to_excitation
from .model import ConditionalRateFFN, save_rate_checkpoint
from .optim import lbfgs_direction


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for v2 fast-bridge low-energy E2E MLP training."""

    output_dir: Path
    profile: str = "adjacent_static_parent_v1"
    knots_per_pool: int = 105
    hidden_dim: int = 128
    max_rate_hz: float = 200.0
    tau_ms: float = 2.0
    rate_scale_hz: float = 100.0
    n_workers: int = 8
    anchor_lbfgs_steps: int = 6
    e2e_epochs: int = 12
    e2e_lr: float = 3e-4
    energy_weight: float = 2e-4
    sat_weight: float = 2e-6
    smooth_weight: float = 2e-7
    distill_steps: int = 500
    seed: int = 42
    timeout_s: float = 120.0
    asset_root: Path | None = None
    runtime_root: Path | None = None
    physics_python: Path | None = None


@dataclass(frozen=True)
class _Task:
    name: str
    scale: float
    delta_deg: float
    delta_rad: float
    target_rad: np.ndarray
    condition: torch.Tensor


class _WorkerPool:
    """Thread-safe pool of persistent MuJoCo (or custom) sessions for parallel VJP rollouts."""

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
        tasks: list[_Task],
        controls_list: list[np.ndarray],
        *,
        need_jacobian: bool,
        clip_boundary_gradient: str = "one_sided",
        seed_base: int = 1000,
    ) -> list[Evaluation]:
        def _run_one(idx_task: tuple[int, _Task, np.ndarray]) -> Evaluation:
            idx, task, ctrl = idx_task
            backend, session = self._slots.get()
            try:
                backend.reset(session, InitialState(task.delta_rad), seed=seed_base + idx)
                req = EvaluationRequest(
                    controls=ctrl,
                    need_jacobian=need_jacobian,
                    mse_target_rad=task.target_rad if need_jacobian else None,
                    clip_boundary_gradient=clip_boundary_gradient,
                    request_id=f"train_{task.name}_{idx}",
                )
                return backend.evaluate(session, req)
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


def _make_task(
    name: str,
    scale: float,
    delta_deg: float,
    teacher_q: np.ndarray,
    teacher_qvel0: float,
) -> _Task:
    goal = goal_from_scale(scale, teacher_q, initial_tibia_delta_deg=delta_deg)
    target = np.asarray(goal["target_rad"], dtype=np.float64)
    delta_rad = math.radians(delta_deg)
    q0 = float(teacher_q[0]) + delta_rad
    cond = encode_joint_condition(
        torch.from_numpy(target).unsqueeze(0),
        torch.tensor([q0], dtype=torch.float64),
        torch.tensor([teacher_qvel0], dtype=torch.float64),
    )[0]
    return _Task(
        name=name,
        scale=float(scale),
        delta_deg=float(delta_deg),
        delta_rad=delta_rad,
        target_rad=target,
        condition=cond,
    )


def _invert_teacher_to_knots(
    teacher_exc: np.ndarray,
    cfg: TrainingConfig,
    *,
    steps: int = 450,
) -> np.ndarray:
    """Invert frozen teacher excitation through fast-bridge dynamics in pure PyTorch."""
    knots = torch.full((1, 2, cfg.knots_per_pool), 12.0, dtype=torch.float64, requires_grad=True)
    target_t = torch.from_numpy(teacher_exc[None, :, :].astype(np.float64))
    opt = torch.optim.Adam([knots], lr=2.0)
    for _ in range(steps):
        opt.zero_grad()
        clamped = knots.clamp(0.0, cfg.max_rate_hz)
        wf = direct_pool_rate_waveform(clamped, max_rate_hz=cfg.max_rate_hz)
        exc = rate_to_excitation(
            wf,
            max_rate_hz=cfg.max_rate_hz,
            rate_scale_hz=cfg.rate_scale_hz,
            tau_ms=cfg.tau_ms,
            clip_boundary_gradient="one_sided",
        )
        mse = ((exc - target_t) ** 2).mean()
        smooth = 2e-5 * (knots[:, :, 1:] - knots[:, :, :-1]).pow(2).mean()
        energy = 1e-4 * (exc ** 2).mean()
        sat = 5e-7 * torch.relu(wf - 94.0).pow(2).mean()
        loss = mse + smooth + energy + sat
        loss.backward()
        opt.step()
        with torch.no_grad():
            knots.clamp_(0.3, min(180.0, cfg.max_rate_hz - 5.0))
    return knots[0].detach().cpu().numpy().copy()


def _knots_to_wf_exc(
    knots: torch.Tensor,
    cfg: TrainingConfig,
    *,
    clip_mode: str = "one_sided",
) -> tuple[torch.Tensor, torch.Tensor]:
    clamped = knots.clamp(0.0, cfg.max_rate_hz)
    wf = direct_pool_rate_waveform(clamped, max_rate_hz=cfg.max_rate_hz)
    exc = rate_to_excitation(
        wf,
        max_rate_hz=cfg.max_rate_hz,
        rate_scale_hz=cfg.rate_scale_hz,
        tau_ms=cfg.tau_ms,
        clip_boundary_gradient=clip_mode,
    )
    return wf, exc


def _regularization_terms(
    knots: torch.Tensor,
    wf: torch.Tensor,
    exc: torch.Tensor,
    cfg: TrainingConfig,
) -> torch.Tensor:
    """Compute per-sample regularization vector [B] for energy, saturation, and smoothness."""
    energy = cfg.energy_weight * (exc ** 2).mean(dim=(1, 2))
    sat = cfg.sat_weight * torch.relu(wf - 92.0).pow(2).mean(dim=(1, 2))
    smooth = cfg.smooth_weight * (knots[:, :, 1:] - knots[:, :, :-1]).pow(2).mean(dim=(1, 2))
    return energy + sat + smooth


def _solve_anchor_knots(
    pool: _WorkerPool,
    anchors: list[_Task],
    base_knots: np.ndarray,
    cfg: TrainingConfig,
    *,
    clip_mode: str,
) -> list[np.ndarray]:
    """Solve low-energy regularized trajectories in parallel across representative anchor tasks."""
    n = len(anchors)
    xs = [base_knots.copy() for _ in range(n)]
    s_hists: list[list[np.ndarray]] = [[] for _ in range(n)]
    y_hists: list[list[np.ndarray]] = [[] for _ in range(n)]
    prev_xs: list[np.ndarray | None] = [None] * n
    prev_gs: list[np.ndarray | None] = [None] * n
    alphas = (2.0, 1.0, 0.5, 0.25, 0.1, 0.025)

    def _eval_reg_loss(x_np: np.ndarray, mse_val: float) -> float:
        with torch.no_grad():
            kt = torch.from_numpy(x_np[None, :, :])
            wf, exc = _knots_to_wf_exc(kt, cfg, clip_mode=clip_mode)
            reg = float(_regularization_terms(kt, wf, exc, cfg)[0].item())
        return mse_val + reg

    for step in range(cfg.anchor_lbfgs_steps):
        excs: list[np.ndarray] = []
        with torch.no_grad():
            for x in xs:
                _, e = _knots_to_wf_exc(torch.from_numpy(x[None, :, :]), cfg, clip_mode=clip_mode)
                excs.append(e[0].cpu().numpy())
        evals = pool.eval_batch(
            anchors,
            excs,
            need_jacobian=True,
            clip_boundary_gradient=clip_mode,
            seed_base=5000 + step * 100,
        )
        curr_losses: list[float] = []
        dirs: list[np.ndarray] = []
        grads_np: list[np.ndarray] = []
        for idx, (anchor, x_np, ev) in enumerate(zip(anchors, xs, evals)):
            mse = float(np.mean((ev.joint_rad - anchor.target_rad) ** 2))
            kt = torch.tensor(x_np[None, :, :], dtype=torch.float64, requires_grad=True)
            wf, exc = _knots_to_wf_exc(kt, cfg, clip_mode=clip_mode)
            reg = _regularization_terms(kt, wf, exc, cfg)[0]
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

        # Evaluate candidate alpha trials in parallel batches per alpha
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
                anchors,
                trial_excs,
                need_jacobian=False,
                seed_base=6000 + step * 100 + j * 10,
            )
            for i in range(n):
                mse_t = float(np.mean((trial_evals[i].joint_rad - anchors[i].target_rad) ** 2))
                loss_t = _eval_reg_loss(trial_xs[i], mse_t)
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


def _evaluate_tasks(
    pool: _WorkerPool,
    model: ConditionalRateFFN,
    tasks: list[_Task],
    cfg: TrainingConfig,
    *,
    clip_mode: str,
) -> dict[str, Any]:
    conds = torch.stack([t.condition for t in tasks], dim=0)
    with torch.no_grad():
        knots = model(conds)
        wf, exc = _knots_to_wf_exc(knots, cfg, clip_mode=clip_mode)
        excs = [exc[i].cpu().numpy() for i in range(len(tasks))]
        wf_np = wf.cpu().numpy()
    evals = pool.eval_batch(tasks, excs, need_jacobian=False, seed_base=9000)
    mses = [float(np.mean((ev.joint_rad - t.target_rad) ** 2)) for ev, t in zip(evals, tasks)]
    rmses = [float(np.degrees(math.sqrt(m))) for m in mses]
    mean_exc2 = float(np.mean(np.square(exc.cpu().numpy())))
    max_wf_hz = float(np.max(wf_np))
    frac_gt95 = float(np.mean(wf_np > 95.0))
    return {
        "mean_mse_rad2": float(np.mean(mses)),
        "max_mse_rad2": float(np.max(mses)),
        "mean_rmse_deg": float(np.mean(rmses)),
        "max_rmse_deg": float(np.max(rmses)),
        "mean_exc2": mean_exc2,
        "max_wf_hz": max_wf_hz,
        "frac_wf_gt95": frac_gt95,
        "per_task_rmse_deg": {t.name: r for t, r in zip(tasks, rmses)},
    }


def train_rate_model(
    cfg: TrainingConfig,
    *,
    backend_factory: Callable[[AssetCatalog, str], Backend] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train a fast-bridge low-energy ConditionalRateFFN and write non-v1 v2 artifacts."""
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
        teacher_q = np.asarray(tf["qpos"][:, TARGET_QPOS_ADDRESS], dtype=np.float64)
        teacher_v0 = float(tf["qvel"][0, TARGET_QPOS_ADDRESS])
        teacher_exc = np.asarray(tf["interval_actuator_ctrl"][:, [13, 14]], dtype=np.float64)

    # Disjoint training & validation sets plus headline benchmark tracking set
    train_tasks: list[_Task] = []
    for s in (0.80, 0.85, 0.90, 0.95, 0.975, 1.00, 1.05, 1.10, 1.15):
        train_tasks.append(_make_task(f"s_{s:.3f}_d+0.0", s, 0.0, teacher_q, teacher_v0))
    for d in (-10.0, -5.0, 5.0, 10.0):
        train_tasks.append(_make_task(f"s_0.975_d{d:+.1f}", 0.975, d, teacher_q, teacher_v0))
    for s, d in ((0.85, -5.0), (0.85, 5.0), (1.10, -5.0), (1.10, 5.0)):
        train_tasks.append(_make_task(f"s_{s:.3f}_d{d:+.1f}", s, d, teacher_q, teacher_v0))

    val_tasks = [
        _make_task(f"val_s_{s:.3f}_d{d:+.1f}", s, d, teacher_q, teacher_v0)
        for s, d in (
            (0.825, 0.0),
            (0.925, 0.0),
            (1.025, 0.0),
            (1.125, 0.0),
            (0.975, -7.5),
            (0.975, 7.5),
        )
    ]
    anchor_tasks = [
        _make_task(f"anc_s_{s:.3f}_d{d:+.1f}", s, d, teacher_q, teacher_v0)
        for s, d in (
            (0.800, 0.0),
            (0.900, 0.0),
            (0.975, 0.0),
            (1.000, 0.0),
            (1.100, 0.0),
            (1.150, 0.0),
            (0.975, -10.0),
            (0.975, 10.0),
        )
    ]

    all_conds = torch.stack([t.condition for t in train_tasks], dim=0)
    cond_mean = all_conds.mean(dim=0)
    cond_std = all_conds.std(dim=0).clamp_min(1e-6)

    base_knots = _invert_teacher_to_knots(teacher_exc, cfg)

    pool = _WorkerPool(
        catalog,
        cfg.profile,
        min(cfg.n_workers, len(anchor_tasks)),
        timeout_s=cfg.timeout_s,
        backend_factory=backend_factory,
    )
    try:
        # Infer clip mode directly from backend capability (one_sided for persistent MuJoCo, native for fake)
        sample_backend = pool._all[0][0]
        details = sample_backend.describe().details or {}
        clip_mode = (
            "one_sided" if bool(details.get("supports_one_sided_boundary_gradient", False)) else "native"
        )

        # Phase 1: Parallel MuJoCo L-BFGS on representative anchors (keeps everything in teacher basin)
        if progress_callback:
            progress_callback({"phase": "anchor_solve", "anchors": len(anchor_tasks)})
        anchor_knots = _solve_anchor_knots(pool, anchor_tasks, base_knots, cfg, clip_mode=clip_mode)

        # Pick central scale=1.000 anchor as residual zero base_knots
        center_idx = 3  # (1.000, 0.0)
        model = ConditionalRateFFN(
            condition_dim=19,
            hidden_dim=cfg.hidden_dim,
            knots_per_pool=cfg.knots_per_pool,
            max_rate_hz=cfg.max_rate_hz,
            base_knots_hz=anchor_knots[center_idx],
            condition_mean=cond_mean,
            condition_std=cond_std,
        )

        # Phase 2A: Pure PyTorch distillation onto solved anchor trajectories
        anchor_conds = torch.stack([a.condition for a in anchor_tasks], dim=0)
        anchor_targets = torch.stack([torch.from_numpy(k) for k in anchor_knots], dim=0)
        opt_distill = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(cfg.distill_steps):
            opt_distill.zero_grad()
            pred = model(anchor_conds)
            loss_d = ((pred - anchor_targets) ** 2).mean()
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_distill.step()

        # Baseline validation evaluation right after Phase 2A distillation
        val_metrics_0 = _evaluate_tasks(pool, model, val_tasks, cfg, clip_mode=clip_mode)
        best_score = val_metrics_0["mean_mse_rad2"] + 0.25 * val_metrics_0["max_mse_rad2"]
        best_state = copy.deepcopy(model.state_dict())
        best_val_metrics = val_metrics_0
        best_epoch = 0

        history: list[dict[str, Any]] = [
            {"epoch": 0, "phase": "distill", "val": val_metrics_0, "score": best_score}
        ]
        if progress_callback:
            progress_callback(history[-1])

        # Phase 2B: End-to-End MuJoCo VJP fine-tuning across full task bank
        opt_e2e = torch.optim.AdamW(model.parameters(), lr=cfg.e2e_lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_e2e, T_max=max(1, cfg.e2e_epochs), eta_min=cfg.e2e_lr * 0.1
        )
        batch_size = min(pool.size, len(train_tasks))
        for epoch in range(1, cfg.e2e_epochs + 1):
            order = rng.permutation(len(train_tasks))
            # Ensure edge cases get frequent coverage alongside random samples
            batch_indices = list(order[:batch_size])
            batch_tasks = [train_tasks[i] for i in batch_indices]

            conds_b = torch.stack([t.condition for t in batch_tasks], dim=0)
            opt_e2e.zero_grad()
            knots_b = model(conds_b)
            wf_b, exc_b = _knots_to_wf_exc(knots_b, cfg, clip_mode=clip_mode)
            excs_np = [exc_b[i].detach().cpu().numpy() for i in range(batch_size)]

            evals = pool.eval_batch(
                batch_tasks,
                excs_np,
                need_jacobian=True,
                clip_boundary_gradient=clip_mode,
                seed_base=10_000 + epoch * 100,
            )
            regs_b = _regularization_terms(knots_b, wf_b, exc_b, cfg)
            physics_surr = torch.zeros((), dtype=torch.float64)
            train_mses: list[float] = []
            for i, (t, ev) in enumerate(zip(batch_tasks, evals)):
                assert ev.control_vjp is not None
                train_mses.append(float(np.mean((ev.joint_rad - t.target_rad) ** 2)))
                physics_surr = physics_surr + (
                    torch.sum(exc_b[i] * torch.from_numpy(ev.control_vjp)) + regs_b[i]
                )
            (physics_surr / float(batch_size)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_e2e.step()
            scheduler.step()

            val_metrics = _evaluate_tasks(pool, model, val_tasks, cfg, clip_mode=clip_mode)
            score = val_metrics["mean_mse_rad2"] + 0.25 * val_metrics["max_mse_rad2"]
            accepted = score < best_score
            if accepted:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                best_val_metrics = val_metrics
                best_epoch = epoch

            rec = {
                "epoch": epoch,
                "phase": "e2e_vjp",
                "train_batch_rmse_deg": float(np.mean([math.degrees(math.sqrt(m)) for m in train_mses])),
                "val": val_metrics,
                "score": score,
                "accepted": accepted,
            }
            history.append(rec)
            if progress_callback:
                progress_callback(rec)

        model.load_state_dict(best_state)
        anchor_eval = _evaluate_tasks(pool, model, anchor_tasks, cfg, clip_mode=clip_mode)
    finally:
        pool.close()

    ckpt_path = out_dir / "best_model.npz"
    model_config = {
        "model_type": "ConditionalRateFFN",
        "condition_dim": 19,
        "hidden_dim": cfg.hidden_dim,
        "knots_per_pool": cfg.knots_per_pool,
        "max_rate_hz": cfg.max_rate_hz,
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
        "schema_version": "sensory_translator_v2_trained_rate_model_v1",
        "created_utc": utc_now_iso(),
        "profile": cfg.profile,
        "best_epoch": int(best_epoch),
        "val_mean_rmse_deg": float(best_val_metrics["mean_rmse_deg"]),
        "val_max_rmse_deg": float(best_val_metrics["max_rmse_deg"]),
        "anchor_mean_rmse_deg": float(anchor_eval["mean_rmse_deg"]),
        "anchor_max_rmse_deg": float(anchor_eval["max_rmse_deg"]),
        "mean_exc2": float(best_val_metrics["mean_exc2"]),
        "max_wf_hz": float(best_val_metrics["max_wf_hz"]),
        "frac_wf_gt95": float(best_val_metrics["frac_wf_gt95"]),
    }
    save_rate_checkpoint(model, ckpt_path, config=model_config, metadata=metadata)

    summary = {
        "schema_version": "sensory_translator_v2_training_report_v1",
        "created_utc": utc_now_iso(),
        "checkpoint_path": str(ckpt_path),
        "config": {
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        },
        "best_epoch": int(best_epoch),
        "best_val": best_val_metrics,
        "anchor_eval": anchor_eval,
        "elapsed_s": time.perf_counter() - t_start,
        "history": history,
    }
    write_json_atomic(out_dir / "training_report.json", summary)
    return summary
