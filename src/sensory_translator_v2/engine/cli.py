"""Command-line entry point for REvFLY."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .. import __version__
from ..assets import export_asset_bundle, load_asset_catalog, verify_asset_catalog
from ..backends.base import FakeBackend
from ..backends.persistent_mujoco import PersistentMujocoBackend
from ..contracts import (
    AssetValidationError,
    DEFAULT_FEEDBACK_KD,
    DEFAULT_FEEDBACK_KP,
    InitialState,
    RequestValidationError,
    STATE_STEPS,
    SessionSpec,
    require_real_float,
)
from ..rate.model import RATE_ARCHITECTURES
from ..rate.multijoint import MultiJointTrainingConfig, train_multijoint_rate_model
from ..rate.trainer import TrainingConfig, train_rate_model
from ..runtime_install import install_runtime_bundle
from .benchmark import run_matched_suite
from .core import DEFAULT_THRESHOLD_DEG, EngineV2, EngineV2Config
from .multijoint import (
    MULTIJOINT_GOAL_FAMILIES,
    MultiJointEngine,
    MultiJointEngineConfig,
    evaluate_bio_realization,
)
from .server import RateControlV2App, RateControlV2HTTPServer, _UI_ASSETS, _find_ui_root


def _parse_family_weights(raw: str | None) -> dict[str, float] | None:
    """Parse CLI `--family-weights` mapping like `swing_scaled=1,reach_and_hold=2`."""
    if raw is None or not raw.strip():
        return None
    weights: dict[str, float] = {}
    for entry in raw.split(","):
        item = entry.strip()
        if not item:
            continue
        if "=" not in item:
            raise RequestValidationError(f"Invalid family weight entry {item!r}; expected family=weight")
        key, val = item.split("=", 1)
        family = key.strip()
        if not family:
            raise RequestValidationError("Empty family name in --family-weights")
        weights[family] = require_real_float(float(val.strip()), f"family_weights[{family}]", positive=True)
    return weights or None


def _build_engine_from_args(args: argparse.Namespace) -> EngineV2:
    cfg = EngineV2Config(
        backend=args.backend,
        profile=args.profile,
        model_id=args.model_id,
        checkpoint_path=Path(args.checkpoint) if getattr(args, "checkpoint", None) else None,
        asset_root=args.asset_root,
        runtime_root=args.runtime_root,
        physics_python=args.physics_python,
        scripts_root=args.scripts_root,
        timeout_s=args.timeout_s,
    )
    return EngineV2(cfg)


def _load_target_file(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "target_rad" not in data.files:
                raise RequestValidationError("NPZ target file must contain 'target_rad'")
            target = np.asarray(data["target_rad"], dtype=np.float64)
            time_s = np.asarray(data["time_s"], dtype=np.float64) if "time_s" in data.files else None
            return target, time_s
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "target_rad" not in payload:
                raise RequestValidationError("JSON target object must contain 'target_rad'")
            target = np.asarray(payload["target_rad"], dtype=np.float64)
            time_s = np.asarray(payload["time_s"], dtype=np.float64) if "time_s" in payload else None
            return target, time_s
        return np.asarray(payload, dtype=np.float64), None
    raise RequestValidationError(f"Unsupported --target file extension: {path.suffix!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="revfly",
        description="REvFLY motor-rate control and simulation CLI.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--backend",
        choices=("v1_subprocess", "persistent_fresh", "persistent_cached", "fake"),
        default="persistent_cached",
        help="Physics evaluation backend (default: persistent_cached)",
    )
    common.add_argument(
        "--profile",
        choices=("original_v1", "multiccd_v1", "adjacent_static_parent_v1"),
        default=None,
        help="Physics profile override",
    )
    common.add_argument("--model-id", default=None, help="Registered v2/v1 model ID")
    common.add_argument("--checkpoint", default=None, help="Custom rate checkpoint NPZ path")
    common.add_argument("--asset-root", default=None, help="Content-addressed bundle or v1 checkout root")
    common.add_argument("--runtime-root", default=None, help="Writable runtime cache & output directory")
    common.add_argument("--physics-python", default=None, help="Python executable containing MuJoCo")
    common.add_argument("--scripts-root", default=None, help="Optional v1 scripts directory for v1_subprocess")
    common.add_argument("--timeout-s", type=float, default=90.0, help="Per-call worker timeout in seconds")

    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capabilities", parents=[common], help="Print engine v2 capabilities JSON")
    cap.add_argument("--output-json", default=None, help="Optional file path to write JSON")

    ver = sub.add_parser("verify-assets", parents=[common], help="Verify asset bundle or checkout hashes")
    ver.add_argument("--output-json", default=None, help="Optional file path to write verification report")

    install = sub.add_parser("install-assets", help="Install a verified release ZIP or an extracted runtime bundle")
    install.add_argument("source", type=Path, help="Local release ZIP or bundle directory")
    install.add_argument("--destination", type=Path, default=None, help="Install under a custom asset root")
    install.add_argument("--replace", action="store_true", help="Replace an existing runtime bundle after verification")

    exp = sub.add_parser("export-assets", help="Export verified assets into a portable content-addressed v2 bundle")
    exp.add_argument("--dest", required=True, help="Destination bundle directory")
    exp.add_argument("--asset-root", default=None, help="Source asset root (default: auto-discover checkout)")
    exp.add_argument("--overwrite", action="store_true", help="Overwrite existing bundle files")
    exp.add_argument("--include-v2-models", action="store_true", help="Include verified V2 checkpoints when available")
    exp.add_argument("--include-ui", action="store_true", help="Copy the built control workbench into the portable bundle")

    run_p = sub.add_parser("run", parents=[common], help="Run single trajectory inference and physics verification")
    goal_grp = run_p.add_mutually_exclusive_group(required=True)
    goal_grp.add_argument("--goal-scale", type=float, help="Registered-family trajectory scale factor (> 0)")
    goal_grp.add_argument("--target", type=Path, help="JSON or NPZ file with target_rad[211]")
    run_p.add_argument("--output", required=True, type=Path, help="Fresh directory for run artifacts")
    run_p.add_argument("--mode", choices=("fast", "auto", "refine", "closed_loop"), default="auto")
    run_p.add_argument("--threshold-deg", type=float, default=DEFAULT_THRESHOLD_DEG)
    run_p.add_argument("--correction-steps", type=int, default=3)
    run_p.add_argument("--correction-lr-hz", type=float, default=0.05)
    run_p.add_argument(
        "--correction-optimizer",
        choices=("lbfgs_linesearch", "adam"),
        default="lbfgs_linesearch",
    )
    run_p.add_argument("--feedback-kp", type=float, default=DEFAULT_FEEDBACK_KP)
    run_p.add_argument("--feedback-kd", type=float, default=DEFAULT_FEEDBACK_KD)
    run_p.add_argument("--initial-tibia-delta-deg", type=float, default=0.0)
    run_p.add_argument("--render", action="store_true", help="Render top-view simulation video")
    run_p.add_argument("--check-directions", action="store_true", help="Run FD vs adjoint directional gates")

    srv = sub.add_parser("serve", parents=[common], help="Start the local HTTP API server")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8766)
    srv.add_argument("--output-base", type=Path, default=None)
    srv.add_argument("--ui", action="store_true", help="Serve the built Control Studio UI")
    srv.add_argument("--multijoint-checkpoint", type=Path, default=None, help="Override only the 15-muscle multi-joint checkpoint")

    bmp = sub.add_parser("benchmark", help="Run matched cold/warm/HTTP cross-backend benchmark")
    bmp.add_argument(
        "--backends",
        nargs="+",
        choices=("v1_subprocess", "persistent_fresh", "persistent_cached", "fake"),
        default=["v1_subprocess", "persistent_fresh", "persistent_cached"],
    )
    bmp.add_argument("--asset-root", default=None)
    bmp.add_argument("--runtime-root", default=None)
    bmp.add_argument("--physics-python", default=None)
    bmp.add_argument("--mode", choices=("fast", "auto", "refine"), default="auto")
    bmp.add_argument("--correction-steps", type=int, default=6)
    bmp.add_argument("--output-json", default=None)
    bmp.add_argument("--no-http", action="store_true")

    trp = sub.add_parser(
        "train-rate-model",
        help="Train a low-energy fast-bridge ConditionalRateFFN model into a non-v1 directory",
    )
    trp.add_argument("--output-dir", required=True, type=Path)
    trp.add_argument(
        "--profile",
        choices=("original_v1", "multiccd_v1", "adjacent_static_parent_v1"),
        default="adjacent_static_parent_v1",
    )
    trp.add_argument("--knots-per-pool", type=int, default=105)
    trp.add_argument("--hidden-dim", type=int, default=128)
    trp.add_argument("--tau-ms", type=float, default=2.0)
    trp.add_argument("--n-workers", type=int, default=8)
    trp.add_argument("--anchor-steps", type=int, default=6)
    trp.add_argument("--epochs", type=int, default=12)
    trp.add_argument("--e2e-lr", type=float, default=3e-4)
    trp.add_argument("--energy-weight", type=float, default=2e-4)
    trp.add_argument("--sat-weight", type=float, default=2e-6)
    trp.add_argument("--smooth-weight", type=float, default=2e-7)
    trp.add_argument("--seed", type=int, default=42)
    trp.add_argument("--timeout-s", type=float, default=120.0)
    trp.add_argument("--asset-root", type=Path, default=None)
    trp.add_argument("--runtime-root", type=Path, default=None)
    trp.add_argument("--physics-python", type=Path, default=None)

    tmj = sub.add_parser(
        "train-multijoint",
        help="Train 15-muscle (0 teacher-assist), 7-DOF LF multi-family rate policy",
    )
    tmj.add_argument("--output-dir", type=Path, required=True)
    tmj.add_argument("--profile", default="adjacent_static_parent_v1")
    tmj.add_argument(
        "--architecture",
        default="GoalStateConditionedContinuousRateFFN",
        help=f"Rate model architecture (registered: {', '.join(sorted(RATE_ARCHITECTURES))})",
    )
    tmj.add_argument(
        "--family-weights",
        default=None,
        help="Optional comma-separated loss weights, e.g. swing_scaled=1,reach_and_hold=2",
    )
    tmj.add_argument("--knots-per-pool", type=int, default=105)
    tmj.add_argument("--hidden-dim", type=int, default=256)
    tmj.add_argument("--num-layers", type=int, default=2)
    tmj.add_argument("--tau-ms", type=float, default=2.0)
    tmj.add_argument("--n-workers", type=int, default=8)
    tmj.add_argument("--anchor-steps", type=int, default=8)
    tmj.add_argument("--epochs", type=int, default=18)
    tmj.add_argument("--e2e-lr", type=float, default=3e-4)
    tmj.add_argument("--seed", type=int, default=42)
    tmj.add_argument("--timeout-s", type=float, default=120.0)
    tmj.add_argument("--asset-root", type=Path, default=None)
    tmj.add_argument("--runtime-root", type=Path, default=None)
    tmj.add_argument("--physics-python", type=Path, default=None)

    rmj = sub.add_parser(
        "run-multijoint",
        help="Run 15-muscle (0 teacher-assist) 7-DOF LF control & optional spike/LIF/MaleCNS bio bridge",
    )
    rmj.add_argument("--output", type=Path, default=None)
    rmj.add_argument("--checkpoint", type=Path, default=None)
    rmj.add_argument("--profile", default="adjacent_static_parent_v1")
    rmj.add_argument("--backend", default="persistent_cached")
    rmj.add_argument("--target-file", type=Path, default=None)
    rmj.add_argument(
        "--goal-family",
        default="swing_scaled",
        choices=MULTIJOINT_GOAL_FAMILIES,
        help="Goal trajectory family when --target-file is omitted",
    )
    rmj.add_argument("--scale", type=float, default=1.0)
    rmj.add_argument("--mode", default="auto")
    rmj.add_argument("--correction-steps", type=int, default=4)
    rmj.add_argument("--threshold-deg", type=float, default=5.729577951308233)
    rmj.add_argument("--feedback-kp", type=float, default=8.0)
    rmj.add_argument("--feedback-kd", type=float, default=0.04)
    rmj.add_argument("--init-delta-deg", default=None, help="7 comma-separated floats in deg")
    rmj.add_argument(
        "--bio-mode",
        default="none",
        choices=("none", "deterministic_isi", "poisson_refractory", "direct_lif", "premotor_connectome"),
    )
    rmj.add_argument("--bio-seed", type=int, default=0)
    rmj.add_argument("--bio-physics-vjp-steps", type=int, default=0)
    rmj.add_argument("--bio-population-scale", type=int, default=1)
    rmj.add_argument("--render", action="store_true")
    rmj.add_argument("--asset-root", type=Path, default=None)
    rmj.add_argument("--runtime-root", type=Path, default=None)
    rmj.add_argument("--physics-python", type=Path, default=None)

    bio = sub.add_parser(
        "bio-realize",
        help="Realize K=105 rate traces into spikes, LIF currents, or MaleCNS premotor connectome drive",
    )
    bio.add_argument("--traces-npz", type=Path, required=True)
    bio.add_argument("--output", type=Path, default=None)
    bio.add_argument("--scope", default="tibia", choices=("tibia", "lf15"))
    bio.add_argument(
        "--bio-mode",
        default="deterministic_isi",
        choices=("deterministic_isi", "poisson_refractory", "direct_lif", "premotor_connectome"),
    )
    bio.add_argument("--profile", default="adjacent_static_parent_v1")
    bio.add_argument("--backend", default="persistent_cached")
    bio.add_argument(
        "--tau-bridge-ms",
        type=float,
        default=None,
        help="Bridge filter tau_ms (default: from traces NPZ if present, else 20.0 tibia / 2.0 lf15)",
    )
    bio.add_argument("--population-scale", type=int, default=1)
    bio.add_argument("--physics-vjp-steps", type=int, default=12)
    bio.add_argument("--seed", type=int, default=0)
    bio.add_argument("--asset-root", type=Path, default=None)
    bio.add_argument("--runtime-root", type=Path, default=None)
    bio.add_argument("--physics-python", type=Path, default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "install-assets":
        try:
            report = install_runtime_bundle(args.source, destination=args.destination, replace=args.replace)
        except AssetValidationError as exc:
            parser.error(str(exc))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if args.command == "export-assets":
        ui_root = _find_ui_root(args.asset_root) if args.include_ui else None
        if args.include_ui and ui_root is None:
            parser.error("--include-ui needs built workbench files. Run npm run build from apps/control_ui.")
        report = export_asset_bundle(
            args.dest,
            asset_root=args.asset_root,
            overwrite=bool(args.overwrite),
            include_v2_models=args.include_v2_models,
        )
        if ui_root is not None:
            destination = Path(args.dest).expanduser().resolve()
            ui_paths = sorted({name for name, _mime in _UI_ASSETS.values()})
            for name in ui_paths:
                target = destination / name
                if target.resolve() != target:
                    parser.error(f"Refusing to export workbench file through a symbolic link: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = ui_root / name
                if source != target:
                    shutil.copyfile(source, target)
            report["ui_files"] = ui_paths
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if args.command == "benchmark":
        report = run_matched_suite(
            backends=args.backends,
            asset_root=args.asset_root,
            runtime_root=args.runtime_root,
            physics_python=args.physics_python,
            output_json=args.output_json,
            mode=args.mode,
            correction_steps=args.correction_steps,
            include_http=not args.no_http,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if args.command == "verify-assets":
        catalog = load_asset_catalog(
            asset_root=args.asset_root,
            runtime_root=args.runtime_root,
            physics_python=args.physics_python,
            scripts_root=args.scripts_root,
            verify_hashes=True,
        )
        report = verify_asset_catalog(catalog)
        rendered = json.dumps(report, indent=2, ensure_ascii=False)
        if args.output_json:
            Path(args.output_json).write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0

    if args.command == "capabilities":
        engine = _build_engine_from_args(args)
        try:
            payload = engine.capabilities()
        finally:
            engine.close()
        rendered = json.dumps(payload, indent=2, ensure_ascii=False)
        if args.output_json:
            Path(args.output_json).write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0 if payload.get("ready") else 1

    if args.command == "run":
        engine = _build_engine_from_args(args)
        try:
            if args.goal_scale is not None:
                goal = engine.goal_from_scale(
                    require_real_float(args.goal_scale, "goal_scale", positive=True),
                    initial_tibia_delta_deg=args.initial_tibia_delta_deg,
                )
                target_rad = goal["target_rad"]
                time_s = goal["time_s"]
            else:
                target_rad, time_s = _load_target_file(args.target)
                if time_s is None:
                    time_s = [i * 0.002 for i in range(STATE_STEPS)]
            result = engine.run(
                target_rad,
                output=args.output,
                mode=args.mode,
                threshold_deg=args.threshold_deg,
                correction_steps=args.correction_steps,
                correction_lr_hz=args.correction_lr_hz,
                correction_optimizer=args.correction_optimizer,
                feedback_kp=args.feedback_kp,
                feedback_kd=args.feedback_kd,
                initial_tibia_delta_deg=args.initial_tibia_delta_deg,
                time_s=time_s,
                render=bool(args.render),
                check_directions=bool(args.check_directions),
            )
            summary = {
                "schema_version": result["schema_version"],
                "backend": result["backend"]["name"],
                "profile": result["profile"],
                "is_certified_frozen_physics": result["is_certified_frozen_physics"],
                "best_rmse_deg": result["best_rmse_deg"],
                "accuracy_threshold_met": result["accuracy_threshold_met"],
                "selected_step": result["selected_step"],
                "timing_s": result["timing_s"],
                "output": str(Path(args.output).resolve()),
                "artifacts": result["artifacts"],
            }
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0
        finally:
            engine.close()

    if args.command == "serve":
        cfg = EngineV2Config(
            backend=args.backend,
            profile=args.profile,
            model_id=args.model_id,
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            asset_root=args.asset_root,
            runtime_root=args.runtime_root,
            physics_python=args.physics_python,
            scripts_root=args.scripts_root,
            timeout_s=args.timeout_s,
        )
        multijoint_config = None
        if args.multijoint_checkpoint is not None:
            multijoint_config = MultiJointEngineConfig(
                backend=args.backend,
                checkpoint_path=args.multijoint_checkpoint,
                asset_root=args.asset_root,
                runtime_root=args.runtime_root,
                physics_python=args.physics_python,
                timeout_s=args.timeout_s,
            )
        app = RateControlV2App(
            ui_enabled=bool(args.ui),
            config=cfg,
            multijoint_config=multijoint_config,
            output_base=args.output_base,
            asset_root=args.asset_root,
        )
        server = RateControlV2HTTPServer((args.host, int(args.port)), app=app)
        stop_event = threading.Event()

        def _handle_sig(_signum: int, _frame: Any) -> None:
            stop_event.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

        prev_int = signal.getsignal(signal.SIGINT)
        prev_term = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, _handle_sig)
        signal.signal(signal.SIGTERM, _handle_sig)
        try:
            print(
                json.dumps(
                    {
                        "listening": f"http://{server.server_address[0]}:{server.server_address[1]}",
                        "backend": cfg.backend,
                        "profile": cfg.profile,
                    }
                ),
                flush=True,
            )
            server.serve_forever()
        finally:
            signal.signal(signal.SIGINT, prev_int)
            signal.signal(signal.SIGTERM, prev_term)
            server.server_close()
        return 0

    if args.command == "train-rate-model":
        cfg_tr = TrainingConfig(
            output_dir=Path(args.output_dir),
            profile=args.profile,
            knots_per_pool=int(args.knots_per_pool),
            hidden_dim=int(args.hidden_dim),
            tau_ms=float(args.tau_ms),
            n_workers=int(args.n_workers),
            anchor_lbfgs_steps=int(args.anchor_steps),
            e2e_epochs=int(args.epochs),
            e2e_lr=float(args.e2e_lr),
            energy_weight=float(args.energy_weight),
            sat_weight=float(args.sat_weight),
            smooth_weight=float(args.smooth_weight),
            seed=int(args.seed),
            timeout_s=float(args.timeout_s),
            asset_root=args.asset_root,
            runtime_root=args.runtime_root,
            physics_python=args.physics_python,
        )
        summary = train_rate_model(
            cfg_tr,
            progress_callback=lambda ev: print(json.dumps(ev, ensure_ascii=False), flush=True),
        )
        print(
            json.dumps(
                {
                    "checkpoint_path": summary["checkpoint_path"],
                    "best_epoch": summary["best_epoch"],
                    "best_val": summary["best_val"],
                    "anchor_eval": summary["anchor_eval"],
                    "elapsed_s": summary["elapsed_s"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "train-multijoint":
        cfg_mj = MultiJointTrainingConfig(
            output_dir=args.output_dir,
            profile=args.profile,
            architecture=str(args.architecture),
            family_weights=_parse_family_weights(args.family_weights),
            knots_per_pool=int(args.knots_per_pool),
            hidden_dim=int(args.hidden_dim),
            num_layers=int(args.num_layers),
            tau_ms=float(args.tau_ms),
            n_workers=int(args.n_workers),
            anchor_lbfgs_steps=int(args.anchor_steps),
            e2e_epochs=int(args.epochs),
            e2e_lr=float(args.e2e_lr),
            seed=int(args.seed),
            timeout_s=float(args.timeout_s),
            asset_root=args.asset_root,
            runtime_root=args.runtime_root,
            physics_python=args.physics_python,
        )
        summary = train_multijoint_rate_model(
            cfg_mj,
            progress_callback=lambda ev: print(json.dumps(ev, ensure_ascii=False), flush=True),
        )
        print(
            json.dumps(
                {
                    "checkpoint_path": summary["checkpoint_path"],
                    "architecture": summary.get("architecture"),
                    "best_epoch": summary["best_epoch"],
                    "best_val": summary["best_val"],
                    "train_eval": summary["train_eval"],
                    "elapsed_s": summary["elapsed_s"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "run-multijoint":
        eng = MultiJointEngine(
            MultiJointEngineConfig(
                backend=args.backend,
                profile=args.profile,
                checkpoint_path=args.checkpoint,
                asset_root=args.asset_root,
                runtime_root=args.runtime_root,
                physics_python=args.physics_python,
            )
        )
        try:
            if args.target_file is not None:
                arr = np.load(args.target_file, allow_pickle=False)
                targ = np.asarray(arr["target_rad"] if hasattr(arr, "files") else arr, dtype=np.float64)
            else:
                goal = eng.goal_from_family(family=args.goal_family, scale=float(args.scale))
                targ = np.asarray(goal["target_rad"], dtype=np.float64)
            delta_deg = (
                np.array([float(x) for x in args.init_delta_deg.split(",")], dtype=np.float64)
                if args.init_delta_deg
                else None
            )
            res = eng.run(
                targ,
                output=args.output,
                mode=args.mode,
                threshold_deg=float(args.threshold_deg),
                correction_steps=int(args.correction_steps),
                feedback_kp=float(args.feedback_kp),
                feedback_kd=float(args.feedback_kd),
                initial_qpos_delta_deg=delta_deg,
                render=bool(args.render),
                bio_mode=args.bio_mode,
                bio_seed=int(args.bio_seed),
                bio_physics_vjp_steps=int(args.bio_physics_vjp_steps),
                bio_population_scale=int(args.bio_population_scale),
            )
            print(json.dumps(res, indent=2, ensure_ascii=False))
            return 0
        finally:
            eng.close()

    if args.command == "bio-realize":
        cat = load_asset_catalog(
            asset_root=args.asset_root,
            runtime_root=args.runtime_root,
            physics_python=args.physics_python,
            verify_hashes=True,
        )
        with np.load(args.traces_npz, allow_pickle=False) as tf:
            wf = np.asarray(tf["command_pool_rates_hz"], dtype=np.float64)
            exc = np.asarray(
                tf["feedforward_excitation"] if "feedforward_excitation" in tf else tf["excitation"],
                dtype=np.float64,
            )
            targ = np.asarray(tf["target_rad"], dtype=np.float64)
            default_tau = 20.0 if args.scope == "tibia" else 2.0
            tau_ms = (
                float(args.tau_bridge_ms)
                if args.tau_bridge_ms is not None
                else (float(tf["bridge_tau_ms"]) if "bridge_tau_ms" in tf else default_tau)
            )
        hashes = cat.three_hash_record(args.profile)
        spec = SessionSpec(
            profile=args.profile,
            asset_manifest_sha256=cat.manifest_sha256,
            teacher_sha256=cat.teacher_sha256,
            raw_xml_sha256=cat.raw_xml_sha256,
            profile_rule_sha256=hashes["profile_rule_sha256"],
            cache_mode="fake" if args.backend == "fake" else "cached_model",
        )
        if args.backend == "fake":
            backend = FakeBackend(profile=args.profile)
        else:
            cache_mode = "cached_model" if args.backend == "persistent_cached" else "fresh"
            backend = PersistentMujocoBackend(cat, profile=args.profile, cache_mode=cache_mode)
        sess = backend.open_session(spec)
        try:
            backend.reset(sess, InitialState(0.0), seed=int(args.seed))
            res = evaluate_bio_realization(
                backend=backend,  # type: ignore[arg-type]
                session=sess,
                catalog=cat,
                scope=args.scope,
                command_rates_hz=wf,
                target_excitation=exc,
                target_rad=targ,
                bio_mode=args.bio_mode,
                seed=int(args.seed),
                tau_bridge_ms=tau_ms,
                population_scale=int(args.population_scale),
                physics_vjp_steps=int(args.physics_vjp_steps),
                output_dir=args.output,
            )
            print(json.dumps(res, indent=2, ensure_ascii=False))
            return 0
        finally:
            backend.close_session(sess)
            backend.close()

    return 2  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
