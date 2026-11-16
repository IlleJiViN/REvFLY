"""Compare frozen ResidualNorm and common-waveform starts on unseen motion families."""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing
import os
import platform
import statistics
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from pathlib import Path


PROFILE = "adjacent_static_parent_v1"
SEED = 20261116
THRESHOLD_DEG = math.degrees(0.1)
METHODS = ("direct", "model")


def run_shard(asset_root, output, jobs, budget_s, cpu):
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})

    import torch
    from sensory_translator_v2.engine import MultiJointEngine, MultiJointEngineConfig
    from sensory_translator_v2.evidence import write_json_atomic
    from benchmark_v2_unseen_solver import UnseenBenchmarkSolver

    torch.set_num_threads(1)
    rows = []
    with tempfile.TemporaryDirectory(prefix="sensory-v2-unseen-") as runtime:
        config = MultiJointEngineConfig(asset_root=asset_root, runtime_root=runtime, profile=PROFILE)
        with closing(MultiJointEngine(config)) as engine:
            solver = UnseenBenchmarkSolver(engine)
            for index, name, family, target_rad, initial_delta_rad in jobs:
                order = METHODS if index % 2 == 0 else METHODS[::-1]
                for position, method in enumerate(order):
                    trial = solver.solve(
                        target_rad,
                        method=method,
                        budget_s=budget_s,
                        threshold_deg=THRESHOLD_DEG,
                        initial_delta_rad=initial_delta_rad,
                        seed=SEED,
                    )
                    trial.update(case=name, family=family, method=method, order=position, cpu=cpu, worker_warmup=solver.warmup)
                    write_json_atomic(output / "trials" / f"{name}_{method}.json", trial)
                    rows.append(trial)
                    print(f"{name} {method}: {trial['best_rmse_deg']} deg, {trial['stop_reason']}", flush=True)
    return rows


def summarize(rows, joint_names):
    groups = {}
    for family in ("all", *sorted({row["family"] for row in rows})):
        by_method = {}
        for method in METHODS:
            trials = [row for row in rows if row["method"] == method and (family == "all" or row["family"] == family)]
            failures = sum(not row["accuracy_threshold_met"] for row in trials)
            by_method[method] = {
                "n": len(trials),
                "failures": failures,
                "failure_rate": failures / len(trials),
                "initial_mean_rmse_deg": statistics.mean(row["initial_rmse_deg"] for row in trials),
                "final_mean_rmse_deg": statistics.mean(row["best_rmse_deg"] for row in trials),
                "worst_case_rmse_deg": max(row["best_rmse_deg"] for row in trials),
                "median_search_s": statistics.median(row["search_elapsed_s"] for row in trials),
                "max_search_overshoot_s": max(row["search_overshoot_s"] for row in trials),
                "total_optimizer_updates": sum(row["optimizer_updates"] for row in trials),
                "worst_joint_rmse_deg": {
                    joint: max(row["per_joint_rmse_deg"][joint] for row in trials) for joint in joint_names
                },
                "worst_joint_peak_abs_deg": {
                    joint: max(row["per_joint_max_abs_error_deg"][joint] for row in trials) for joint in joint_names
                },
            }
        groups[family] = by_method
    paired = {}
    for row in rows:
        paired.setdefault(row["case"], {})[row["method"]] = row["best_rmse_deg"]
    return {
        "groups": groups,
        "model_wins": sum(pair["model"] < pair["direct"] for pair in paired.values()),
        "direct_wins": sum(pair["direct"] < pair["model"] for pair in paired.values()),
        "exact_ties": sum(pair["model"] == pair["direct"] for pair in paired.values()),
        "all_replays_verified": all(row["independent_replay_verified"] for row in rows),
    }


def write_summary(output, protocol, summary, rows):
    lines = [
        "# 미사용 동작 계열 역제어 비교",
        "",
        f"모델 `{protocol['model_id']}`를 고정했다. 동작 계열 3개, 합성 목표 9개를 같은 MuJoCo 물리에서 생성했다.",
        f"각 방법의 탐색 예산은 {protocol['budget_s']:g}초다. 기준은 7축 전체 RMSE {THRESHOLD_DEG:.4f}° 이하다.",
        "",
        "| 계열 | 방법 | 초기 평균 RMSE | 최종 평균 RMSE | 가장 큰 케이스 RMSE | 실패율 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for family, group in summary["groups"].items():
        for method, values in group.items():
            lines.append(
                f"| {family} | {method} | {values['initial_mean_rmse_deg']:.3f}° | "
                f"{values['final_mean_rmse_deg']:.3f}° | {values['worst_case_rmse_deg']:.3f}° | "
                f"{values['failures']}/{values['n']} ({values['failure_rate']:.1%}) |"
            )
    lines += [
        "",
        "| 관절축 | 직접 최대 RMSE | 모델 최대 RMSE | 직접 최대 절대오차 | 모델 최대 절대오차 |",
        "|---|---:|---:|---:|---:|",
    ]
    direct, model = (summary["groups"]["all"][method] for method in METHODS)
    for joint in protocol["joint_names"]:
        lines.append(
            f"| {joint} | {direct['worst_joint_rmse_deg'][joint]:.3f}° | {model['worst_joint_rmse_deg'][joint]:.3f}° | "
            f"{direct['worst_joint_peak_abs_deg'][joint]:.3f}° | {model['worst_joint_peak_abs_deg'][joint]:.3f}° |"
        )
    lines += [
        "",
        "`direct`는 두 방법이 공유하는 목표 비의존 기본 발화율 파형에서, `model`은 목표 조건부 파형에서 시작한다.",
        "목표를 만든 근육 구동과 발화율 매듭은 `targets.npz`에 저장했다. 두 방법에는 목표 관절각과 초기 자세만 제공했다.",
        "아래 수치는 같은 목표에 대한 쌍별 비교다. 독립 재생은 탐색 시간 이후에 수행하고 별도 기록했다.",
        "",
        "| 목표 | 직접 RMSE | 모델 RMSE |",
        "|---|---:|---:|",
    ]
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case"], {})[row["method"]] = row
    for name, pair in by_case.items():
        lines.append(f"| {name} | {pair['direct']['best_rmse_deg']:.3f}° | {pair['model']['best_rmse_deg']:.3f}° |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["case", "family", "method", "initial_rmse_deg", "best_rmse_deg", "accuracy_threshold_met"]
        fields += ["search_elapsed_s", "search_overshoot_s", "time_to_first_pass_s", "independent_replay_elapsed_s", "optimizer_updates", "stop_reason"]
        fields += [f"{metric}.{joint}" for metric in ("per_joint_rmse_deg", "per_joint_max_abs_error_deg") for joint in protocol["joint_names"]]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for metric in ("per_joint_rmse_deg", "per_joint_max_abs_error_deg"):
                flat.update({f"{metric}.{joint}": value for joint, value in row[metric].items()})
            writer.writerow({field: flat.get(field) for field in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-s", type=float, default=30.0)
    parser.add_argument("--cpus", type=int, nargs="+", help="One worker per specified Linux logical CPU")
    args = parser.parse_args()
    if not math.isfinite(args.budget_s) or args.budget_s <= 0:
        parser.error("--budget-s must be positive and finite")
    if args.cpus is not None and (
        len(set(args.cpus)) != len(args.cpus) or not set(args.cpus) <= os.sched_getaffinity(0)
    ):
        parser.error("--cpus must contain distinct available CPUs")

    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    os.environ["MUJOCO_GL"] = "disable"
    import numpy as np
    import torch
    from sensory_translator_v2.assets import MULTIJOINT_CHECKPOINT_SHA256, load_asset_catalog, sha256_file
    from sensory_translator_v2.contracts import CONTROL_TIMESTEP_S, LF_JOINT_NAMES
    from sensory_translator_v2.evidence import write_json_atomic, write_npz_atomic
    from sensory_translator_v2.rate.model import load_rate_checkpoint
    from benchmark_v2_unseen_solver import INITIAL_STEP_RMS_HZ, LINE_SEARCH_FACTORS, MAX_STEP_RMS_HZ
    from benchmark_v2_unseen_targets import FAMILY_DESCRIPTIONS, TRAINING_FAMILIES, generate_targets

    torch.set_num_threads(1)
    output = args.output.resolve()
    asset_root = args.asset_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "trials").mkdir()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    cpus = args.cpus or [None]
    with tempfile.TemporaryDirectory(prefix="sensory-v2-targets-") as runtime:
        catalog = load_asset_catalog(asset_root=asset_root, runtime_root=runtime)
        checkpoint = catalog.resolve_multijoint_model()
        if checkpoint is None or checkpoint.checkpoint_sha256 != MULTIJOINT_CHECKPOINT_SHA256:
            raise ValueError("The registered ResidualNorm checkpoint is required")
        model, config, _ = load_rate_checkpoint(checkpoint.checkpoint_path, expected_sha256=checkpoint.checkpoint_sha256)
        bridge = config["bridge"]
        parameterization = {
            "knots_per_pool": model.knots_per_pool,
            "max_rate_hz": model.max_rate_hz,
            "rate_scale_hz": bridge["rate_scale_hz"],
            "tau_ms": bridge["tau_ms"],
            "control_bin_ms": round(CONTROL_TIMESTEP_S * 1000),
        }
        del model
        protocol = {
            "revision": revision,
            "model_id": checkpoint.model_id,
            "checkpoint_sha256": sha256_file(checkpoint.checkpoint_path),
            "model_frozen": True,
            "budget_s": args.budget_s,
            "threshold_deg": THRESHOLD_DEG,
            "seed": SEED,
            "profile": PROFILE,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "physics_python": str(catalog.physics_python),
            "physics_hashes": catalog.three_hash_record(PROFILE),
            "warm_workers": True,
            "cpu_affinities": cpus,
            "trial_order": "counterbalanced within each target, methods run serially on the same CPU",
            "joint_names": list(LF_JOINT_NAMES),
            "training_families": TRAINING_FAMILIES,
            "test_families": FAMILY_DESCRIPTIONS,
            "parameterization": parameterization,
            "optimizer": {
                "name": "projected L-BFGS",
                "initial_step_rms_hz": INITIAL_STEP_RMS_HZ,
                "max_step_rms_hz": MAX_STEP_RMS_HZ,
                "line_search_factors": LINE_SEARCH_FACTORS,
                "history": 5,
                "check_directions": True,
            },
            "method": {
                "direct": "checkpoint's common, target-independent base_knots_hz + shared physical refinement",
                "model": "frozen ResidualNorm goal-conditioned initial knots + shared physical refinement",
            },
            "verification": "independent physics replay after the search, measured separately",
            "sources": {
                path.name: sha256_file(path)
                for path in sorted(Path(__file__).resolve().parent.glob("benchmark_v2_unseen_*.py"))
            },
        }
        write_json_atomic(output / "protocol.json", protocol)
        targets = generate_targets(catalog, profile=PROFILE, seed=SEED, **parameterization)
    if len(targets) != 9 or len({target.name for target in targets}) != 9 or len({target.family for target in targets}) != 3:
        raise RuntimeError("Expected nine distinct targets from three frozen families")
    write_npz_atomic(output / "targets.npz", {
        "names": np.array([target.name for target in targets]),
        "families": np.array([target.family for target in targets]),
        "target_rad": np.stack([target.target_rad for target in targets]),
        "initial_delta_rad": np.stack([target.initial_delta_rad for target in targets]),
        "witness_controls": np.stack([target.witness_controls for target in targets]),
        "witness_knots_hz": np.stack([target.witness_knots_hz for target in targets]),
    })
    write_json_atomic(output / "target_manifest.json", {
        "sha256": sha256_file(output / "targets.npz"),
        "cases": [{"name": target.name, "family": target.family, "parameters": target.parameters} for target in targets],
    })

    jobs = [(index, target.name, target.family, target.target_rad, target.initial_delta_rad) for index, target in enumerate(targets)]
    if len(cpus) == 1:
        rows = run_shard(asset_root, output, jobs, args.budget_s, cpus[0])
    else:
        with ProcessPoolExecutor(max_workers=len(cpus), mp_context=multiprocessing.get_context("spawn")) as workers:
            batches = [
                workers.submit(run_shard, asset_root, output, jobs[index::len(cpus)], args.budget_s, cpu)
                for index, cpu in enumerate(cpus)
            ]
            rows = [trial for batch in batches for trial in batch.result()]
    case_order = {target.name: index for index, target in enumerate(targets)}
    rows.sort(key=lambda row: (case_order[row["case"]], row["method"]))
    if sha256_file(checkpoint.checkpoint_path) != protocol["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed during the benchmark")
    if any(row["best_rmse_deg"] is None for row in rows):
        raise RuntimeError("At least one trial has no scored rollout. See its trial JSON.")
    summary = summarize(rows, LF_JOINT_NAMES)
    write_json_atomic(output / "summary.json", summary)
    write_summary(output, protocol, summary, rows)
    print(f"Saved {len(rows)} trials to {output}")


if __name__ == "__main__":
    main()
