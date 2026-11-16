"""Repeat a completed unseen-family trial with forward-simulation pattern search."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from pathlib import Path

from benchmark_v2_unseen_families import METHODS, summarize, write_summary


def run_shard(asset_root, output, jobs, protocol, cpu):
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})

    import torch
    from sensory_translator_v2.engine import MultiJointEngine, MultiJointEngineConfig
    from sensory_translator_v2.evidence import write_json_atomic
    from benchmark_v2_unseen_pattern import UnseenPatternSolver

    torch.set_num_threads(1)
    trials = []
    with tempfile.TemporaryDirectory(prefix="sensory-forward-benchmark-") as runtime:
        config = MultiJointEngineConfig(asset_root=asset_root, runtime_root=runtime, profile=protocol["profile"])
        with closing(MultiJointEngine(config)) as engine:
            if engine.checkpoint_sha256() != protocol["checkpoint_sha256"]:
                raise RuntimeError("The worker did not load the original frozen checkpoint")
            solver = UnseenPatternSolver(engine)
            for index, name, family, target, delta in jobs:
                order = METHODS if index % 2 == 0 else METHODS[::-1]
                for position, method in enumerate(order):
                    trial = solver.solve(
                        target, initial_delta_rad=delta, method=method,
                        budget_s=protocol["budget_s"], threshold_deg=protocol["threshold_deg"], seed=protocol["seed"],
                    )
                    trial.update(case=name, family=family, method=method, order=position, cpu=cpu, worker_warmup=solver.warmup)
                    write_json_atomic(output / "trials" / f"{name}_{method}.json", trial)
                    trials.append(trial)
                    print(f"{name} {method}: {trial['best_rmse_deg']} deg, {trial['stop_reason']}", flush=True)
    return trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=Path.cwd())
    parser.add_argument("--cpus", type=int, nargs="+", help="One worker per specified Linux logical CPU")
    args = parser.parse_args()
    cpus = args.cpus or [None]
    if args.cpus is not None and (
        len(set(cpus)) != len(cpus) or not set(cpus) <= os.sched_getaffinity(0)
    ):
        parser.error("--cpus must contain distinct available CPUs")

    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    os.environ["MUJOCO_GL"] = "disable"
    import numpy as np
    from sensory_translator_v2.assets import load_asset_catalog, sha256_file
    from sensory_translator_v2.evidence import write_json_atomic
    from benchmark_v2_unseen_pattern import INITIAL_STEP_HZ, MINIMUM_STEP_HZ

    source = args.source.resolve()
    output = args.output.resolve()
    asset_root = args.asset_root.resolve()
    protocol = json.loads((source / "protocol.json").read_text())
    manifest = json.loads((source / "target_manifest.json").read_text())
    if not (source / "summary.json").is_file():
        raise RuntimeError("Expected a completed original paired run")
    target_file = source / "targets.npz"
    if sha256_file(target_file) != manifest["sha256"]:
        raise RuntimeError("Original targets do not match the saved checksum")
    catalog = load_asset_catalog(asset_root=asset_root)
    checkpoint = catalog.resolve_multijoint_model()
    if checkpoint is None or sha256_file(checkpoint.checkpoint_path) != protocol["checkpoint_sha256"]:
        raise RuntimeError("The checkpoint does not match the original paired run")
    if catalog.three_hash_record(protocol["profile"]) != protocol["physics_hashes"]:
        raise RuntimeError("The physics assets do not match the original paired run")
    with np.load(target_file, allow_pickle=False) as data:
        jobs = [
            (index, str(name), str(family), target, delta)
            for index, (name, family, target, delta) in enumerate(
                zip(data["names"], data["families"], data["target_rad"], data["initial_delta_rad"], strict=True)
            )
        ]
    protocol.update(
        original_protocol_sha256=sha256_file(source / "protocol.json"),
        target_file=str(target_file),
        target_sha256=manifest["sha256"],
        cpu_affinities=cpus,
        optimizer={
            "name": "projected smooth pattern search",
            "initial_step_hz": INITIAL_STEP_HZ,
            "minimum_step_hz": MINIMUM_STEP_HZ,
            "direction_basis": "three cosine modes crossed with four global and fifteen per-muscle directions",
            "step_rule": "halve after a sweep with no improvement",
            "derivatives": False,
        },
        experiment="Supplemental forward-simulation search after the checked VJP rejected gradients in preliminary trials",
        sources={
            path.name: sha256_file(path)
            for path in sorted(Path(__file__).resolve().parent.glob("benchmark_v2_unseen_*.py"))
        },
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "trials").mkdir()
    write_json_atomic(output / "protocol.json", protocol)
    if len(cpus) == 1:
        rows = run_shard(asset_root, output, jobs, protocol, cpus[0])
    else:
        with ProcessPoolExecutor(max_workers=len(cpus), mp_context=multiprocessing.get_context("spawn")) as workers:
            futures = [
                workers.submit(run_shard, asset_root, output, jobs[index::len(cpus)], protocol, cpu)
                for index, cpu in enumerate(cpus)
            ]
            rows = [trial for future in futures for trial in future.result()]
    if any(row["best_rmse_deg"] is None for row in rows):
        raise RuntimeError("At least one trial has no scored rollout. See its trial JSON.")
    if sha256_file(checkpoint.checkpoint_path) != protocol["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed during the benchmark")
    order = {name: index for index, name, *_ in jobs}
    rows.sort(key=lambda row: (order[row["case"]], row["method"]))
    summary = summarize(rows, protocol["joint_names"])
    write_json_atomic(output / "summary.json", summary)
    write_summary(output, protocol, summary, rows)
    print(f"Saved {len(rows)} trials to {output}")


if __name__ == "__main__":
    main()
