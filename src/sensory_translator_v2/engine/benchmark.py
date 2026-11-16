"""Matched cold, warm, and HTTP end-to-end benchmark runner for Engine v2."""

from __future__ import annotations

import http.client
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from ..assets import sha256_file
from ..evidence import utc_now_iso, write_json_atomic
from .core import DEFAULT_THRESHOLD_DEG, EngineV2, EngineV2Config
from .server import RateControlV2App, RateControlV2HTTPServer

REGRESSION_SCALES: tuple[float, ...] = (0.90625, 0.98125, 1.01875, 1.09375)


def _summarize_times(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return {"count": 0}
    return {
        "count": len(ordered),
        "median_s": float(statistics.median(ordered)),
        "mean_s": float(statistics.mean(ordered)),
        "min_s": float(ordered[0]),
        "max_s": float(ordered[-1]),
    }


def benchmark_backend(
    backend_name: str,
    *,
    profile: str = "adjacent_static_parent_v1",
    model_id: str = "20261114_v6_longrun",
    asset_root: str | Path | None = None,
    runtime_root: str | Path | None = None,
    physics_python: str | Path | None = None,
    scales: Sequence[float] = REGRESSION_SCALES,
    mode: str = "auto",
    initializer: str = "mlp",
    threshold_deg: float = DEFAULT_THRESHOLD_DEG,
    correction_steps: int = 6,
    correction_lr_hz: float = 0.05,
    budget_s: float = 90.0,
    include_http: bool = True,
) -> dict[str, Any]:
    """Benchmark one backend separating cold startup, warm trials, and HTTP end-to-end."""
    cfg = EngineV2Config(
        backend=backend_name,
        profile=profile,
        model_id=model_id,
        asset_root=asset_root,
        runtime_root=runtime_root,
        physics_python=physics_python,
        timeout_s=budget_s,
    )

    # 1. Cold measurement: fresh instance + asset verify + worker startup + first rollout
    cold_t0 = time.perf_counter()
    cold_engine = EngineV2(cfg)
    try:
        caps = cold_engine.capabilities()
        cold_caps_s = time.perf_counter() - cold_t0
        first_scale = float(scales[0])
        goal0 = cold_engine.goal_from_scale(first_scale)
        run0_t0 = time.perf_counter()
        res0 = cold_engine.run(
            goal0["target_rad"],
            mode=mode,
            initializer=initializer,
            threshold_deg=threshold_deg,
            correction_steps=correction_steps,
            correction_lr_hz=correction_lr_hz,
        )
        cold_first_run_s = time.perf_counter() - run0_t0
        cold_total_s = time.perf_counter() - cold_t0
        cold_record = {
            "capabilities_plus_handshake_s": cold_caps_s,
            "first_run_s": cold_first_run_s,
            "total_cold_s": cold_total_s,
            "goal_scale": first_scale,
            "initializer": initializer,
            "best_rmse_deg": res0["best_rmse_deg"],
            "accuracy_threshold_met": res0["accuracy_threshold_met"],
            "correction_steps_taken": res0["correction_steps_taken"],
            "timing_s": res0["timing_s"],
            "is_certified_frozen_physics": res0["is_certified_frozen_physics"],
        }
    finally:
        cold_engine.close()

    # 2. Warm SDK measurement (separate warmed EngineV2 instance, warmup goal before measuring)
    warm_engine = EngineV2(cfg)
    warm_trials: list[dict[str, Any]] = []
    try:
        warm_engine.capabilities()
        if hasattr(warm_engine.backend, "ensure_ready"):
            warm_engine.backend.ensure_ready()
        # Explicit warmup rollout excluded from warm distribution.
        warmup_goal = warm_engine.goal_from_scale(float(scales[0]))
        warm_engine.run(
            warmup_goal["target_rad"],
            mode="fast",
            initializer=initializer,
            threshold_deg=threshold_deg,
        )

        for scale in scales:
            goal = warm_engine.goal_from_scale(float(scale))
            t0 = time.perf_counter()
            res = warm_engine.run(
                goal["target_rad"],
                mode=mode,
                initializer=initializer,
                threshold_deg=threshold_deg,
                correction_steps=correction_steps,
                correction_lr_hz=correction_lr_hz,
            )
            dt = time.perf_counter() - t0
            warm_trials.append(
                {
                    "goal_scale": float(scale),
                    "wall_s": dt,
                    "budget_met": bool(dt <= budget_s),
                    "best_rmse_deg": float(res["best_rmse_deg"]),
                    "nominal_rmse_deg": float(res["nominal_rmse_deg"]),
                    "accuracy_threshold_met": bool(res["accuracy_threshold_met"]),
                    "correction_steps_taken": int(res["correction_steps_taken"]),
                    "independent_replay_verified": bool(res["independent_replay_verified"]),
                    "timing_s": res["timing_s"],
                }
            )
    finally:
        warm_engine.close()

    # 3. Warm HTTP end-to-end measurement (client POST /v2/jobs -> poll GET until done)
    http_summary: dict[str, Any] | None = None
    if include_http and initializer == "mlp":
        http_engine = EngineV2(cfg)
        app = RateControlV2App(controller=http_engine, ui_enabled=False)
        server = RateControlV2HTTPServer(("127.0.0.1", 0), app=app)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        http_trials: list[dict[str, Any]] = []
        try:
            # Ensure capabilities & worker warm before measuring warm HTTP E2E
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=budget_s + 5.0)
            conn.request("GET", "/v2/capabilities")
            r_cap = conn.getresponse()
            r_cap.read()
            if hasattr(http_engine.backend, "ensure_ready"):
                http_engine.backend.ensure_ready()

            for scale in scales:
                t0 = time.perf_counter()
                body = json.dumps(
                    {
                        "goal_scale": float(scale),
                        "mode": mode,
                        "threshold_deg": threshold_deg,
                        "correction_steps": correction_steps,
                        "correction_lr_hz": correction_lr_hz,
                        "render": False,
                    }
                ).encode("utf-8")
                conn.request(
                    "POST",
                    "/v2/jobs",
                    body=body,
                    headers={
                        "Host": f"127.0.0.1:{port}",
                        "Content-Type": "application/json",
                    },
                )
                post_resp = conn.getresponse()
                post_data = json.loads(post_resp.read())
                job_id = post_data["job_id"]
                final_state: dict[str, Any] = {}
                while True:
                    conn.request("GET", f"/v2/jobs/{job_id}")
                    poll_resp = conn.getresponse()
                    final_state = json.loads(poll_resp.read())
                    if final_state["status"] in ("completed", "failed"):
                        break
                    time.sleep(0.005)
                dt = time.perf_counter() - t0
                if final_state["status"] != "completed":
                    raise RuntimeError(
                        f"HTTP job failed for scale={scale}: {final_state.get('error')}"
                    )
                res_obj = final_state["result"]
                http_trials.append(
                    {
                        "goal_scale": float(scale),
                        "wall_s": dt,
                        "best_rmse_deg": float(res_obj["best_rmse_deg"]),
                        "accuracy_threshold_met": bool(res_obj["accuracy_threshold_met"]),
                    }
                )
            conn.close()
            http_summary = {
                "trials": http_trials,
                "summary": _summarize_times([t["wall_s"] for t in http_trials]),
            }
        finally:
            server.shutdown()
            server.server_close()

    return {
        "backend": backend_name,
        "profile": profile,
        "model_id": model_id,
        "mode": mode,
        "capabilities_backend": caps.get("backend"),
        "cold": cold_record,
        "warm_sdk": {
            "trials": warm_trials,
            "wall_summary": _summarize_times([t["wall_s"] for t in warm_trials]),
            "forward_summary": _summarize_times(
                [t["timing_s"]["physics_forward_s"] for t in warm_trials]
            ),
            "vjp_summary": _summarize_times(
                [t["timing_s"]["physics_vjp_s"] for t in warm_trials]
            ),
            "all_threshold_met": all(t["accuracy_threshold_met"] for t in warm_trials),
            "all_budget_met": all(t["budget_met"] for t in warm_trials),
        },
        "warm_http": http_summary,
    }


def run_matched_suite(
    *,
    backends: Sequence[str] = ("v1_subprocess", "persistent_fresh", "persistent_cached"),
    asset_root: str | Path | None = None,
    runtime_root: str | Path | None = None,
    physics_python: str | Path | None = None,
    output_json: str | Path | None = None,
    mode: str = "auto",
    correction_steps: int = 6,
    include_http: bool = True,
    include_fixed_nominal: bool = True,
) -> dict[str, Any]:
    """Execute matched cross-backend evaluation and compute empirical speedup ratios."""
    results: dict[str, Any] = {}
    for name in backends:
        results[name] = benchmark_backend(
            name,
            asset_root=asset_root,
            runtime_root=runtime_root,
            physics_python=physics_python,
            mode=mode,
            initializer="mlp",
            correction_steps=correction_steps,
            include_http=include_http,
        )

    fixed_vs_mlp: dict[str, Any] | None = None
    if include_fixed_nominal and "persistent_cached" in backends:
        fixed_rec = benchmark_backend(
            "persistent_cached",
            asset_root=asset_root,
            runtime_root=runtime_root,
            physics_python=physics_python,
            mode="auto",
            initializer="fixed_nominal",
            correction_steps=correction_steps,
            include_http=False,
        )
        mlp_rec = results["persistent_cached"]
        paired = []
        for f_t, m_t in zip(fixed_rec["warm_sdk"]["trials"], mlp_rec["warm_sdk"]["trials"]):
            paired.append(
                {
                    "goal_scale": f_t["goal_scale"],
                    "fixed_nominal_wall_s": f_t["wall_s"],
                    "fixed_nominal_rmse_deg": f_t["best_rmse_deg"],
                    "fixed_nominal_steps": f_t["correction_steps_taken"],
                    "mlp_wall_s": m_t["wall_s"],
                    "mlp_rmse_deg": m_t["best_rmse_deg"],
                    "mlp_steps": m_t["correction_steps_taken"],
                    "speedup_mlp_vs_fixed_x": f_t["wall_s"] / m_t["wall_s"],
                }
            )
        fixed_vs_mlp = {
            "backend": "persistent_cached",
            "fixed_nominal_warm_median_s": fixed_rec["warm_sdk"]["wall_summary"]["median_s"],
            "mlp_warm_median_s": mlp_rec["warm_sdk"]["wall_summary"]["median_s"],
            "median_speedup_mlp_vs_fixed_x": (
                fixed_rec["warm_sdk"]["wall_summary"]["median_s"]
                / mlp_rec["warm_sdk"]["wall_summary"]["median_s"]
            ),
            "fixed_nominal_all_threshold_met": fixed_rec["warm_sdk"]["all_threshold_met"],
            "mlp_all_threshold_met": mlp_rec["warm_sdk"]["all_threshold_met"],
            "paired_goals": paired,
        }

    comparisons: dict[str, Any] = {}
    if "v1_subprocess" in results:
        base_cold = results["v1_subprocess"]["cold"]["total_cold_s"]
        base_warm = results["v1_subprocess"]["warm_sdk"]["wall_summary"]["median_s"]
        for name, rec in results.items():
            if name == "v1_subprocess":
                continue
            cold_val = rec["cold"]["total_cold_s"]
            warm_val = rec["warm_sdk"]["wall_summary"]["median_s"]
            per_scale_ratios = []
            for b_trial, n_trial in zip(
                results["v1_subprocess"]["warm_sdk"]["trials"],
                rec["warm_sdk"]["trials"],
            ):
                per_scale_ratios.append(
                    {
                        "goal_scale": b_trial["goal_scale"],
                        "baseline_wall_s": b_trial["wall_s"],
                        "candidate_wall_s": n_trial["wall_s"],
                        "speedup_x": b_trial["wall_s"] / n_trial["wall_s"],
                        "rmse_diff_deg": abs(
                            b_trial["best_rmse_deg"] - n_trial["best_rmse_deg"]
                        ),
                    }
                )
            comparisons[f"v1_subprocess_vs_{name}"] = {
                "cold_speedup_x": base_cold / cold_val,
                "warm_median_speedup_x": base_warm / warm_val,
                "per_scale": per_scale_ratios,
            }

    payload = {
        "schema_version": "sensory_translator_v2_benchmark_v1",
        "created_utc": utc_now_iso(),
        "regression_scales": list(REGRESSION_SCALES),
        "mode": mode,
        "correction_steps": correction_steps,
        "backends": results,
        "comparisons": comparisons,
        "fixed_nominal_vs_mlp": fixed_vs_mlp,
    }
    if output_json is not None:
        write_json_atomic(output_json, payload)
    return payload
