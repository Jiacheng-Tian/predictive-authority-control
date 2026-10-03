#!/usr/bin/env python3
"""SSPO alpha-bias calibration on the v4 runner (non-learning control arm).

Coordinate descent over the four archived authority windows, mirroring the
v2-era procedure: for each window, try every bias in the grid while the
others stay fixed, score by rolling out all search episodes, and keep the
lexicographic best (score, then smaller |bias|).  The objective is

    score = mean rmse_3d
          + window_regret_weight * mean positive window regret vs best fixed
          + rmse_guard_weight * max(0, mean rmse_3d - backbone mean rmse_3d)

with fixed-controller and backbone references computed on the same search
seeds first.  Search seeds (44000-44004 role) are disjoint from every
evaluation partition.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pac.evaluation.diagnostics import compute_axis_window_metrics  # noqa: E402
from pac.residual.config import load_residual_config  # noqa: E402
from pac.residual.disturbances import STRUCTURED_FAMILY, build_paired_episode_spec  # noqa: E402
from pac.residual.eval.formal import build_episode_tasks  # noqa: E402
from pac.residual.eval.runner import (  # noqa: E402
    FixedControllerPolicy,
    build_backbone_policy,
    run_residual_policy_episode,
)

WINDOW_NAMES = ("startup_0_3s", "pre_step_3_10s",
                "step_recovery_10_13s", "post_step_13_20p9s")


def _episode_plan(config, seed: int) -> list[dict]:
    return [
        {"family": STRUCTURED_FAMILY, "scenario_id": scenario, "seed": int(seed)}
        for scenario in (1, 2, 3)
    ]


def _run_episode(arguments):
    config_path, method, scenario, seed, schedule_json = arguments
    config = load_residual_config(config_path)
    if method in ("smc", "mpc"):
        policy = FixedControllerPolicy(
            "primary" if method == "smc" else "authority"
        )
    else:
        schedule = json.loads(schedule_json) if schedule_json else None
        policy = build_backbone_policy(
            config, config.seeds.model[0], alpha_bias_schedule=schedule
        )
    spec = build_paired_episode_spec(
        STRUCTURED_FAMILY, scenario, seed,
        int(config.environment.steps), float(config.environment.dt),
    )
    metrics = run_residual_policy_episode(policy, spec, config)
    windows = compute_axis_window_metrics(
        metrics["ts"], dt=float(config.environment.dt)
    )
    return {
        "method": method,
        "scenario_id": scenario,
        "seed": int(seed),
        "rmse_3d": float(metrics["rmse_3d"]),
        "windows": windows[["window", "rmse_3d"]].to_dict(orient="records"),
    }


def _window_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        for window in result["windows"]:
            rows.append({
                "method": result["method"],
                "scenario_id": result["scenario_id"],
                "seed": result["seed"],
                "window": window["window"],
                "rmse_3d": float(window["rmse_3d"]),
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "pac_residual.yaml"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--progress-every", type=int, default=1)
    args = parser.parse_args(argv)

    config_path = str(Path(args.config).resolve())
    config = load_residual_config(config_path)
    output = Path(args.out_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    seeds = [int(seed) for seed in config.sspo.search_seeds[:3]]
    plan = [
        (seed, scenario)
        for seed in seeds
        for scenario in (1, 2, 3)
    ]
    jobs = max(1, int(args.jobs))
    context = mp.get_context("spawn")
    started = time.perf_counter()

    def evaluate(method: str, schedule: dict | None) -> list[dict]:
        schedule_json = json.dumps(schedule, sort_keys=True) if schedule else ""
        tasks = [
            (config_path, method, scenario, seed, schedule_json)
            for seed, scenario in plan
        ]
        with context.Pool(processes=min(jobs, len(tasks))) as pool:
            return pool.map(_run_episode, tasks)

    # references: fixed controllers and the frozen backbone
    fixed_results = {"smc": evaluate("smc", None), "mpc": evaluate("mpc", None)}
    backbone_results = evaluate("backbone", None)
    fixed_table = _window_table(fixed_results["smc"] + fixed_results["mpc"])
    best_fixed = fixed_table.groupby(["scenario_id", "seed", "window"])[
        "rmse_3d"
    ].min().rename("best_fixed")
    backbone_rmse = float(np.mean([
        result["rmse_3d"] for result in backbone_results
    ]))
    window_regret_weight = 0.02
    rmse_guard_weight = 5.0

    def score(schedule: dict) -> float:
        results = evaluate("backbone", schedule)
        mean_rmse = float(np.mean([result["rmse_3d"] for result in results]))
        table = _window_table(results).merge(
            best_fixed, on=["scenario_id", "seed", "window"]
        )
        positive_regret = float(np.mean(
            np.maximum(table["rmse_3d"] - table["best_fixed"], 0.0)
            / table["best_fixed"].clip(lower=1e-9)
        ))
        return (
            mean_rmse
            + window_regret_weight * positive_regret
            + rmse_guard_weight * max(0.0, mean_rmse - backbone_rmse)
        )

    bias_grid = [float(value) for value in config.sspo.bias_grid]
    schedule = {name: 0.0 for name in WINDOW_NAMES}
    trials = []
    for iteration in range(int(config.sspo.iterations)):
        for window in WINDOW_NAMES:
            best = (score(schedule), schedule[window])
            for bias in bias_grid:
                if bias == schedule[window]:
                    continue
                candidate = dict(schedule)
                candidate[window] = bias
                value = score(candidate)
                trials.append({
                    "iteration": iteration,
                    "window": window,
                    "bias": bias,
                    "score": value,
                    "schedule_json": json.dumps(candidate, sort_keys=True),
                })
                if value < best[0] - 1e-12:
                    best = (value, bias)
            schedule[window] = best[1]
            print(
                f"[sspo] it={iteration} window={window} -> bias={best[1]:+.2f} "
                f"score={best[0]:.5f} elapsed={time.perf_counter()-started:.0f}s",
                file=sys.stderr, flush=True,
            )
    final_score = score(schedule)
    pd.DataFrame(trials).to_csv(output / "sspo_bias_search.csv", index=False)
    (output / "best_bias_schedule.json").write_text(
        json.dumps(schedule, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "sspo_summary.json").write_text(
        json.dumps({
            "search_seeds": seeds,
            "bias_grid": bias_grid,
            "iterations": int(config.sspo.iterations),
            "window_regret_weight": window_regret_weight,
            "rmse_guard_weight": rmse_guard_weight,
            "backbone_mean_rmse": backbone_rmse,
            "best_schedule": schedule,
            "best_score": final_score,
            "elapsed_s": time.perf_counter() - started,
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"schedule": schedule, "score": final_score}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
