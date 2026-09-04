"""Self-supervised performance optimization for pretrained PAC alpha models.

This experiment keeps the supervised predictive alpha model frozen and searches
small evaluation-window alpha biases using rollout performance. It is intended as the
first low-risk SSPO stage after supervised predictive initialization.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from evaluation.diagnose_3d_authority import (
    DEFAULT_WINDOWS,
    SCENARIO_NAMES,
    compute_axis_window_metrics,
)
from evaluation.predictive_3d_authority_alpha import (
    _scalar_metrics,
    load_alpha_model_checkpoint,
    run_predictive_alpha_episode,
)


DEFAULT_OUT_DIR = ROOT / "results" / "3d_authority_diagnosis" / "sspo_alpha_calibration_seed20_v1"
DEFAULT_MODEL = ROOT / "results" / "formal_seeded_v2" / "pac_train_seed_20" / "predictive_alpha_model.pt"
DEFAULT_FIXED_DIR = ROOT / "results" / "formal_seeded_v2" / "fixed_controllers"
DEFAULT_PAC_DIR = ROOT / "results" / "formal_seeded_v2" / "pac_train_seed_20"
DEFAULT_SCENARIOS = [1, 2, 3]
DEFAULT_SEARCH_SEEDS = [20000]
DEFAULT_EVAL_SEEDS = [20000 + idx for idx in range(10)]
DEFAULT_WINDOW_NAMES = [name for name, _start, _end in DEFAULT_WINDOWS]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_csv_ints(value: str) -> list[int]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError("integer list must contain at least one value")
    return [int(item) for item in items]


def parse_bias_grid(value: str) -> list[float]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError("bias grid must contain at least one value")
    return [float(item) for item in items]


def schedule_key(schedule: dict[str, float], windows: list[str]) -> tuple[float, ...]:
    return tuple(round(float(schedule.get(window, 0.0)), 10) for window in windows)


def mean_positive_window_regret(candidate: pd.DataFrame, fixed: pd.DataFrame) -> float:
    """Mean positive RMSE regret relative to the best fixed baseline per window."""
    required_candidate = {"scenario", "window", "rmse_3d"}
    required_fixed = {"scenario", "window", "controller", "rmse_3d"}
    missing_candidate = required_candidate - set(candidate.columns)
    missing_fixed = required_fixed - set(fixed.columns)
    if missing_candidate:
        raise ValueError(f"candidate window metrics missing columns: {sorted(missing_candidate)}")
    if missing_fixed:
        raise ValueError(f"fixed window metrics missing columns: {sorted(missing_fixed)}")
    if candidate.empty or fixed.empty:
        return 0.0

    keys = ["scenario", "window"]
    candidate_mean = (
        candidate.groupby(keys, as_index=False)["rmse_3d"]
        .mean()
        .rename(columns={"rmse_3d": "candidate_rmse"})
    )
    fixed_mean = fixed.groupby(keys + ["controller"], as_index=False)["rmse_3d"].mean()
    best_fixed = (
        fixed_mean.sort_values(keys + ["rmse_3d", "controller"])
        .groupby(keys, as_index=False)
        .first()
        .rename(columns={"rmse_3d": "best_fixed_rmse"})
    )
    merged = candidate_mean.merge(best_fixed[keys + ["best_fixed_rmse"]], on=keys, how="inner")
    if merged.empty:
        return 0.0
    denominator = np.maximum(np.abs(merged["best_fixed_rmse"].to_numpy(dtype=float)), 1.0e-12)
    regret = np.maximum(
        merged["candidate_rmse"].to_numpy(dtype=float)
        - merged["best_fixed_rmse"].to_numpy(dtype=float),
        0.0,
    ) / denominator
    return float(np.mean(regret))


def score_candidate(
        *,
        overall_rmse: float,
        positive_window_regret: float,
        baseline_rmse: float,
        window_regret_weight: float,
        rmse_guard_weight: float,
        saturation_step_fraction: float = 0.0,
        saturation_weight: float = 0.0,
        action_jerk_mean: float = 0.0,
        jerk_weight: float = 0.0) -> float:
    """Scalar objective for SSPO search.

    Overall RMSE is the primary target. The guard adds a strong penalty when a
    candidate exceeds the pretrained PAC RMSE on the same search episodes.
    """
    rmse = float(overall_rmse)
    guard_penalty = max(0.0, rmse - float(baseline_rmse))
    return float(
        rmse
        + float(rmse_guard_weight) * guard_penalty
        + float(window_regret_weight) * float(positive_window_regret)
        + float(saturation_weight) * float(saturation_step_fraction)
        + float(jerk_weight) * float(action_jerk_mean)
    )


def _filter_reference(
        table: pd.DataFrame,
        seeds: list[int],
        scenarios: list[int]) -> pd.DataFrame:
    out = table.copy()
    if "seed" in out.columns:
        out = out[out["seed"].isin(seeds)].copy()
    if "scenario_id" in out.columns:
        out = out[out["scenario_id"].isin(scenarios)].copy()
    return out


def _baseline_rmse(raw: pd.DataFrame) -> float:
    if raw.empty or "rmse" not in raw.columns:
        return float("inf")
    return float(raw["rmse"].mean())


def _attach_episode_metadata(window: pd.DataFrame, row: dict) -> pd.DataFrame:
    out = window.copy()
    existing = set(out.columns)
    for key, value in row.items():
        if not np.isscalar(value):
            continue
        column = key if key not in existing else f"episode_{key}"
        out[column] = value
    return out


def evaluate_bias_schedule(
        *,
        model,
        model_meta: dict,
        schedule: dict[str, float],
        scenarios: list[int],
        seeds: list[int],
        args,
        method: str,
        save_timeseries: bool = False,
        progress_label: str = "") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    windows = []
    timeseries_frames = []
    total = len(scenarios) * len(seeds)
    done = 0
    start = time.time()
    for scenario in scenarios:
        for seed in seeds:
            metrics = run_predictive_alpha_episode(
                model=model,
                scenario=int(scenario),
                seed=int(seed),
                steps=int(args.steps),
                mass_scale_xy=float(args.mass_scale_xy),
                damping_scale_xy=float(args.damping_scale_xy),
                current_amplitude_scale=float(args.current_amplitude_scale),
                current_frequency_scale=float(args.current_frequency_scale),
                vertical_current=float(args.vertical_current),
                primary_controller=str(args.primary_controller),
                authority_controller=str(args.authority_controller),
                feature_mode=str(model_meta["feature_mode"]),
                initial_position_std=float(args.eval_initial_position_std),
                initial_velocity_std=float(args.eval_initial_velocity_std),
                start_time=float(args.start_time),
                alpha_bias_schedule=schedule,
                alpha_gain=float(args.alpha_gain),
                alpha_threshold=float(args.alpha_threshold),
                alpha_mode="model",
                vehicle_profile=str(args.vehicle_profile),
                action_mode=str(args.action_mode),
                thruster_layout=str(args.thruster_layout),
                alpha_smoothing=float(args.alpha_smoothing),
                alpha_rate_limit=float(args.alpha_rate_limit),
                alpha_deadband=float(args.alpha_deadband),
                policy_architecture=str(model_meta["policy_architecture"]),
                history_len=int(model_meta["history_len"]),
                uncertainty_mode=str(args.uncertainty_mode),
                uncertainty_samples=int(args.uncertainty_samples),
                uncertainty_gain=float(args.uncertainty_gain),
                uncertainty_threshold=float(args.uncertainty_threshold),
                save_ts=True,
            )
            row = {
                "method": method,
                "scenario": SCENARIO_NAMES.get(int(scenario), str(scenario)),
                "scenario_id": int(scenario),
                "seed": int(seed),
                "start_time": float(args.start_time),
                "mass_scale_xy": float(args.mass_scale_xy),
                "damping_scale_xy": float(args.damping_scale_xy),
                "initial_position_std": float(args.eval_initial_position_std),
                "initial_velocity_std": float(args.eval_initial_velocity_std),
                **_scalar_metrics(metrics),
            }
            rows.append(row)
            ts = metrics["ts"].copy()
            ts["method"] = method
            ts["scenario"] = row["scenario"]
            ts["scenario_id"] = int(scenario)
            ts["seed"] = int(seed)
            ts["start_time"] = float(args.start_time)
            ts["current_amplitude_scale"] = float(args.current_amplitude_scale)
            ts["current_frequency_scale"] = float(args.current_frequency_scale)
            ts["initial_position_std"] = float(args.eval_initial_position_std)
            ts["initial_velocity_std"] = float(args.eval_initial_velocity_std)
            ts["relative_time"] = ts["step"].to_numpy(dtype=float) * 0.01
            if save_timeseries:
                timeseries_frames.append(ts)
            window = compute_axis_window_metrics(ts, dt=0.01)
            windows.append(_attach_episode_metadata(window, row))
            done += 1
            if args.progress_every and progress_label and (
                    done == 1
                    or done % int(args.progress_every) == 0
                    or done == total):
                print(
                    f"[{progress_label}] {done}/{total} scenario={scenario} seed={seed} "
                    f"elapsed_s={time.time() - start:.1f}",
                    flush=True,
                )
    raw = pd.DataFrame(rows)
    window_df = pd.concat(windows, ignore_index=True) if windows else pd.DataFrame()
    ts_df = pd.concat(timeseries_frames, ignore_index=True) if timeseries_frames else pd.DataFrame()
    return raw, window_df, ts_df


def summarize_candidate(
        raw: pd.DataFrame,
        window: pd.DataFrame,
        fixed_window: pd.DataFrame,
        baseline_rmse: float,
        args) -> dict:
    regret = mean_positive_window_regret(window, fixed_window)
    overall_rmse = float(raw["rmse"].mean())
    saturation = float(raw.get("action_saturation_step_fraction", pd.Series([0.0])).mean())
    jerk = float(raw.get("action_jerk_mean", pd.Series([0.0])).mean())
    score = score_candidate(
        overall_rmse=overall_rmse,
        positive_window_regret=regret,
        baseline_rmse=baseline_rmse,
        window_regret_weight=float(args.window_regret_weight),
        rmse_guard_weight=float(args.rmse_guard_weight),
        saturation_step_fraction=saturation,
        saturation_weight=float(args.saturation_weight),
        action_jerk_mean=jerk,
        jerk_weight=float(args.jerk_weight),
    )
    return {
        "score": score,
        "overall_rmse": overall_rmse,
        "baseline_rmse": float(baseline_rmse),
        "positive_window_regret": regret,
        "action_saturation_step_fraction": saturation,
        "action_jerk_mean": jerk,
    }


def coordinate_search(
        *,
        model,
        model_meta: dict,
        fixed_window: pd.DataFrame,
        baseline_raw: pd.DataFrame,
        args) -> tuple[dict[str, float], pd.DataFrame]:
    scenarios = parse_csv_ints(args.search_scenarios)
    seeds = parse_csv_ints(args.search_seeds)
    windows = [name.strip() for name in str(args.search_windows).split(",") if name.strip()]
    bias_grid = parse_bias_grid(args.bias_grid)
    schedule = {window: 0.0 for window in windows}
    baseline_rmse = _baseline_rmse(baseline_raw)
    cache: dict[tuple[float, ...], dict] = {}
    records = []

    def evaluate(schedule_candidate: dict[str, float]) -> dict:
        key = schedule_key(schedule_candidate, windows)
        if key not in cache:
            raw, window, _ts = evaluate_bias_schedule(
                model=model,
                model_meta=model_meta,
                schedule=schedule_candidate,
                scenarios=scenarios,
                seeds=seeds,
                args=args,
                method="predictive_alpha_sspo_search",
                save_timeseries=False,
            )
            cache[key] = summarize_candidate(raw, window, fixed_window, baseline_rmse, args)
        return cache[key]

    zero_summary = evaluate(schedule)
    records.append({
        "iteration": 0,
        "search_window": "initial",
        "trial_bias": 0.0,
        "selected": True,
        "schedule_json": json.dumps(schedule, sort_keys=True),
        **zero_summary,
    })

    for iteration in range(1, int(args.iterations) + 1):
        for window_name in windows:
            trial_rows = []
            for bias in bias_grid:
                candidate = dict(schedule)
                candidate[window_name] = float(bias)
                summary = evaluate(candidate)
                trial_row = {
                    "iteration": iteration,
                    "search_window": window_name,
                    "trial_bias": float(bias),
                    "selected": False,
                    "schedule_json": json.dumps(candidate, sort_keys=True),
                    **summary,
                }
                trial_rows.append(trial_row)
            best = min(
                trial_rows,
                key=lambda row: (
                    row["score"],
                    row["overall_rmse"],
                    row["positive_window_regret"],
                    abs(row["trial_bias"]),
                ),
            )
            schedule[window_name] = float(best["trial_bias"])
            for row in trial_rows:
                row["selected"] = bool(row["trial_bias"] == best["trial_bias"])
                records.append(row)
            if args.progress_every:
                print(
                    "[sspo_search] "
                    f"iteration={iteration} window={window_name} "
                    f"bias={best['trial_bias']:.4f} score={best['score']:.6f} "
                    f"rmse={best['overall_rmse']:.6f} regret={best['positive_window_regret']:.6f}",
                    flush=True,
                )
    return schedule, pd.DataFrame(records)


def build_window_regret_table(
        candidate_window: pd.DataFrame,
        pac_window: pd.DataFrame,
        fixed_window: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario", "window"]
    candidate = candidate_window.groupby(keys, as_index=False)["rmse_3d"].mean().rename(
        columns={"rmse_3d": "sspo_rmse_3d"}
    )
    pac = pac_window.groupby(keys, as_index=False)["rmse_3d"].mean().rename(
        columns={"rmse_3d": "formal_pac_rmse_3d"}
    )
    fixed_mean = fixed_window.groupby(keys + ["controller"], as_index=False)["rmse_3d"].mean()
    best_fixed = (
        fixed_mean.sort_values(keys + ["rmse_3d", "controller"])
        .groupby(keys, as_index=False)
        .first()
        .rename(columns={"controller": "best_fixed_controller", "rmse_3d": "best_fixed_rmse_3d"})
    )
    out = candidate.merge(pac, on=keys, how="left").merge(best_fixed, on=keys, how="left")
    out["sspo_gain_vs_formal_pac_pct"] = (
        (out["formal_pac_rmse_3d"] - out["sspo_rmse_3d"]) / out["formal_pac_rmse_3d"] * 100.0
    )
    out["sspo_gain_vs_best_fixed_pct"] = (
        (out["best_fixed_rmse_3d"] - out["sspo_rmse_3d"]) / out["best_fixed_rmse_3d"] * 100.0
    )
    out["formal_pac_gain_vs_best_fixed_pct"] = (
        (out["best_fixed_rmse_3d"] - out["formal_pac_rmse_3d"]) / out["best_fixed_rmse_3d"] * 100.0
    )
    order = {name: idx for idx, name in enumerate(DEFAULT_WINDOW_NAMES)}
    out["_order"] = out["window"].map(order).fillna(999).astype(int)
    return out.sort_values(["scenario", "_order"]).drop(columns="_order").reset_index(drop=True)


def run_sspo(args) -> dict:
    out_dir = resolve_path(args.out_dir)
    ts_dir = out_dir / "timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_dir.mkdir(parents=True, exist_ok=True)

    model, model_meta = load_alpha_model_checkpoint(
        resolve_path(args.pretrained_alpha_model),
        expected_feature_mode=args.feature_mode,
    )
    search_scenarios = parse_csv_ints(args.search_scenarios)
    search_seeds = parse_csv_ints(args.search_seeds)
    eval_scenarios = parse_csv_ints(args.eval_scenarios)
    eval_seeds = parse_csv_ints(args.eval_seeds)

    fixed_window_all = pd.read_csv(resolve_path(args.fixed_dir) / "window_metrics.csv")
    pac_raw_all = pd.read_csv(resolve_path(args.pac_dir) / "raw_metrics.csv")
    pac_window_all = pd.read_csv(resolve_path(args.pac_dir) / "window_metrics.csv")
    fixed_search = _filter_reference(fixed_window_all, search_seeds, search_scenarios)
    pac_search = _filter_reference(pac_raw_all, search_seeds, search_scenarios)
    fixed_eval = _filter_reference(fixed_window_all, eval_seeds, eval_scenarios)
    pac_eval_raw = _filter_reference(pac_raw_all, eval_seeds, eval_scenarios)
    pac_eval_window = _filter_reference(pac_window_all, eval_seeds, eval_scenarios)

    best_schedule, search_log = coordinate_search(
        model=model,
        model_meta=model_meta,
        fixed_window=fixed_search,
        baseline_raw=pac_search,
        args=args,
    )
    search_log.to_csv(out_dir / "sspo_bias_search.csv", index=False)
    (out_dir / "best_bias_schedule.json").write_text(
        json.dumps(best_schedule, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    raw, window, ts = evaluate_bias_schedule(
        model=model,
        model_meta=model_meta,
        schedule=best_schedule,
        scenarios=eval_scenarios,
        seeds=eval_seeds,
        args=args,
        method="predictive_alpha_sspo",
        save_timeseries=True,
        progress_label="sspo_eval",
    )
    raw.to_csv(out_dir / "raw_metrics.csv", index=False)
    window.to_csv(out_dir / "window_metrics.csv", index=False)
    ts.to_csv(ts_dir / "timeseries_3d.csv", index=False)

    comparison = pd.DataFrame([{
        "method": "formal_pac",
        "rmse": float(pac_eval_raw["rmse"].mean()),
        "post_startup_rmse": float(pac_eval_raw["post_startup_rmse"].mean()),
        "action_saturation_step_fraction": float(pac_eval_raw["action_saturation_step_fraction"].mean()),
        "action_jerk_mean": float(pac_eval_raw["action_jerk_mean"].mean()),
    }, {
        "method": "predictive_alpha_sspo",
        "rmse": float(raw["rmse"].mean()),
        "post_startup_rmse": float(raw["post_startup_rmse"].mean()),
        "action_saturation_step_fraction": float(raw["action_saturation_step_fraction"].mean()),
        "action_jerk_mean": float(raw["action_jerk_mean"].mean()),
    }])
    comparison.to_csv(out_dir / "sspo_vs_formal_pac_overall.csv", index=False)

    window_regret = build_window_regret_table(window, pac_eval_window, fixed_eval)
    window_regret.to_csv(out_dir / "sspo_window_regret_summary.csv", index=False)

    summary = {
        "method_name": "Supervised Predictive Initialization followed by Self-Supervised Performance Optimization",
        "best_bias_schedule": best_schedule,
        "search_seeds": search_seeds,
        "eval_seeds": eval_seeds,
        "eval_scenarios": eval_scenarios,
        "formal_pac_rmse": float(pac_eval_raw["rmse"].mean()),
        "sspo_rmse": float(raw["rmse"].mean()),
        "rmse_gain_vs_formal_pac_pct": float(
            (pac_eval_raw["rmse"].mean() - raw["rmse"].mean()) / pac_eval_raw["rmse"].mean() * 100.0
        ),
        "formal_positive_window_regret": mean_positive_window_regret(pac_eval_window, fixed_eval),
        "sspo_positive_window_regret": mean_positive_window_regret(window, fixed_eval),
        "outputs": {
            "search_log": str(out_dir / "sspo_bias_search.csv"),
            "best_bias_schedule": str(out_dir / "best_bias_schedule.json"),
            "raw_metrics": str(out_dir / "raw_metrics.csv"),
            "window_metrics": str(out_dir / "window_metrics.csv"),
            "timeseries": str(ts_dir / "timeseries_3d.csv"),
            "overall_comparison": str(out_dir / "sspo_vs_formal_pac_overall.csv"),
            "window_regret": str(out_dir / "sspo_window_regret_summary.csv"),
        },
    }
    (out_dir / "sspo_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--pretrained-alpha-model", default=str(DEFAULT_MODEL))
    parser.add_argument("--fixed-dir", default=str(DEFAULT_FIXED_DIR))
    parser.add_argument("--pac-dir", default=str(DEFAULT_PAC_DIR))
    parser.add_argument("--search-scenarios", default=",".join(str(item) for item in DEFAULT_SCENARIOS))
    parser.add_argument("--eval-scenarios", default=",".join(str(item) for item in DEFAULT_SCENARIOS))
    parser.add_argument("--search-seeds", default=",".join(str(seed) for seed in DEFAULT_SEARCH_SEEDS))
    parser.add_argument("--eval-seeds", default=",".join(str(seed) for seed in DEFAULT_EVAL_SEEDS))
    parser.add_argument("--search-windows", default=",".join(DEFAULT_WINDOW_NAMES))
    parser.add_argument("--bias-grid", default="-0.15,0,0.15,0.30")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--window-regret-weight", type=float, default=0.01)
    parser.add_argument("--rmse-guard-weight", type=float, default=5.0)
    parser.add_argument("--saturation-weight", type=float, default=0.0)
    parser.add_argument("--jerk-weight", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=2100)
    parser.add_argument("--mass-scale-xy", type=float, default=1.0)
    parser.add_argument("--damping-scale-xy", type=float, default=1.0)
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--current-amplitude-scale", type=float, default=2.5)
    parser.add_argument("--current-frequency-scale", type=float, default=1.0)
    parser.add_argument("--eval-initial-position-std", type=float, default=0.03)
    parser.add_argument("--eval-initial-velocity-std", type=float, default=0.01)
    parser.add_argument("--vertical-current", type=float, default=0.75)
    parser.add_argument("--vehicle-profile", default="real_10kg_v1")
    parser.add_argument("--action-mode", default="thruster")
    parser.add_argument("--thruster-layout", default="real_10kg_x")
    parser.add_argument("--primary-controller", default="real10kg_smc_steady")
    parser.add_argument("--authority-controller", default="real10kg_mpc_event")
    parser.add_argument("--feature-mode", default="state_phase")
    parser.add_argument("--alpha-gain", type=float, default=1.2)
    parser.add_argument("--alpha-threshold", type=float, default=0.0)
    parser.add_argument("--alpha-smoothing", type=float, default=0.5)
    parser.add_argument("--alpha-rate-limit", type=float, default=0.0125)
    parser.add_argument("--alpha-deadband", type=float, default=0.0)
    parser.add_argument("--uncertainty-mode", default="none")
    parser.add_argument("--uncertainty-samples", type=int, default=1)
    parser.add_argument("--uncertainty-gain", type=float, default=0.0)
    parser.add_argument("--uncertainty-threshold", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_sspo(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
