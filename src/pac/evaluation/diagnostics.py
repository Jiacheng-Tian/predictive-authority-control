"""Formal fixed-controller evaluation and window summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pac.evaluation.episodes import run_fixed_controller_episode
from pac.evaluation.metrics import compute_action_metrics


SCENARIO_NAMES = {1: "constant", 2: "sinusoidal", 3: "step_change"}
DEFAULT_WINDOWS = (
    ("startup_0_3s", 0.0, 3.0),
    ("pre_step_3_10s", 3.0, 10.0),
    ("step_recovery_10_13s", 10.0, 13.0),
    ("post_step_13_20p9s", 13.0, 20.9),
)


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values ** 2))) if values.size else float("nan")


def times_from_timeseries(timeseries: pd.DataFrame, dt: float) -> np.ndarray:
    if "step" in timeseries:
        return timeseries["step"].to_numpy(dtype=float) * float(dt)
    return np.arange(len(timeseries), dtype=float) * float(dt)


def z_error_from_timeseries(timeseries: pd.DataFrame) -> np.ndarray:
    if "z_error" in timeseries:
        return timeseries["z_error"].to_numpy(dtype=float)
    if {"target_z", "z"}.issubset(timeseries.columns):
        return (
            timeseries["target_z"].to_numpy(dtype=float)
            - timeseries["z"].to_numpy(dtype=float)
        )
    return np.zeros(len(timeseries), dtype=float)


def compute_axis_window_metrics(
        timeseries: pd.DataFrame,
        dt: float = 0.01,
        windows=DEFAULT_WINDOWS) -> pd.DataFrame:
    """Compute the archived four-window tracking and action metrics."""
    errors = timeseries["error"].to_numpy(dtype=float)
    z_errors = z_error_from_timeseries(timeseries)
    xy_errors = np.sqrt(np.maximum(errors ** 2 - z_errors ** 2, 0.0))
    times = times_from_timeseries(timeseries, dt)
    rows = []
    for name, start, end in windows:
        mask = (times >= float(start)) & (times < float(end))
        if not np.any(mask):
            continue
        rmse_3d = rmse(errors[mask])
        xy_rmse = rmse(xy_errors[mask])
        z_rmse = rmse(z_errors[mask])
        row = {
            "window": str(name),
            "window_start_s": float(start),
            "window_end_s": float(end),
            "n_steps": int(np.sum(mask)),
            "rmse_3d": rmse_3d,
            "xy_rmse": xy_rmse,
            "z_rmse": z_rmse,
            "xy_share": float(xy_rmse / max(rmse_3d, 1e-12)),
            "z_share": float(z_rmse / max(rmse_3d, 1e-12)),
            "z_energy_share": float((z_rmse ** 2) / max(rmse_3d ** 2, 1e-12)),
            "mean_error": float(np.mean(errors[mask])),
            "max_error": float(np.max(errors[mask])),
        }
        for key, value in compute_action_metrics(timeseries.loc[mask]).items():
            row[f"window_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def attach_episode_metadata(window: pd.DataFrame, episode: dict) -> pd.DataFrame:
    output = window.copy()
    existing = set(output.columns)
    for key, value in episode.items():
        if np.isscalar(value):
            output[key if key not in existing else f"episode_{key}"] = value
    return output


def add_reference_gains(raw: pd.DataFrame, reference_controller: str) -> pd.DataFrame:
    keys = [
        "scenario_id",
        "seed",
        "start_time",
        "current_amplitude_scale",
        "current_frequency_scale",
    ]
    metrics = ["rmse", "post_startup_rmse", "z_rmse", "action_jerk_mean"]
    reference = raw.loc[raw["controller"].eq(reference_controller), keys + metrics].copy()
    reference = reference.rename(columns={name: f"reference_{name}" for name in metrics})
    output = raw.merge(reference, on=keys, how="left")
    for metric in metrics:
        output[f"{metric}_gain_vs_smc_pct"] = (
            (output[f"reference_{metric}"] - output[metric])
            / np.maximum(output[f"reference_{metric}"], 1e-12)
            * 100.0
        )
    return output


def aggregate_metrics(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "rmse", "post_startup_rmse", "tail_rmse", "disturbance_window_rmse",
        "mean_error", "max_error", "energy", "z_rmse", "z_max_abs_error",
        "action_peak_abs", "action_saturation_step_fraction", "action_tv_mean",
        "action_jerk_mean", "rmse_gain_vs_smc_pct",
        "post_startup_rmse_gain_vs_smc_pct", "z_rmse_gain_vs_smc_pct",
        "action_jerk_mean_gain_vs_smc_pct",
    ]
    aggregation = {column: "mean" for column in columns if column in raw}
    summary = raw.groupby([
        "controller", "scenario", "current_amplitude_scale", "current_frequency_scale",
    ], as_index=False).agg(aggregation).sort_values([
        "current_amplitude_scale", "current_frequency_scale", "scenario", "rmse",
    ])
    overall = raw.groupby([
        "controller", "current_amplitude_scale", "current_frequency_scale",
    ], as_index=False).agg(aggregation).sort_values([
        "current_amplitude_scale", "current_frequency_scale", "rmse",
    ])
    return summary, overall


def run_diagnosis(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the two formal fixed controllers over the requested episode grid."""
    out_dir = Path(args.out_dir).resolve()
    figure_dir = Path(args.figure_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "timeseries").mkdir(parents=True, exist_ok=True)
    scenarios = parse_csv_ints(args.scenarios)
    seeds = parse_csv_ints(args.seeds)
    controllers = parse_csv_strings(args.base_controllers)
    amplitudes = parse_csv_floats(args.current_amplitude_scales)
    frequencies = parse_csv_floats(args.current_frequency_scales)
    rows, windows, timeseries_frames = [], [], []
    for controller in controllers:
        for scenario in scenarios:
            for seed in seeds:
                for amplitude in amplitudes:
                    for frequency in frequencies:
                        metrics = run_fixed_controller_episode(
                            scenario=scenario,
                            seed=seed,
                            steps=args.steps,
                            mass_scale_xy=args.mass_scale_xy,
                            damping_scale_xy=args.damping_scale_xy,
                            base_controller=controller,
                            current_amplitude_scale=amplitude,
                            current_frequency_scale=frequency,
                            initial_position_std=args.initial_position_std,
                            initial_velocity_std=args.initial_velocity_std,
                            vertical_current=args.vertical_current,
                            vehicle_profile=args.vehicle_profile,
                            thruster_layout=args.thruster_layout,
                            save_ts=True,
                        )
                        row = {
                            "controller": metrics["base_controller"],
                            "scenario": SCENARIO_NAMES.get(scenario, str(scenario)),
                            "scenario_id": scenario,
                            "seed": seed,
                            **{key: value for key, value in metrics.items() if key != "ts" and np.isscalar(value)},
                        }
                        rows.append(row)
                        timeseries = metrics["ts"].copy()
                        timeseries["time"] = timeseries["step"].to_numpy(dtype=float) * 0.01
                        timeseries["controller"] = row["controller"]
                        timeseries["scenario"] = row["scenario"]
                        timeseries["scenario_id"] = scenario
                        timeseries["seed"] = seed
                        timeseries["start_time"] = 0.0
                        timeseries["current_amplitude_scale"] = amplitude
                        timeseries["current_frequency_scale"] = frequency
                        timeseries_frames.append(timeseries)
                        windows.append(attach_episode_metadata(
                            compute_axis_window_metrics(timeseries),
                            row,
                        ))
    raw = add_reference_gains(pd.DataFrame(rows), args.reference_controller)
    summary, overall = aggregate_metrics(raw)
    window_metrics = pd.concat(windows, ignore_index=True)
    window_summary = window_metrics.groupby(["controller", "window"], as_index=False).agg({
        "rmse_3d": "mean",
        "xy_rmse": "mean",
        "z_rmse": "mean",
        "z_energy_share": "mean",
        "window_action_jerk_mean": "mean",
    })
    combined_timeseries = pd.concat(timeseries_frames, ignore_index=True)
    raw.to_csv(out_dir / "raw_metrics.csv", index=False)
    summary.to_csv(out_dir / "summary_by_controller_scenario.csv", index=False)
    overall.to_csv(out_dir / "overall_summary.csv", index=False)
    window_metrics.to_csv(out_dir / "window_metrics.csv", index=False)
    window_summary.to_csv(out_dir / "window_summary.csv", index=False)
    combined_timeseries.to_csv(out_dir / "timeseries" / "timeseries_3d.csv", index=False)
    metadata = {
        "steps": int(args.steps),
        "scenarios": scenarios,
        "seeds": seeds,
        "base_controllers": controllers,
        "current_amplitude_scales": amplitudes,
        "current_frequency_scales": frequencies,
        "initial_position_std": float(args.initial_position_std),
        "initial_velocity_std": float(args.initial_velocity_std),
        "vertical_current": float(args.vertical_current),
        "vehicle_profile": args.vehicle_profile,
        "action_mode": "thruster",
        "thruster_layout": args.thruster_layout,
        "start_times": [0.0],
        "mass_scale_xy": float(args.mass_scale_xy),
        "damping_scale_xy": float(args.damping_scale_xy),
        "reference_controller": args.reference_controller,
        "trajectory3d": True,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not args.no_figures:
        for scenario in scenarios:
            subset = combined_timeseries[combined_timeseries["scenario_id"].eq(scenario)]
            figure, axis = plt.subplots(figsize=(7, 4))
            for controller, controller_rows in subset.groupby("controller"):
                first_seed = controller_rows["seed"].min()
                sample = controller_rows[controller_rows["seed"].eq(first_seed)]
                axis.plot(sample["time"], sample["error"], label=controller)
            axis.set_xlabel("Time (s)")
            axis.set_ylabel("3-D error (m)")
            axis.legend()
            figure.tight_layout()
            figure.savefig(figure_dir / f"timeseries_scenario{scenario}_seed{seeds[0]}.png", dpi=180)
            plt.close(figure)
    return raw, summary, overall


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline"], default="baseline")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--scenarios", default="1,2,3")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--steps", type=int, default=2100)
    parser.add_argument("--mass-scale-xy", type=float, default=1.0)
    parser.add_argument("--damping-scale-xy", type=float, default=1.0)
    parser.add_argument("--current-amplitude-scales", default="2.5")
    parser.add_argument("--current-frequency-scales", default="1.0")
    parser.add_argument("--initial-position-std", type=float, default=0.03)
    parser.add_argument("--initial-velocity-std", type=float, default=0.01)
    parser.add_argument("--vertical-current", type=float, default=0.75)
    parser.add_argument("--vehicle-profile", default="real_10kg_v1")
    parser.add_argument("--thruster-layout", default="real_10kg_x")
    parser.add_argument("--base-controllers", default="real10kg_smc_steady,real10kg_predictive_event")
    parser.add_argument("--reference-controller", default="real10kg_smc_steady")
    parser.add_argument("--no-figures", action="store_true")
    return parser


def main(argv=None) -> int:
    run_diagnosis(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
