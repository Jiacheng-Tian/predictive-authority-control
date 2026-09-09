"""Run one formal PAC Transformer training seed and its evaluations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pac.authority.evaluation import run_predictive_alpha_episode, scalar_metrics
from pac.authority.model import train_alpha_model
from pac.authority.training import collect_teacher_dataset
from pac.config import PACConfig, load_config
from pac.evaluation.diagnostics import SCENARIO_NAMES, attach_episode_metadata, compute_axis_window_metrics


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT / "config" / "pac.yaml"


def parse_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values:
        raise ValueError("integer list must not be empty")
    return values


def parse_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values:
        raise ValueError("float list must not be empty")
    return values


def portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def require_empty_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")


def plot_summary(overall: pd.DataFrame, figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    labels = overall["method"].tolist()
    for axis, metric, title in zip(
            axes,
            ["rmse", "post_startup_rmse", "authority_alpha_mean"],
            ["3-D RMSE", "Post-startup RMSE", "Mean authority"],
    ):
        axis.bar(np.arange(len(overall)), overall[metric].to_numpy(dtype=float))
        axis.set_title(title)
        axis.set_xticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = figure_dir / "predictive_alpha_summary.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def run_experiment(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train one Transformer seed and evaluate the requested episode grid."""
    out_dir = Path(args.out_dir).resolve()
    figure_dir = Path(args.figure_dir).resolve()
    require_empty_directory(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    timeseries_dir = out_dir / "timeseries"
    timeseries_dir.mkdir(parents=True, exist_ok=True)

    train_scenarios = parse_ints(args.train_scenarios)
    train_seeds = parse_ints(args.train_seeds)
    eval_scenarios = parse_ints(args.eval_scenarios)
    eval_seeds = parse_ints(args.eval_seeds)
    alpha_grid = parse_floats(args.oracle_alpha_grid)
    features, labels, label_table = collect_teacher_dataset(
        scenarios=train_scenarios,
        data_seeds=train_seeds,
        steps=args.steps,
        mass_scale_xy=args.mass_scale_xy,
        damping_scale_xy=args.damping_scale_xy,
        current_amplitude_scale=args.current_amplitude_scale,
        current_frequency_scale=args.current_frequency_scale,
        vertical_current=args.vertical_current,
        primary_controller=args.primary_controller,
        authority_controller=args.authority_controller,
        feature_mode=args.feature_mode,
        oracle_alpha_grid=alpha_grid,
        oracle_horizon_steps=args.oracle_horizon_steps,
        oracle_action_saturation_weight=args.oracle_action_saturation_weight,
        oracle_action_delta_weight=args.oracle_action_delta_weight,
        oracle_alpha_delta_weight=args.oracle_alpha_delta_weight,
        vehicle_profile=args.vehicle_profile,
        thruster_layout=args.thruster_layout,
    )
    label_table.to_csv(out_dir / "teacher_alpha_labels.csv", index=False)
    model, train_metrics = train_alpha_model(
        features,
        labels,
        label_table,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.train_seed,
        history_len=args.history_len,
        embed_dim=args.transformer_embed_dim,
        heads=args.transformer_heads,
        layers=args.transformer_layers,
        dropout=args.model_dropout,
    )
    train_metrics.update({
        "teacher_cache_path": "",
        "teacher_cache_loaded": False,
        "teacher_samples_total": int(labels.size),
        "teacher_samples_used": int(labels.size),
        "max_train_samples": 0,
    })

    rows, window_frames, timeseries_frames = [], [], []
    total = len(eval_scenarios) * len(eval_seeds)
    completed = 0
    started = time.time()
    for scenario in eval_scenarios:
        for seed in eval_seeds:
            metrics = run_predictive_alpha_episode(
                model=model,
                scenario=scenario,
                seed=seed,
                steps=args.steps,
                mass_scale_xy=args.mass_scale_xy,
                damping_scale_xy=args.damping_scale_xy,
                current_amplitude_scale=args.current_amplitude_scale,
                current_frequency_scale=args.current_frequency_scale,
                vertical_current=args.vertical_current,
                primary_controller=args.primary_controller,
                authority_controller=args.authority_controller,
                feature_mode=args.feature_mode,
                initial_position_std=args.eval_initial_position_std,
                initial_velocity_std=args.eval_initial_velocity_std,
                alpha_gain=args.alpha_gain,
                alpha_threshold=0.0,
                vehicle_profile=args.vehicle_profile,
                thruster_layout=args.thruster_layout,
                alpha_smoothing=args.alpha_smoothing,
                alpha_rate_limit=args.alpha_rate_limit,
                alpha_deadband=args.alpha_deadband,
                policy_architecture="transformer",
                history_len=args.history_len,
                save_ts=True,
            )
            row = {
                "method": "predictive_alpha",
                "scenario": SCENARIO_NAMES.get(scenario, str(scenario)),
                "scenario_id": scenario,
                "seed": seed,
                "start_time": 0.0,
                "mass_scale_xy": args.mass_scale_xy,
                "damping_scale_xy": args.damping_scale_xy,
                "initial_position_std": args.eval_initial_position_std,
                "initial_velocity_std": args.eval_initial_velocity_std,
                **scalar_metrics(metrics),
            }
            rows.append(row)
            timeseries = metrics["ts"].copy()
            timeseries["relative_time"] = timeseries["step"].to_numpy(dtype=float) * 0.01
            timeseries["method"] = "predictive_alpha"
            timeseries["scenario_id"] = scenario
            timeseries["seed"] = seed
            timeseries["start_time"] = 0.0
            timeseries["current_amplitude_scale"] = args.current_amplitude_scale
            timeseries["current_frequency_scale"] = args.current_frequency_scale
            timeseries["initial_position_std"] = args.eval_initial_position_std
            timeseries["initial_velocity_std"] = args.eval_initial_velocity_std
            if args.save_timeseries:
                timeseries_frames.append(timeseries)
            window_frames.append(attach_episode_metadata(
                compute_axis_window_metrics(timeseries),
                row,
            ))
            completed += 1
            if args.progress_every and (
                    completed == 1
                    or completed % args.progress_every == 0
                    or completed == total):
                print(
                    f"[pac] eval {completed}/{total} scenario={scenario} seed={seed} "
                    f"elapsed_s={time.time() - started:.1f}",
                    flush=True,
                )

    raw = pd.DataFrame(rows)
    aggregate = {
        "rmse": "mean",
        "post_startup_rmse": "mean",
        "tail_rmse": "mean",
        "mean_error": "mean",
        "max_error": "mean",
        "energy": "mean",
        "z_rmse": "mean",
        "authority_active_fraction": "mean",
        "authority_alpha_mean": "mean",
        "authority_alpha_std": "mean",
        "action_saturation_step_fraction": "mean",
        "action_tv_mean": "mean",
        "action_jerk_mean": "mean",
        "alpha_raw_mean": "mean",
        "alpha_uncertainty_std": "mean",
    }
    overall = raw.groupby("method", as_index=False).agg(aggregate).sort_values("rmse")
    by_scenario = raw.groupby(["method", "scenario"], as_index=False).agg(aggregate).sort_values([
        "method", "scenario",
    ])
    window_metrics = pd.concat(window_frames, ignore_index=True)
    raw.to_csv(out_dir / "raw_metrics.csv", index=False)
    overall.to_csv(out_dir / "overall_summary.csv", index=False)
    by_scenario.to_csv(out_dir / "by_scenario.csv", index=False)
    window_metrics.to_csv(out_dir / "window_metrics.csv", index=False)
    if args.save_timeseries and timeseries_frames:
        pd.concat(timeseries_frames, ignore_index=True).to_csv(
            timeseries_dir / "timeseries_3d.csv",
            index=False,
        )
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": int(features.shape[1]),
        "feature_mode": args.feature_mode,
        "hidden_dim": 64,
        "policy_architecture": "transformer",
        "history_len": args.history_len,
        "transformer_embed_dim": args.transformer_embed_dim,
        "transformer_heads": args.transformer_heads,
        "transformer_layers": args.transformer_layers,
        "model_dropout": args.model_dropout,
    }, out_dir / "predictive_alpha_model.pt")
    manifest = {
        "protocol": {
            key: portable_path(value) if key in {"config", "out_dir", "figure_dir"} else value
            for key, value in vars(args).items()
        },
        "dataset_samples": int(labels.size),
        "training_samples": int(labels.size),
        "save_timeseries": bool(args.save_timeseries),
        "feature_dim": int(features.shape[1]),
        "train_metrics": train_metrics,
        "model": {
            "policy_architecture": "transformer",
            "feature_mode": args.feature_mode,
            "input_dim": int(features.shape[1]),
            "hidden_dim": 64,
            "history_len": args.history_len,
            "transformer_embed_dim": args.transformer_embed_dim,
            "transformer_heads": args.transformer_heads,
            "transformer_layers": args.transformer_layers,
            "model_dropout": args.model_dropout,
        },
        "formal_scope_guardrail": (
            "The learned model predicts only authority alpha between the SMC primary "
            "and one-step predictive authority controller."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_summary(overall, figure_dir)
    (out_dir / "predictive_alpha_summary.md").write_text(
        "# Predictive Authority Control\n\n## Overall\n```text\n"
        + overall.to_string(index=False)
        + "\n```\n",
        encoding="utf-8",
    )
    return raw, overall


def build_arg_parser(config: PACConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--steps", type=int, default=config.environment.steps)
    parser.add_argument("--train-scenarios", default=",".join(map(str, config.environment.scenarios)))
    parser.add_argument("--train-seeds", default=",".join(map(str, config.training.data_seeds)))
    parser.add_argument("--eval-scenarios", default=",".join(map(str, config.environment.scenarios)))
    parser.add_argument("--eval-seeds", default=",".join(map(str, config.evaluation.episode_seeds)))
    parser.add_argument("--mass-scale-xy", type=float, default=config.environment.mass_scale_xy)
    parser.add_argument("--damping-scale-xy", type=float, default=config.environment.damping_scale_xy)
    parser.add_argument("--current-amplitude-scale", type=float, default=config.environment.current_amplitude_scale)
    parser.add_argument("--current-frequency-scale", type=float, default=config.environment.current_frequency_scale)
    parser.add_argument("--eval-initial-position-std", type=float, default=config.environment.eval_initial_position_std)
    parser.add_argument("--eval-initial-velocity-std", type=float, default=config.environment.eval_initial_velocity_std)
    parser.add_argument("--vertical-current", type=float, default=config.environment.vertical_current)
    parser.add_argument("--vehicle-profile", default=config.environment.vehicle_profile)
    parser.add_argument("--thruster-layout", default=config.environment.thruster_layout)
    parser.add_argument("--primary-controller", default=config.controllers.primary)
    parser.add_argument("--authority-controller", default=config.controllers.authority)
    parser.add_argument("--teacher-mode", choices=["oracle"], default="oracle")
    parser.add_argument("--feature-mode", choices=["state_phase"], default="state_phase")
    parser.add_argument("--policy-architecture", choices=["transformer"], default="transformer")
    parser.add_argument("--history-len", type=int, default=config.authority.history_len)
    parser.add_argument("--transformer-embed-dim", type=int, default=config.authority.embed_dim)
    parser.add_argument("--transformer-heads", type=int, default=config.authority.heads)
    parser.add_argument("--transformer-layers", type=int, default=config.authority.layers)
    parser.add_argument("--model-dropout", type=float, default=config.authority.dropout)
    parser.add_argument("--epochs", type=int, default=config.training.epochs)
    parser.add_argument("--batch-size", type=int, default=config.training.batch_size)
    parser.add_argument("--lr", type=float, default=config.training.learning_rate)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--alpha-gain", type=float, default=config.authority.alpha_gain)
    parser.add_argument("--alpha-threshold", type=float, default=0.0)
    parser.add_argument("--oracle-alpha-grid", default=",".join(map(str, config.authority.oracle_alpha_grid)))
    parser.add_argument("--oracle-horizon-steps", type=int, default=config.authority.oracle_horizon_steps)
    parser.add_argument("--oracle-action-saturation-weight", type=float, default=config.authority.oracle_action_saturation_weight)
    parser.add_argument("--oracle-action-delta-weight", type=float, default=config.authority.oracle_action_delta_weight)
    parser.add_argument("--oracle-alpha-delta-weight", type=float, default=config.authority.oracle_alpha_delta_weight)
    parser.add_argument("--alpha-smoothing", type=float, default=config.authority.alpha_smoothing)
    parser.add_argument("--alpha-rate-limit", type=float, default=config.authority.alpha_rate_limit)
    parser.add_argument("--alpha-deadband", type=float, default=config.authority.alpha_deadband)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--save-timeseries", dest="save_timeseries", action="store_true", default=True)
    parser.add_argument("--no-save-timeseries", dest="save_timeseries", action="store_false")
    return parser


def main(argv=None) -> int:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    known, _unknown = config_parser.parse_known_args(argv)
    config = load_config(known.config)
    run_experiment(build_arg_parser(config).parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
