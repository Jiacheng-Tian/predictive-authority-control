"""Diagnose 3D authority opportunities with fixed baseline controllers.

This script does not introduce a new controller. It reuses the existing
3D trajectory environment and supported fixed baselines, then summarizes
where a later neural authority policy could intervene.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from evaluation.engineering_metrics import (
    compute_action_metrics,
    compute_timeseries_engineering_metrics,
)


SCENARIO_NAMES = {
    1: "constant",
    2: "sinusoidal",
    3: "step_change",
}

DEFAULT_WINDOWS = [
    ("startup_0_3s", 0.0, 3.0),
    ("pre_step_3_10s", 3.0, 10.0),
    ("step_recovery_10_13s", 10.0, 13.0),
    ("post_step_13_20p9s", 13.0, 20.9),
]


def choose_startup_handover_action(
        primary_action,
        authority_action,
        t: float,
        authority_until: float,
        blend_duration: float = 0.0,
        blend_curve: str = "linear") -> tuple[np.ndarray, dict[str, float]]:
    """Return a startup authority blend before cutoff."""
    alpha = authority_blend_alpha(
        t,
        authority_until=authority_until,
        blend_duration=blend_duration,
        curve=blend_curve,
    )
    primary = np.asarray(primary_action, dtype=float)
    authority = np.asarray(authority_action, dtype=float)
    action = (1.0 - alpha) * primary + alpha * authority
    active = alpha > 1e-12
    return np.asarray(action, dtype=float).copy(), {
        "authority_active": float(active),
        "authority_alpha": float(alpha),
    }


def authority_blend_alpha(
        t: float,
        authority_until: float,
        blend_duration: float = 0.0,
        curve: str = "linear") -> float:
    """Compute authority weight for hard or smooth startup handover."""
    t = float(t)
    authority_until = float(authority_until)
    blend_duration = max(0.0, float(blend_duration))
    if blend_duration <= 0.0:
        return 1.0 if t < authority_until else 0.0
    blend_start = authority_until - blend_duration
    if t < blend_start:
        return 1.0
    if t >= authority_until:
        return 0.0
    progress = (t - blend_start) / max(blend_duration, 1e-12)
    progress = float(np.clip(progress, 0.0, 1.0))
    curve = str(curve or "linear").strip().lower()
    if curve == "smoothstep":
        progress = progress * progress * (3.0 - 2.0 * progress)
    elif curve != "linear":
        raise ValueError(f"Unknown blend curve: {curve}")
    return float(np.clip(1.0 - progress, 0.0, 1.0))


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [str(item.strip()) for item in str(value).split(",") if item.strip()]


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values ** 2)))


def xy_error_from_3d_components(error, z_error) -> np.ndarray:
    """Recover XY error magnitude from 3D norm and signed z error."""
    error_arr = np.asarray(error, dtype=float)
    z_arr = np.asarray(z_error, dtype=float)
    squared = np.maximum(error_arr ** 2 - z_arr ** 2, 0.0)
    return np.sqrt(squared)


def _times_from_timeseries(ts: pd.DataFrame, dt: float) -> np.ndarray:
    if "time" in ts.columns:
        return ts["time"].to_numpy(dtype=float)
    if "step" in ts.columns:
        return ts["step"].to_numpy(dtype=float) * float(dt)
    return np.arange(len(ts), dtype=float) * float(dt)


def _z_error_from_timeseries(ts: pd.DataFrame) -> np.ndarray:
    if "z_error" in ts.columns:
        return ts["z_error"].to_numpy(dtype=float)
    if {"target_z", "z"}.issubset(set(ts.columns)):
        return (
            ts["target_z"].to_numpy(dtype=float)
            - ts["z"].to_numpy(dtype=float)
        )
    return np.zeros(len(ts), dtype=float)


def run_startup_handover_episode(
        scenario: int,
        seed: int,
        steps: int,
        mass_scale_xy: float,
        damping_scale_xy: float,
        current_amplitude_scale: float,
        current_frequency_scale: float,
        primary_controller: str = "real10kg_smc_steady",
        authority_controller: str = "real10kg_mpc_event",
        authority_until: float = 0.0,
        blend_duration: float = 0.0,
        blend_curve: str = "linear",
        start_time: float = 0.0,
        vertical_current: float = 0.0,
        vehicle_profile: str = "real_10kg_v1",
        action_mode: str = "wrench",
        thruster_layout: str = "real_10kg_x",
        save_ts: bool = False) -> dict:
    """Run a 3D startup authority handover episode."""
    from env.auv_env import AUVTrackingEnv
    from evaluation.current_control import _compute_metrics, _desired_heading
    from evaluation.current_control import (
        _build_base_controller,
        _compute_base_action,
    )

    env = AUVTrackingEnv(
        scenario=int(scenario),
        max_steps=int(steps),
        trajectory3d=True,
        mass_scale_xy=float(mass_scale_xy),
        damping_scale_xy=float(damping_scale_xy),
        current_amplitude_scale=float(current_amplitude_scale),
        current_frequency_scale=float(current_frequency_scale),
        start_time=float(start_time),
        vertical_current=float(vertical_current),
        vehicle_profile=str(vehicle_profile),
        action_mode=str(action_mode),
        thruster_layout=str(thruster_layout),
    )
    primary_name, primary = _build_base_controller(primary_controller)
    authority_name, authority = _build_base_controller(authority_controller)
    for controller in (primary, authority):
        if hasattr(controller, "reset"):
            controller.reset()
        if hasattr(controller, "set_trajectory3d"):
            controller.set_trajectory3d(True)
    env.reset(seed=int(seed))

    errors, energies, actions = [], [], []
    headings, desired_headings, step_rows = [], [], []
    rolls, pitches, desired_rolls, desired_pitches = [], [], [], []
    xs, ys, zs, target_zs, z_errors = [], [], [], [], []
    authority_active = []
    authority_alpha = []
    primary_action_rows, authority_action_rows = [], []
    thruster_forces = []
    done = False
    while not done:
        dyn = env.dynamics
        t = env.start_time + env.current_step * dyn.dt
        target = env._get_target(t)
        current_for_controller = env.privileged_state[:3]
        primary_action = _compute_base_action(
            primary,
            primary_name,
            target,
            dyn.eta,
            dyn.nu,
            t,
            dyn.dt,
            current_for_controller,
        )
        authority_action = _compute_base_action(
            authority,
            authority_name,
            target,
            dyn.eta,
            dyn.nu,
            t,
            dyn.dt,
            current_for_controller,
        )
        action, info = choose_startup_handover_action(
            primary_action,
            authority_action,
            t=t - env.start_time,
            authority_until=authority_until,
            blend_duration=blend_duration,
            blend_curve=blend_curve,
        )
        step_rows.append(int(env.current_step))
        rolls.append(float(dyn.eta[3]))
        pitches.append(float(dyn.eta[4]))
        headings.append(float(dyn.eta[5]))
        desired_rolls.append(float(target[3]))
        desired_pitches.append(float(target[4]))
        desired_headings.append(float(_desired_heading(t)))
        xs.append(float(dyn.eta[0]))
        ys.append(float(dyn.eta[1]))
        zs.append(float(dyn.eta[2]))
        target_zs.append(float(target[2]))
        primary_action_rows.append(np.asarray(primary_action, dtype=float).copy())
        authority_action_rows.append(np.asarray(authority_action, dtype=float).copy())
        authority_active.append(float(info["authority_active"]))
        authority_alpha.append(float(info["authority_alpha"]))
        _obs, _reward, done, truncated, step_info = env.step(action)
        done = done or truncated
        errors.append(float(step_info["dist_error"]))
        energies.append(float(step_info["energy"]))
        z_errors.append(float(step_info.get("z_error", target[2] - dyn.eta[2])))
        actions.append(np.asarray(action, dtype=float).copy())
        if "thruster_forces" in step_info:
            thruster_forces.append(
                np.asarray(step_info["thruster_forces"], dtype=float))

    metrics = _compute_metrics(
        errors,
        energies,
        actions,
        headings,
        desired_headings,
        env.dynamics.dt,
        z_errors=z_errors,
    )
    metrics["primary_controller"] = primary_name
    metrics["authority_controller"] = authority_name
    metrics["authority_until"] = float(authority_until)
    metrics["blend_duration"] = float(blend_duration)
    metrics["blend_curve"] = str(blend_curve)
    metrics["authority_active_fraction"] = (
        float(np.mean(authority_active)) if authority_active else 0.0
    )
    metrics["authority_alpha_mean"] = (
        float(np.mean(authority_alpha)) if authority_alpha else 0.0
    )
    metrics["trajectory3d"] = True
    metrics["current_amplitude_scale"] = float(current_amplitude_scale)
    metrics["current_frequency_scale"] = float(current_frequency_scale)
    metrics["vertical_current"] = float(vertical_current)
    metrics["vehicle_profile"] = str(vehicle_profile)
    metrics["action_mode"] = str(action_mode)
    metrics["thruster_layout"] = str(thruster_layout)
    ts = pd.DataFrame({
        "step": step_rows,
        "error": errors,
        "energy": energies,
        "roll": rolls,
        "pitch": pitches,
        "yaw": headings,
        "desired_roll": desired_rolls,
        "desired_pitch": desired_pitches,
        "desired_yaw": desired_headings,
        "heading": headings,
        "desired_heading": desired_headings,
        "x": xs,
        "y": ys,
        "z": zs,
        "target_z": target_zs,
        "z_error": z_errors,
        "authority_active": authority_active,
        "authority_alpha": authority_alpha,
    })
    action_arr = np.asarray(actions, dtype=float)
    primary_arr = np.asarray(primary_action_rows, dtype=float)
    authority_arr = np.asarray(authority_action_rows, dtype=float)
    for idx in range(action_arr.shape[1]):
        ts[f"action_{idx}"] = action_arr[:, idx]
        ts[f"primary_action_{idx}"] = primary_arr[:, idx]
        ts[f"authority_action_{idx}"] = authority_arr[:, idx]
    if thruster_forces:
        force_arr = np.asarray(thruster_forces, dtype=float)
        for idx in range(force_arr.shape[1]):
            ts[f"thruster_force_{idx}"] = force_arr[:, idx]
    ts["vehicle_profile"] = str(vehicle_profile)
    ts["action_mode"] = str(action_mode)
    ts["thruster_layout"] = str(thruster_layout)
    metrics.update(compute_timeseries_engineering_metrics(ts, dt=env.dynamics.dt))
    if save_ts:
        metrics["ts"] = ts
    return metrics


def compute_axis_window_metrics(
        ts: pd.DataFrame,
        dt: float = 0.01,
        windows: list[tuple[str, float, float]] | None = None) -> pd.DataFrame:
    """Compute 3D, XY, and Z tracking metrics in named time windows."""
    windows = windows or DEFAULT_WINDOWS
    errors = ts["error"].to_numpy(dtype=float)
    z_errors = _z_error_from_timeseries(ts)
    xy_errors = xy_error_from_3d_components(errors, z_errors)
    times = _times_from_timeseries(ts, dt=dt)
    rows = []
    for name, start, end in windows:
        mask = (times >= float(start)) & (times < float(end))
        if not np.any(mask):
            continue
        rmse_3d = _rmse(errors[mask])
        xy_rmse = _rmse(xy_errors[mask])
        z_rmse = _rmse(z_errors[mask])
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
        action_metrics = compute_action_metrics(ts.loc[mask])
        for key, value in action_metrics.items():
            row[f"window_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _scalar_metrics(metrics: dict) -> dict:
    return {
        key: value
        for key, value in metrics.items()
        if key != "ts" and np.isscalar(value)
    }


def _prepare_timeseries(ts: pd.DataFrame, dt: float) -> pd.DataFrame:
    out = ts.copy()
    out["time"] = _times_from_timeseries(out, dt=dt)
    out["z_error"] = _z_error_from_timeseries(out)
    out["xy_error"] = xy_error_from_3d_components(
        out["error"].to_numpy(dtype=float),
        out["z_error"].to_numpy(dtype=float),
    )
    out["abs_z_error"] = np.abs(out["z_error"].to_numpy(dtype=float))
    return out


def attach_episode_metadata_to_window_metrics(
        window_metrics: pd.DataFrame,
        episode_row: dict) -> pd.DataFrame:
    """Attach episode metadata without overwriting window-specific metrics."""
    out = window_metrics.copy()
    window_columns = set(out.columns)
    for key, value in episode_row.items():
        if not np.isscalar(value):
            continue
        column = key if key not in window_columns else f"episode_{key}"
        out[column] = value
    return out


def _add_reference_gain_columns(
        raw: pd.DataFrame,
        reference_controller: str) -> pd.DataFrame:
    if raw.empty or reference_controller not in set(raw["controller"]):
        return raw
    key_cols = [
        "scenario_id",
        "seed",
        "start_time",
        "current_amplitude_scale",
        "current_frequency_scale",
    ]
    if "blend_curve" in raw.columns:
        key_cols.append("blend_curve")
    metrics = ["rmse", "post_startup_rmse", "z_rmse", "action_jerk_mean"]
    ref_cols = key_cols + [col for col in metrics if col in raw.columns]
    ref = raw.loc[raw["controller"] == reference_controller, ref_cols].copy()
    ref = ref.rename(columns={
        col: f"reference_{col}"
        for col in metrics
        if col in ref.columns
    })
    out = raw.merge(ref, on=key_cols, how="left")
    for metric in metrics:
        ref_metric = f"reference_{metric}"
        if metric in out.columns and ref_metric in out.columns:
            out[f"{metric}_gain_vs_{reference_controller}_pct"] = (
                (out[ref_metric] - out[metric])
                / np.maximum(out[ref_metric], 1e-12)
                * 100.0
            )
    return out


def _aggregate_metrics(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg_cols = [
        "rmse",
        "post_startup_rmse",
        "tail_rmse",
        "disturbance_window_rmse",
        "mean_error",
        "max_error",
        "energy",
        "z_rmse",
        "z_max_abs_error",
        "action_peak_abs",
        "action_saturation_step_fraction",
        "action_tv_mean",
        "action_jerk_mean",
        "rmse_gain_vs_smc_pct",
        "post_startup_rmse_gain_vs_smc_pct",
        "z_rmse_gain_vs_smc_pct",
        "action_jerk_mean_gain_vs_smc_pct",
    ]
    agg_map = {col: "mean" for col in agg_cols if col in raw.columns}
    summary = (
        raw.groupby([
            "controller",
            "scenario",
            "current_amplitude_scale",
            "current_frequency_scale",
        ], as_index=False)
        .agg(agg_map)
        .sort_values([
            "current_amplitude_scale",
            "current_frequency_scale",
            "scenario",
            "rmse",
        ])
    )
    overall = (
        raw.groupby([
            "controller",
            "current_amplitude_scale",
            "current_frequency_scale",
        ], as_index=False)
        .agg(agg_map)
        .sort_values([
            "current_amplitude_scale",
            "current_frequency_scale",
            "rmse",
        ])
    )
    return summary, overall


def _add_handover_gain_columns(
        raw: pd.DataFrame,
        reference_until: float = 0.0) -> pd.DataFrame:
    if raw.empty:
        return raw
    key_cols = [
        "scenario_id",
        "seed",
        "start_time",
        "current_amplitude_scale",
        "current_frequency_scale",
    ]
    metrics = ["rmse", "post_startup_rmse", "z_rmse", "action_jerk_mean"]
    ref_cols = key_cols + [col for col in metrics if col in raw.columns]
    ref = raw.loc[
        np.isclose(raw["authority_until"], reference_until)
        & np.isclose(raw["blend_duration"], 0.0),
        ref_cols,
    ]
    ref = ref.rename(columns={
        col: f"reference_{col}"
        for col in metrics
        if col in ref.columns
    })
    out = raw.merge(ref, on=key_cols, how="left")
    for metric in metrics:
        ref_metric = f"reference_{metric}"
        if metric in out.columns and ref_metric in out.columns:
            out[f"{metric}_gain_vs_authority0_pct"] = (
                (out[ref_metric] - out[metric])
                / np.maximum(out[ref_metric], 1e-12)
                * 100.0
            )
    return out


def _aggregate_handover_metrics(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg_cols = [
        "rmse",
        "post_startup_rmse",
        "tail_rmse",
        "mean_error",
        "max_error",
        "energy",
        "z_rmse",
        "z_max_abs_error",
        "authority_active_fraction",
        "authority_alpha_mean",
        "action_peak_abs",
        "action_saturation_step_fraction",
        "action_tv_mean",
        "action_jerk_mean",
        "rmse_gain_vs_authority0_pct",
        "post_startup_rmse_gain_vs_authority0_pct",
        "z_rmse_gain_vs_authority0_pct",
        "action_jerk_mean_gain_vs_authority0_pct",
    ]
    agg_map = {col: "mean" for col in agg_cols if col in raw.columns}
    summary = (
        raw.groupby([
            "primary_controller",
            "authority_controller",
            "authority_until",
            "blend_duration",
            "blend_curve",
            "scenario",
            "current_amplitude_scale",
            "current_frequency_scale",
        ], as_index=False)
        .agg(agg_map)
        .sort_values([
            "authority_until",
            "current_amplitude_scale",
            "scenario",
        ])
    )
    overall = (
        raw.groupby([
            "primary_controller",
            "authority_controller",
            "authority_until",
            "blend_duration",
            "blend_curve",
            "current_amplitude_scale",
            "current_frequency_scale",
        ], as_index=False)
        .agg(agg_map)
        .sort_values(["rmse", "authority_until", "blend_duration"])
    )
    return summary, overall


def summarize_handover_windows(window_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate handover window metrics and sort by configured window order."""
    if window_metrics.empty:
        return pd.DataFrame()
    group_cols = ["authority_until", "window"]
    if "blend_duration" in window_metrics.columns:
        group_cols.insert(1, "blend_duration")
    if "blend_curve" in window_metrics.columns:
        group_cols.insert(2, "blend_curve")
    window_summary = (
        window_metrics.groupby(group_cols, as_index=False)
        .agg({
            "rmse_3d": "mean",
            "xy_rmse": "mean",
            "z_rmse": "mean",
            "z_energy_share": "mean",
            "mean_error": "mean",
            "max_error": "mean",
            "window_action_peak_abs": "mean",
            "window_action_saturation_step_fraction": "mean",
            "window_action_tv_mean": "mean",
            "window_action_jerk_mean": "mean",
        })
    )
    order = {name: idx for idx, (name, _start, _end) in enumerate(DEFAULT_WINDOWS)}
    window_summary["_window_order"] = window_summary["window"].map(order).fillna(999)
    sort_cols = ["authority_until"]
    if "blend_duration" in window_summary.columns:
        sort_cols.append("blend_duration")
    if "blend_curve" in window_summary.columns:
        sort_cols.append("blend_curve")
    sort_cols.append("_window_order")
    return (
        window_summary
        .sort_values(sort_cols, ignore_index=True)
        .drop(columns=["_window_order"])
    )


def _write_table_block(frame: pd.DataFrame, columns: list[str]) -> str:
    visible = [col for col in columns if col in frame.columns]
    if frame.empty or not visible:
        return "(no rows)"
    return frame[visible].to_string(index=False, float_format=lambda x: f"{x:.6f}")


def _plot_overall(overall: pd.DataFrame, figure_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    if overall.empty:
        return paths
    plot_df = overall.copy()
    labels = plot_df["controller"].astype(str).tolist()
    x = np.arange(len(labels))
    metrics = [
        ("rmse", "3D RMSE"),
        ("z_rmse", "Z RMSE"),
        ("post_startup_rmse", "Post-startup RMSE"),
        ("action_jerk_mean", "Action jerk"),
    ]
    metrics = [(col, label) for col, label in metrics if col in plot_df.columns]
    if not metrics:
        return paths
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.0))
    if len(metrics) == 1:
        axes = [axes]
    for ax, (col, title) in zip(axes, metrics):
        ax.bar(x, plot_df[col].to_numpy(dtype=float), color="#2F6B69")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "baseline_overall_metrics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def _plot_axis_windows(window_metrics: pd.DataFrame, figure_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    if window_metrics.empty:
        return paths
    agg = (
        window_metrics.groupby(["controller", "window"], as_index=False)
        .agg({"xy_rmse": "mean", "z_rmse": "mean", "rmse_3d": "mean"})
    )
    controllers = sorted(agg["controller"].unique())
    window_order = [name for name, _start, _end in DEFAULT_WINDOWS]
    fig, axes = plt.subplots(
        len(controllers), 1, figsize=(9.5, max(3.2, 2.8 * len(controllers))),
        sharex=True,
    )
    if len(controllers) == 1:
        axes = [axes]
    for ax, controller in zip(axes, controllers):
        sub = agg[agg["controller"] == controller].set_index("window")
        sub = sub.reindex([name for name in window_order if name in sub.index])
        x = np.arange(len(sub))
        width = 0.36
        ax.bar(x - width / 2, sub["xy_rmse"], width, label="XY RMSE",
               color="#3D7EA6")
        ax.bar(x + width / 2, sub["z_rmse"], width, label="Z RMSE",
               color="#D17A22")
        ax.plot(x, sub["rmse_3d"], color="#252525", marker="o",
                linewidth=1.5, label="3D RMSE")
        ax.set_title(controller)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].set_xticks(np.arange(len(sub)))
    axes[-1].set_xticklabels(
        [label.replace("_", "\n") for label in sub.index],
        rotation=0,
    )
    fig.tight_layout()
    path = figure_dir / "axis_window_rmse.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def _plot_timeseries_samples(timeseries: pd.DataFrame, figure_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    if timeseries.empty:
        return paths
    seed = int(timeseries["seed"].min())
    scale = float(timeseries["current_amplitude_scale"].min())
    for scenario_id in sorted(timeseries["scenario_id"].unique()):
        sub = timeseries[
            (timeseries["seed"] == seed)
            & (timeseries["scenario_id"] == scenario_id)
            & np.isclose(timeseries["current_amplitude_scale"], scale)
        ]
        if sub.empty:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True)
        for controller, ctrl_df in sub.groupby("controller"):
            axes[0].plot(ctrl_df["time"], ctrl_df["error"], label=controller)
            axes[1].plot(ctrl_df["time"], ctrl_df["xy_error"],
                         label=f"{controller} XY")
            axes[1].plot(ctrl_df["time"], ctrl_df["abs_z_error"],
                         linestyle="--", label=f"{controller} |Z|")
        axes[0].set_title(
            f"Scenario {scenario_id}: {SCENARIO_NAMES.get(int(scenario_id), scenario_id)}")
        axes[0].set_ylabel("3D error (m)")
        axes[1].set_ylabel("Axis error (m)")
        axes[1].set_xlabel("Time (s)")
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        path = figure_dir / f"timeseries_scenario{int(scenario_id)}_seed{seed}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _plot_handover_overall(overall: pd.DataFrame, figure_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if overall.empty:
        return []
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    for key, sub in overall.groupby(["blend_curve", "blend_duration"], dropna=False):
        blend_curve, blend_duration = key
        x = sub["authority_until"].to_numpy(dtype=float)
        order = np.argsort(x)
        x = x[order]
        label = f"{blend_curve}, {float(blend_duration):.2g}s"
        for ax, metric, _title in [
                (axes[0], "rmse", "3D RMSE"),
                (axes[1], "post_startup_rmse", "Post-startup RMSE"),
                (axes[2], "action_jerk_mean", "Action jerk")]:
            if metric not in sub.columns:
                continue
            y = sub[metric].to_numpy(dtype=float)[order]
            ax.plot(x, y, marker="o", linewidth=1.4, label=label)
    for ax, _metric, title in [
            (axes[0], "rmse", "3D RMSE"),
            (axes[1], "post_startup_rmse", "Post-startup RMSE"),
            (axes[2], "action_jerk_mean", "Action jerk")]:
        ax.set_title(title)
        ax.set_xlabel("Authority until (s)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = figure_dir / "startup_handover_overall.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [path]


def _write_report(
        out_dir: Path,
        figure_paths: list[Path],
        raw: pd.DataFrame,
        summary: pd.DataFrame,
        overall: pd.DataFrame,
        window_summary: pd.DataFrame,
        metadata: dict) -> None:
    report = [
        "# 3D Authority Diagnosis",
        "",
        "This diagnostic package evaluates fixed 3D baselines only. It is intended "
        "to locate authority intervention windows before training a neural policy.",
        "",
        "## Metadata",
        "```json",
        json.dumps(metadata, indent=2),
        "```",
        "",
        "## Overall Metrics",
        "```text",
        _write_table_block(overall, [
            "controller",
            "current_amplitude_scale",
            "rmse",
            "post_startup_rmse",
            "z_rmse",
            "action_jerk_mean",
            "action_saturation_step_fraction",
            "rmse_gain_vs_smc_pct",
        ]),
        "```",
        "",
        "## Scenario Metrics",
        "```text",
        _write_table_block(summary, [
            "controller",
            "scenario",
            "current_amplitude_scale",
            "rmse",
            "post_startup_rmse",
            "z_rmse",
            "action_jerk_mean",
            "rmse_gain_vs_smc_pct",
        ]),
        "```",
        "",
        "## Window Axis Metrics",
        "```text",
        _write_table_block(window_summary, [
            "controller",
            "window",
            "rmse_3d",
            "xy_rmse",
            "z_rmse",
            "z_energy_share",
            "window_action_jerk_mean",
        ]),
        "```",
        "",
        "## Figures",
    ]
    if figure_paths:
        report.extend([f"- {path}" for path in figure_paths])
    else:
        report.append("- No figures generated.")
    report.extend([
        "",
        "## Files",
        "- raw_metrics.csv",
        "- summary_by_controller_scenario.csv",
        "- overall_summary.csv",
        "- window_metrics.csv",
        "- window_summary.csv",
        "- timeseries/timeseries_3d.csv",
    ])
    del raw
    (out_dir / "diagnosis_summary.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )


def run_startup_handover_sweep(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir = _resolve_project_path(args.out_dir)
    figure_dir = _resolve_project_path(args.figure_dir)
    ts_dir = out_dir / "timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    ts_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _parse_csv_ints(args.scenarios)
    seeds = _parse_csv_ints(args.seeds)
    authority_untils = _parse_csv_floats(args.authority_untils)
    blend_durations = _parse_csv_floats(args.blend_durations)
    amplitude_scales = _parse_csv_floats(args.current_amplitude_scales)
    frequency_scales = _parse_csv_floats(args.current_frequency_scales)
    start_times = _parse_csv_floats(args.start_times)
    rows, ts_frames, window_frames = [], [], []
    dt = 0.01
    for authority_until in authority_untils:
        for blend_duration in blend_durations:
            for amp_scale in amplitude_scales:
                for freq_scale in frequency_scales:
                    for scenario in scenarios:
                        for seed in seeds:
                            for start_time in start_times:
                                metrics = run_startup_handover_episode(
                                    scenario=scenario,
                                    seed=seed,
                                    steps=args.steps,
                                    mass_scale_xy=args.mass_scale_xy,
                                    damping_scale_xy=args.damping_scale_xy,
                                    current_amplitude_scale=amp_scale,
                                    current_frequency_scale=freq_scale,
                                    primary_controller=args.primary_controller,
                                    authority_controller=args.authority_controller,
                                    authority_until=authority_until,
                                    blend_duration=blend_duration,
                                    blend_curve=args.blend_curve,
                                    start_time=start_time,
                                    vertical_current=args.vertical_current,
                                    vehicle_profile=args.vehicle_profile,
                                    action_mode=args.action_mode,
                                    thruster_layout=args.thruster_layout,
                                    save_ts=True,
                                )
                                scalar = _scalar_metrics(metrics)
                                row = {
                                    "method": "startup_handover",
                                    "scenario": SCENARIO_NAMES.get(
                                        int(scenario), str(scenario)),
                                    "scenario_id": int(scenario),
                                    "seed": int(seed),
                                    "start_time": float(start_time),
                                    "mass_scale_xy": float(args.mass_scale_xy),
                                    "damping_scale_xy": float(args.damping_scale_xy),
                                    **scalar,
                                }
                                rows.append(row)
                                ts = _prepare_timeseries(metrics["ts"], dt=dt)
                                ts["method"] = "startup_handover"
                                ts["scenario"] = row["scenario"]
                                ts["scenario_id"] = int(scenario)
                                ts["seed"] = int(seed)
                                ts["start_time"] = float(start_time)
                                ts["authority_until"] = float(authority_until)
                                ts["blend_duration"] = float(blend_duration)
                                ts["blend_curve"] = str(args.blend_curve)
                                ts["current_amplitude_scale"] = float(amp_scale)
                                ts["current_frequency_scale"] = float(freq_scale)
                                ts_frames.append(ts)
                                window = compute_axis_window_metrics(ts, dt=dt)
                                window_frames.append(
                                    attach_episode_metadata_to_window_metrics(
                                        window,
                                        row,
                                    )
                                )

    raw = _add_handover_gain_columns(pd.DataFrame(rows), reference_until=0.0)
    summary, overall = _aggregate_handover_metrics(raw)
    timeseries = (
        pd.concat(ts_frames, ignore_index=True)
        if ts_frames else pd.DataFrame()
    )
    window_metrics = (
        pd.concat(window_frames, ignore_index=True)
        if window_frames else pd.DataFrame()
    )
    window_summary = summarize_handover_windows(window_metrics)
    raw.to_csv(out_dir / "raw_metrics.csv", index=False)
    summary.to_csv(out_dir / "summary_by_authority_window.csv", index=False)
    overall.to_csv(out_dir / "overall_summary.csv", index=False)
    window_metrics.to_csv(out_dir / "window_metrics.csv", index=False)
    window_summary.to_csv(out_dir / "window_summary.csv", index=False)
    timeseries.to_csv(ts_dir / "timeseries_3d.csv", index=False)
    metadata = {
        "mode": "startup_handover",
        "steps": int(args.steps),
        "scenarios": scenarios,
        "seeds": seeds,
        "authority_untils": authority_untils,
        "blend_durations": blend_durations,
        "blend_curve": str(args.blend_curve),
        "primary_controller": str(args.primary_controller),
        "authority_controller": str(args.authority_controller),
        "current_amplitude_scales": amplitude_scales,
        "current_frequency_scales": frequency_scales,
        "vertical_current": float(args.vertical_current),
        "vehicle_profile": str(args.vehicle_profile),
        "action_mode": str(args.action_mode),
        "thruster_layout": str(args.thruster_layout),
        "start_times": start_times,
        "mass_scale_xy": float(args.mass_scale_xy),
        "damping_scale_xy": float(args.damping_scale_xy),
        "trajectory3d": True,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    figure_paths = []
    if not args.no_figures:
        figure_paths.extend(_plot_handover_overall(overall, figure_dir))
    report = [
        "# 3D Startup Authority Handover",
        "",
        "Primary controller is SMC; authority controller is used only before "
        "the configured startup cutoff.",
        "",
        "## Overall Metrics",
        "```text",
        _write_table_block(overall, [
            "primary_controller",
            "authority_controller",
            "authority_until",
            "blend_duration",
            "blend_curve",
            "rmse",
            "post_startup_rmse",
            "z_rmse",
            "authority_active_fraction",
            "authority_alpha_mean",
            "action_jerk_mean",
            "rmse_gain_vs_authority0_pct",
            "post_startup_rmse_gain_vs_authority0_pct",
        ]),
        "```",
        "",
        "## Window Metrics",
        "```text",
        _write_table_block(window_summary, [
            "authority_until",
            "blend_duration",
            "blend_curve",
            "window",
            "rmse_3d",
            "xy_rmse",
            "z_rmse",
            "z_energy_share",
            "window_action_jerk_mean",
        ]),
        "```",
        "",
        "## Figures",
    ]
    report.extend([f"- {path}" for path in figure_paths] or ["- No figures generated."])
    (out_dir / "handover_summary.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")
    print("\nStartup handover overall summary:")
    print(overall.to_string(index=False))
    print(f"\nSaved handover metrics to: {out_dir}")
    print(f"Saved handover figures to: {figure_dir}")
    return raw, summary, overall


def run_diagnosis(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from evaluation.current_control import run_fixed_controller_episode

    out_dir = _resolve_project_path(args.out_dir)
    figure_dir = _resolve_project_path(args.figure_dir)
    ts_dir = out_dir / "timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    ts_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _parse_csv_ints(args.scenarios)
    seeds = _parse_csv_ints(args.seeds)
    controllers = _parse_csv_strings(args.base_controllers)
    amplitude_scales = _parse_csv_floats(args.current_amplitude_scales)
    frequency_scales = _parse_csv_floats(args.current_frequency_scales)
    start_times = _parse_csv_floats(args.start_times)
    rows = []
    ts_frames = []
    window_frames = []
    dt = 0.01

    for controller in controllers:
        for amp_scale in amplitude_scales:
            for freq_scale in frequency_scales:
                for scenario in scenarios:
                    for seed in seeds:
                        for start_time in start_times:
                            metrics = run_fixed_controller_episode(
                                scenario=scenario,
                                seed=seed,
                                steps=args.steps,
                                mass_scale_xy=args.mass_scale_xy,
                                damping_scale_xy=args.damping_scale_xy,
                                base_controller=controller,
                                start_time=start_time,
                                current_amplitude_scale=amp_scale,
                                current_frequency_scale=freq_scale,
                                initial_position_std=args.initial_position_std,
                                initial_velocity_std=args.initial_velocity_std,
                                vertical_current=args.vertical_current,
                                vehicle_profile=args.vehicle_profile,
                                action_mode=args.action_mode,
                                thruster_layout=args.thruster_layout,
                                trajectory3d=True,
                                save_ts=True,
                            )
                            scalar = _scalar_metrics(metrics)
                            canonical_controller = str(
                                scalar.get("base_controller", controller))
                            row = {
                                "controller": canonical_controller,
                                "scenario": SCENARIO_NAMES.get(
                                    int(scenario), str(scenario)),
                                "scenario_id": int(scenario),
                                "seed": int(seed),
                                "start_time": float(start_time),
                                "mass_scale_xy": float(args.mass_scale_xy),
                                "damping_scale_xy": float(args.damping_scale_xy),
                                "initial_position_std": float(args.initial_position_std),
                                "initial_velocity_std": float(args.initial_velocity_std),
                                **scalar,
                            }
                            rows.append(row)
                            ts = _prepare_timeseries(metrics["ts"], dt=dt)
                            ts["controller"] = canonical_controller
                            ts["scenario"] = row["scenario"]
                            ts["scenario_id"] = int(scenario)
                            ts["seed"] = int(seed)
                            ts["start_time"] = float(start_time)
                            ts["current_amplitude_scale"] = float(amp_scale)
                            ts["current_frequency_scale"] = float(freq_scale)
                            ts["initial_position_std"] = float(args.initial_position_std)
                            ts["initial_velocity_std"] = float(args.initial_velocity_std)
                            ts_frames.append(ts)
                            window = compute_axis_window_metrics(ts, dt=dt)
                            window_frames.append(
                                attach_episode_metadata_to_window_metrics(
                                    window,
                                    row,
                                )
                            )

    raw = _add_reference_gain_columns(
        pd.DataFrame(rows),
        reference_controller=args.reference_controller,
    )
    summary, overall = _aggregate_metrics(raw)
    timeseries = (
        pd.concat(ts_frames, ignore_index=True)
        if ts_frames else pd.DataFrame()
    )
    window_metrics = (
        pd.concat(window_frames, ignore_index=True)
        if window_frames else pd.DataFrame()
    )
    if not window_metrics.empty:
        window_summary = (
            window_metrics.groupby(["controller", "window"], as_index=False)
            .agg({
                "rmse_3d": "mean",
                "xy_rmse": "mean",
                "z_rmse": "mean",
                "z_energy_share": "mean",
                "mean_error": "mean",
                "max_error": "mean",
                "window_action_peak_abs": "mean",
                "window_action_saturation_step_fraction": "mean",
                "window_action_tv_mean": "mean",
                "window_action_jerk_mean": "mean",
            })
        )
        order = {name: idx for idx, (name, _s, _e) in enumerate(DEFAULT_WINDOWS)}
        window_summary["_window_order"] = window_summary["window"].map(order)
        window_summary = window_summary.sort_values([
            "controller",
            "_window_order",
        ]).drop(columns=["_window_order"])
    else:
        window_summary = pd.DataFrame()

    raw.to_csv(out_dir / "raw_metrics.csv", index=False)
    summary.to_csv(out_dir / "summary_by_controller_scenario.csv", index=False)
    overall.to_csv(out_dir / "overall_summary.csv", index=False)
    window_metrics.to_csv(out_dir / "window_metrics.csv", index=False)
    window_summary.to_csv(out_dir / "window_summary.csv", index=False)
    timeseries.to_csv(ts_dir / "timeseries_3d.csv", index=False)
    metadata = {
        "steps": int(args.steps),
        "scenarios": scenarios,
        "seeds": seeds,
        "base_controllers": controllers,
        "current_amplitude_scales": amplitude_scales,
        "current_frequency_scales": frequency_scales,
        "initial_position_std": float(args.initial_position_std),
        "initial_velocity_std": float(args.initial_velocity_std),
        "vertical_current": float(args.vertical_current),
        "vehicle_profile": str(args.vehicle_profile),
        "action_mode": str(args.action_mode),
        "thruster_layout": str(args.thruster_layout),
        "start_times": start_times,
        "mass_scale_xy": float(args.mass_scale_xy),
        "damping_scale_xy": float(args.damping_scale_xy),
        "reference_controller": str(args.reference_controller),
        "trajectory3d": True,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    figure_paths = []
    if not args.no_figures:
        figure_paths.extend(_plot_overall(overall, figure_dir))
        figure_paths.extend(_plot_axis_windows(window_metrics, figure_dir))
        figure_paths.extend(_plot_timeseries_samples(timeseries, figure_dir))
    _write_report(
        out_dir=out_dir,
        figure_paths=figure_paths,
        raw=raw,
        summary=summary,
        overall=overall,
        window_summary=window_summary,
        metadata=metadata,
    )
    print("\nOverall summary:")
    print(overall.to_string(index=False))
    print("\nWindow summary:")
    print(window_summary.to_string(index=False))
    print(f"\nSaved metrics to: {out_dir}")
    print(f"Saved figures to: {figure_dir}")
    return raw, summary, overall


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="baseline",
        choices=["baseline", "startup_handover"],
    )
    parser.add_argument(
        "--out-dir",
        default="results/3d_authority_diagnosis/baseline_scale2p5",
    )
    parser.add_argument(
        "--figure-dir",
        default="results/figures/3d_authority_diagnosis/baseline_scale2p5",
    )
    parser.add_argument("--scenarios", default="1,2,3")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--start-times", default="0")
    parser.add_argument("--steps", type=int, default=2100)
    parser.add_argument("--mass-scale-xy", type=float, default=1.5)
    parser.add_argument("--damping-scale-xy", type=float, default=0.5)
    parser.add_argument("--current-amplitude-scales", default="2.5")
    parser.add_argument("--current-frequency-scales", default="1.0")
    parser.add_argument("--initial-position-std", type=float, default=0.0)
    parser.add_argument("--initial-velocity-std", type=float, default=0.0)
    parser.add_argument("--vertical-current", type=float, default=0.0)
    parser.add_argument("--vehicle-profile", default="real_10kg_v1")
    parser.add_argument("--action-mode", default="wrench", choices=["wrench", "thruster"])
    parser.add_argument("--thruster-layout", default="real_10kg_x")
    parser.add_argument(
        "--base-controllers",
        default="real10kg_smc_steady,real10kg_mpc_event",
    )
    parser.add_argument("--primary-controller", default="real10kg_smc_steady")
    parser.add_argument("--authority-controller", default="real10kg_mpc_event")
    parser.add_argument("--authority-untils", default="0,0.5,1,1.5,2,2.5,3")
    parser.add_argument("--blend-durations", default="0")
    parser.add_argument(
        "--blend-curve",
        default="linear",
        choices=["linear", "smoothstep"],
    )
    parser.add_argument("--reference-controller", default="real10kg_smc_steady")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "startup_handover":
        run_startup_handover_sweep(args)
    else:
        run_diagnosis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

