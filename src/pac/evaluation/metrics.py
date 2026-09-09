"""Engineering metrics derived from validation timeseries."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _action_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        [
            col for col in df.columns
            if str(col).startswith("action_")
            and str(col).split("_", 1)[1].isdigit()
        ],
        key=lambda name: int(str(name).split("_", 1)[1]),
    )


def compute_action_metrics(
        df: pd.DataFrame,
        saturation_threshold: float = 0.99) -> dict[str, float]:
    """Compute actuator-load metrics from action columns."""
    cols = _action_columns(df)
    if not cols:
        return {
            "action_peak_abs": 0.0,
            "action_saturation_fraction": 0.0,
            "action_saturation_step_fraction": 0.0,
            "action_tv_mean": 0.0,
            "action_jerk_mean": 0.0,
        }
    actions = df[cols].to_numpy(dtype=float)
    abs_actions = np.abs(actions)
    saturated = abs_actions >= float(saturation_threshold)
    if actions.shape[0] > 1:
        delta = np.diff(actions, axis=0)
        tv = np.sum(np.abs(delta), axis=1)
    else:
        delta = np.empty((0, actions.shape[1]))
        tv = np.empty(0)
    if delta.shape[0] > 1:
        jerk = np.sum(np.abs(np.diff(delta, axis=0)), axis=1)
    else:
        jerk = np.empty(0)
    return {
        "action_peak_abs": float(np.max(abs_actions)) if abs_actions.size else 0.0,
        "action_saturation_fraction": float(np.mean(saturated)) if saturated.size else 0.0,
        "action_saturation_step_fraction": (
            float(np.mean(np.any(saturated, axis=1))) if saturated.size else 0.0
        ),
        "action_tv_mean": float(np.mean(tv)) if tv.size else 0.0,
        "action_jerk_mean": float(np.mean(jerk)) if jerk.size else 0.0,
    }


def _rmse(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values ** 2)))


def compute_error_window_metrics(
        df: pd.DataFrame,
        dt: float,
        post_startup_time: float = 3.0,
        tail_fraction: float = 0.2,
        disturbance_time: float = 30.0,
        disturbance_window: float = 10.0) -> dict[str, float]:
    """Compute post-startup, tail, and disturbance-window tracking metrics."""
    # Preserve the formal-v2 disturbance fields for result-schema compatibility.
    if "error" not in df:
        return {
            "post_startup_rmse": float("nan"),
            "tail_rmse": float("nan"),
            "disturbance_window_rmse": float("nan"),
            "disturbance_recovery_time_s": float("nan"),
        }
    errors = df["error"].to_numpy(dtype=float)
    if "time" in df:
        times = df["time"].to_numpy(dtype=float)
    elif "step" in df:
        times = df["step"].to_numpy(dtype=float) * float(dt)
    else:
        times = np.arange(errors.size, dtype=float) * float(dt)
    post_mask = times >= float(post_startup_time)
    tail_start = int(max(0, np.floor((1.0 - float(tail_fraction)) * errors.size)))
    disturbance_mask = (
        (times >= float(disturbance_time))
        & (times < float(disturbance_time) + float(disturbance_window))
    )
    recovery_threshold = 0.05
    after_event = np.where(times >= float(disturbance_time))[0]
    recovery_time = float("nan")
    if after_event.size:
        for idx in after_event:
            if errors[idx] <= recovery_threshold:
                recovery_time = float(times[idx] - float(disturbance_time))
                break
    return {
        "post_startup_rmse": _rmse(errors[post_mask]),
        "tail_rmse": _rmse(errors[tail_start:]),
        "disturbance_window_rmse": _rmse(errors[disturbance_mask]),
        "disturbance_recovery_time_s": recovery_time,
    }


def compute_timeseries_engineering_metrics(
        df: pd.DataFrame,
        dt: float = 0.01,
        saturation_threshold: float = 0.99) -> dict[str, float]:
    """Compute all engineering metrics from a validation timeseries."""
    metrics = compute_action_metrics(df, saturation_threshold=saturation_threshold)
    if "actuator_rate_limited_fraction" in df.columns:
        values = pd.to_numeric(
            df["actuator_rate_limited_fraction"], errors="coerce"
        ).dropna().to_numpy(dtype=float)
        metrics["actuator_rate_limited_fraction_mean"] = (
            float(np.mean(values)) if values.size else 0.0
        )
        metrics["actuator_rate_limit_episode_mean"] = metrics[
            "actuator_rate_limited_fraction_mean"
        ]
    if "actuator_amplitude_clipped_fraction" in df.columns:
        values = pd.to_numeric(
            df["actuator_amplitude_clipped_fraction"], errors="coerce"
        ).dropna().to_numpy(dtype=float)
        metrics["actuator_amplitude_clipped_fraction_mean"] = (
            float(np.mean(values)) if values.size else 0.0
        )
    if "authority_alpha" in df.columns:
        alpha = (
            pd.to_numeric(df["authority_alpha"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        if alpha.size >= 2:
            alpha_delta = np.abs(np.diff(alpha))
            metrics["alpha_delta_mean"] = (
                float(np.mean(alpha_delta)) if alpha_delta.size else 0.0
            )
            metrics["alpha_delta_max"] = (
                float(np.max(alpha_delta)) if alpha_delta.size else 0.0
            )
            if alpha_delta.size >= 2:
                metrics["alpha_jerk_mean"] = float(
                    np.mean(np.abs(np.diff(alpha_delta))))
            else:
                metrics["alpha_jerk_mean"] = 0.0
        else:
            metrics["alpha_delta_mean"] = 0.0
            metrics["alpha_delta_max"] = 0.0
            metrics["alpha_jerk_mean"] = 0.0
    metrics.update(compute_error_window_metrics(df, dt=dt))
    return metrics
