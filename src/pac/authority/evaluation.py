"""Formal-v2-compatible closed-loop PAC evaluation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from pac.authority.features import build_alpha_feature
from pac.authority.model import build_temporal_feature_window
from pac.evaluation.episodes import build_controller, compute_controller_action, compute_episode_metrics, desired_heading
from pac.evaluation.metrics import compute_timeseries_engineering_metrics
from pac.simulation.core import AUVSimulator


AUTHORITY_WINDOWS = (
    ("startup_0_3s", 0.0, 3.0),
    ("pre_step_3_10s", 3.0, 10.0),
    ("step_recovery_10_13s", 10.0, 13.0),
    ("post_step_13_20p9s", 13.0, 20.9),
)


def blend_actions_with_predicted_alpha(
        primary_action,
        authority_action,
        alpha: float) -> tuple[np.ndarray, dict[str, float]]:
    """Blend two normalized actions with a bounded authority coefficient."""
    primary = np.asarray(primary_action, dtype=float).reshape(6)
    authority = np.asarray(authority_action, dtype=float).reshape(6)
    bounded_alpha = float(np.clip(float(alpha), 0.0, 1.0))
    action = (1.0 - bounded_alpha) * primary + bounded_alpha * authority
    return np.clip(action, -1.0, 1.0), {
        "authority_alpha": bounded_alpha,
        "authority_active": float(bounded_alpha > 1.0e-4),
    }


def filter_authority_alpha(
        raw_alpha: float,
        previous_alpha: float,
        smoothing: float = 0.0,
        rate_limit: float = 1.0,
        deadband: float = 0.0) -> float:
    """Apply the archived causal smoothing and rate-limit order."""
    raw = float(np.clip(raw_alpha, 0.0, 1.0))
    previous = float(np.clip(previous_alpha, 0.0, 1.0))
    smooth = float(np.clip(smoothing, 0.0, 0.999))
    candidate = smooth * previous + (1.0 - smooth) * raw
    limit = max(0.0, float(rate_limit))
    if limit < 1.0:
        candidate = previous + float(np.clip(candidate - previous, -limit, limit))
    if candidate < float(deadband):
        candidate = 0.0
    return float(np.clip(candidate, 0.0, 1.0))


def alpha_bias_at_time(t: float, schedule: dict[str, float]) -> float:
    """Return the configured SSPO bias for the current formal window."""
    time_value = float(t)
    for name, start, end in AUTHORITY_WINDOWS:
        if float(start) <= time_value < float(end):
            return float(schedule.get(name, 0.0))
    return 0.0


def calibrate_alpha_with_bias_schedule(
        alpha: float,
        t: float,
        bias_schedule: dict[str, float] | None) -> float:
    """Add an SSPO window bias and clip the result to ``[0, 1]``."""
    value = float(alpha)
    if bias_schedule:
        value += alpha_bias_at_time(t, bias_schedule)
    return float(np.clip(value, 0.0, 1.0))


def run_predictive_alpha_episode(
        model,
        scenario: int,
        seed: int,
        steps: int,
        mass_scale_xy: float,
        damping_scale_xy: float,
        current_amplitude_scale: float,
        current_frequency_scale: float,
        vertical_current: float,
        primary_controller: str,
        authority_controller: str,
        feature_mode: str,
        initial_position_std: float = 0.0,
        initial_velocity_std: float = 0.0,
        alpha_bias_schedule: dict[str, float] | None = None,
        alpha_gain: float = 1.0,
        alpha_threshold: float = 0.0,
        vehicle_profile: str = "real_10kg_v1",
        thruster_layout: str = "real_10kg_x",
        alpha_smoothing: float = 0.0,
        alpha_rate_limit: float = 1.0,
        alpha_deadband: float = 0.0,
        policy_architecture: str = "transformer",
        history_len: int = 16,
        save_ts: bool = False) -> dict:
    """Evaluate PAC while preserving the archived formal-v2 metric timing."""
    if policy_architecture != "transformer":
        raise ValueError("formal PAC evaluation requires transformer architecture")
    environment = AUVSimulator(
        scenario=int(scenario),
        max_steps=int(steps),
        mass_scale_xy=float(mass_scale_xy),
        damping_scale_xy=float(damping_scale_xy),
        current_amplitude_scale=float(current_amplitude_scale),
        current_frequency_scale=float(current_frequency_scale),
        initial_position_std=float(initial_position_std),
        initial_velocity_std=float(initial_velocity_std),
        vertical_current=float(vertical_current),
        vehicle_profile=str(vehicle_profile),
        thruster_layout=str(thruster_layout),
    )
    primary_name, primary = build_controller(primary_controller)
    authority_name, authority = build_controller(authority_controller)
    for controller in (primary, authority):
        controller.reset()
        controller.set_trajectory3d(True)
    environment.reset(seed=int(seed))
    model.eval()

    episode_end = float(steps) * environment.dynamics.dt
    errors, energies, actions, thruster_forces = [], [], [], []
    headings, desired_headings, steps_out, times = [], [], [], []
    rolls, pitches, desired_rolls, desired_pitches = [], [], [], []
    xs, ys, zs, target_zs, z_errors = [], [], [], [], []
    active_values, alpha_values, raw_values = [], [], []
    history = []
    previous_alpha = 0.0
    done = False
    while not done:
        dynamics = environment.dynamics
        t = environment.current_step * dynamics.dt
        target = environment._get_target(t)
        current = environment.privileged_state[:3]
        primary_action = compute_controller_action(
            primary, primary_name, target, dynamics.eta, dynamics.nu, t, dynamics.dt, current
        )
        authority_action = compute_controller_action(
            authority, authority_name, target, dynamics.eta, dynamics.nu, t, dynamics.dt, current
        )
        feature = build_alpha_feature(
            target,
            dynamics.eta,
            dynamics.nu,
            primary_action,
            authority_action,
            current[:2],
            t,
            episode_end,
            feature_mode,
        )
        history.append(feature)
        model_input = torch.from_numpy(
            build_temporal_feature_window(history, history_len)
        ).unsqueeze(0)
        with torch.no_grad():
            alpha_raw = float(model(model_input).item())
        alpha = calibrate_alpha_with_bias_schedule(alpha_raw, t, alpha_bias_schedule)
        alpha = float(np.clip(alpha * float(alpha_gain), 0.0, 1.0))
        if alpha < float(alpha_threshold):
            alpha = 0.0
        alpha = filter_authority_alpha(
            alpha,
            previous_alpha,
            smoothing=alpha_smoothing,
            rate_limit=alpha_rate_limit,
            deadband=alpha_deadband,
        )
        action, blend_info = blend_actions_with_predicted_alpha(
            primary_action,
            authority_action,
            alpha,
        )

        steps_out.append(int(environment.current_step))
        times.append(float(t))
        rolls.append(float(dynamics.eta[3]))
        pitches.append(float(dynamics.eta[4]))
        headings.append(float(dynamics.eta[5]))
        desired_rolls.append(float(target[3]))
        desired_pitches.append(float(target[4]))
        desired_headings.append(float(desired_heading(t)))
        xs.append(float(dynamics.eta[0]))
        ys.append(float(dynamics.eta[1]))
        zs.append(float(dynamics.eta[2]))
        target_zs.append(float(target[2]))
        active_values.append(float(blend_info["authority_active"]))
        alpha_values.append(float(blend_info["authority_alpha"]))
        raw_values.append(float(alpha_raw))

        # Formal v2 records the pre-step state alongside the post-step error.
        done, step_info = environment.step(action)
        errors.append(float(step_info["dist_error"]))
        energies.append(float(step_info["energy"]))
        z_errors.append(float(step_info["z_error"]))
        actions.append(np.asarray(action, dtype=float).copy())
        if "thruster_forces" in step_info:
            thruster_forces.append(np.asarray(step_info["thruster_forces"], dtype=float).copy())
        previous_alpha = float(alpha)

    metrics = compute_episode_metrics(
        errors,
        energies,
        actions,
        headings,
        desired_headings,
        environment.dynamics.dt,
        z_errors=z_errors,
    )
    metrics.update({
        "primary_controller": primary_name,
        "authority_controller": authority_name,
        "authority_until": -1.0,
        "blend_duration": -1.0,
        "blend_curve": "predictive_alpha_sspo" if alpha_bias_schedule else "predictive_alpha",
        "authority_active_fraction": float(np.mean(active_values)),
        "authority_alpha_mean": float(np.mean(alpha_values)),
        "authority_alpha_std": float(np.std(alpha_values)),
        "alpha_raw_mean": float(np.mean(raw_values)),
        "alpha_uncertainty_std": 0.0,
        "policy_architecture": "transformer",
        "history_len": int(history_len),
        "trajectory3d": True,
        "start_time": 0.0,
        "current_amplitude_scale": float(current_amplitude_scale),
        "current_frequency_scale": float(current_frequency_scale),
        "initial_position_std": float(initial_position_std),
        "initial_velocity_std": float(initial_velocity_std),
        "vertical_current": float(vertical_current),
        "vehicle_profile": str(vehicle_profile),
        "action_mode": "thruster",
        "thruster_layout": str(thruster_layout),
    })
    if alpha_bias_schedule:
        metrics["alpha_bias_schedule_json"] = json.dumps(alpha_bias_schedule, sort_keys=True)
        metrics["alpha_bias_abs_mean"] = float(np.mean(np.abs(list(alpha_bias_schedule.values()))))

    timeseries = pd.DataFrame({
        "step": steps_out,
        "time": times,
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
        "authority_active": active_values,
        "authority_alpha": alpha_values,
        "alpha_raw": raw_values,
        "alpha_uncertainty": np.zeros(len(raw_values)),
    })
    action_array = np.asarray(actions, dtype=float)
    for index in range(action_array.shape[1]):
        timeseries[f"action_{index}"] = action_array[:, index]
    if thruster_forces:
        force_array = np.asarray(thruster_forces, dtype=float)
        for index in range(force_array.shape[1]):
            timeseries[f"thruster_force_{index}"] = force_array[:, index]
    timeseries["vehicle_profile"] = str(vehicle_profile)
    timeseries["action_mode"] = "thruster"
    timeseries["thruster_layout"] = str(thruster_layout)
    metrics.update(compute_timeseries_engineering_metrics(timeseries, dt=environment.dynamics.dt))
    if save_ts:
        metrics["ts"] = timeseries
    return metrics


def scalar_metrics(metrics: dict) -> dict:
    return {
        key: value
        for key, value in metrics.items()
        if key != "ts" and np.isscalar(value)
    }
