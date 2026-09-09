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
from pac.evaluation.seeds import episode_uid as make_episode_uid
from pac.simulation.core import AUVSimulator
from pac.simulation.observations import CausalCurrentEstimator


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
        save_ts: bool = False,
        episode_spec=None,
        dt: float = 0.01,
        actuator_max_delta_per_step: float | None = None,
        aligned_metrics: bool = False) -> dict:
    """Evaluate PAC while preserving the archived formal-v2 metric timing."""
    if policy_architecture != "transformer":
        raise ValueError("formal PAC evaluation requires transformer architecture")
    aligned = bool(aligned_metrics or episode_spec is not None)
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
        dt=float(dt),
        actuator_max_delta_per_step=actuator_max_delta_per_step,
    )
    primary_name, primary = build_controller(primary_controller)
    authority_name, authority = build_controller(authority_controller)
    for controller in (primary, authority):
        controller.reset()
        controller.set_trajectory3d(True)
    environment.reset(
        seed=None if episode_spec is not None else int(seed),
        episode_spec=episode_spec,
    )
    estimator = None
    if episode_spec is not None:
        estimator = CausalCurrentEstimator(
            episode_spec.current_delay_steps,
            episode_spec.current_estimation_noise,
        )
        estimator.reset(environment.privileged_state[:3])
    model.eval()

    episode_end = float(steps) * environment.dynamics.dt
    errors, energies, actions, thruster_forces = [], [], [], []
    requested_actions = []
    true_currents, estimated_currents = [], []
    amplitude_clipped_fractions, rate_limited_fractions = [], []
    headings, desired_headings, steps_out, times = [], [], [], []
    sample_times = []
    target_states = []
    rolls, pitches, desired_rolls, desired_pitches = [], [], [], []
    xs, ys, zs, target_zs, z_errors = [], [], [], [], []
    active_values, alpha_values, raw_values = [], [], []
    forced_primary_values = []
    solver_fallbacks, solver_deadlines = [], []
    history = []
    previous_alpha = 0.0
    done = False
    while not done:
        dynamics = environment.dynamics
        t = environment.current_step * dynamics.dt
        target = environment._get_target(t)
        pre_eta = np.asarray(dynamics.eta, dtype=float).copy()
        if episode_spec is not None:
            true_current = np.asarray(
                environment._current_for_dynamics(
                    environment._generate_current(t)
                ),
                dtype=float,
            ).copy()
        else:
            true_current = np.asarray(environment.privileged_state[:3], dtype=float).copy()
        if estimator is not None:
            current = estimator.estimate(true_current, environment.current_step)
        else:
            current = true_current.copy()
        primary_action = compute_controller_action(
            primary, primary_name, target, dynamics.eta, dynamics.nu, t, dynamics.dt, current
        )
        authority_action = compute_controller_action(
            authority, authority_name, target, dynamics.eta, dynamics.nu, t, dynamics.dt, current
        )
        telemetry = getattr(authority, "last_telemetry", {})
        if not isinstance(telemetry, dict):
            telemetry = {}
        solver_fallbacks.append(float(str(telemetry.get("fallback_mode", "none")) != "none"))
        solver_deadlines.append(float(bool(telemetry.get("deadline_missed", False))))
        authority_forced_primary = bool(
            getattr(authority, "force_primary_authority", False)
            or getattr(authority, "force_primary", False)
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
        if authority_forced_primary:
            alpha = 0.0
        action, blend_info = blend_actions_with_predicted_alpha(
            primary_action,
            authority_action,
            alpha,
        )

        active_values.append(float(blend_info["authority_active"]))
        alpha_values.append(float(blend_info["authority_alpha"]))
        raw_values.append(float(alpha_raw))
        forced_primary_values.append(bool(authority_forced_primary))

        pre_step = int(environment.current_step)
        done, step_info = environment.step(action)
        sample_time = float(step_info.get("sample_time", t + dynamics.dt))
        if aligned:
            row_target = environment._get_target(sample_time)
            row_eta = np.asarray(dynamics.eta, dtype=float).copy()
            row_error = row_target - row_eta
            for index in (3, 4, 5):
                row_error[index] = float((row_error[index] + np.pi) % (2.0 * np.pi) - np.pi)
            row_dist_error = float(np.linalg.norm(row_error[:3]))
            row_z_error = float(row_error[2])
            row_desired_heading = float(desired_heading(sample_time))
            row_time = sample_time
        else:
            row_target = target
            row_eta = pre_eta
            row_dist_error = float(step_info["dist_error"])
            row_z_error = float(step_info["z_error"])
            row_desired_heading = float(desired_heading(t))
            row_time = t
        steps_out.append(pre_step)
        times.append(float(row_time))
        sample_times.append(sample_time)
        rolls.append(float(row_eta[3]))
        pitches.append(float(row_eta[4]))
        headings.append(float(row_eta[5]))
        desired_rolls.append(float(row_target[3]))
        desired_pitches.append(float(row_target[4]))
        desired_headings.append(row_desired_heading)
        xs.append(float(row_eta[0]))
        ys.append(float(row_eta[1]))
        zs.append(float(row_eta[2]))
        target_zs.append(float(row_target[2]))
        target_states.append(np.asarray(row_target, dtype=float).copy())
        errors.append(row_dist_error)
        energies.append(float(step_info["energy"]))
        z_errors.append(row_z_error)
        actions.append(np.asarray(step_info["applied_action"], dtype=float).copy())
        requested_actions.append(np.asarray(step_info["requested_action"], dtype=float).copy())
        true_currents.append(true_current.copy())
        estimated_currents.append(np.asarray(current, dtype=float).copy())
        amplitude_clipped_fractions.append(float(step_info.get("actuator_amplitude_clipped_fraction", 0.0)))
        rate_limited_fractions.append(float(step_info.get("actuator_rate_limited_fraction", 0.0)))
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
        "authority_forced_primary": bool(any(forced_primary_values)),
        "authority_forced_primary_fraction": float(np.mean(forced_primary_values)),
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
        "episode_uid": (
            str(episode_spec.episode_uid)
            if episode_spec is not None
            else make_episode_uid(int(scenario), int(seed))
        ),
        "environment_seed": int(
            episode_spec.episode_seed if episode_spec is not None else seed
        ),
        "aligned_metrics": aligned,
        "actuator_rate_limited_fraction_mean": float(np.mean(rate_limited_fractions)) if rate_limited_fractions else 0.0,
        "actuator_rate_limit_episode_mean": float(np.mean(rate_limited_fractions)) if rate_limited_fractions else 0.0,
        "solver_fallback_step_fraction": float(np.mean(solver_fallbacks)) if solver_fallbacks else 0.0,
        "solver_deadline_miss_step_fraction": float(np.mean(solver_deadlines)) if solver_deadlines else 0.0,
    })
    if alpha_bias_schedule:
        metrics["alpha_bias_schedule_json"] = json.dumps(alpha_bias_schedule, sort_keys=True)
        metrics["alpha_bias_abs_mean"] = float(np.mean(np.abs(list(alpha_bias_schedule.values()))))

    timeseries = pd.DataFrame({
        "step": steps_out,
        "time": times,
        "sample_time": sample_times,
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
        "authority_forced_primary": forced_primary_values,
        "alpha_raw": raw_values,
        "alpha_uncertainty": np.zeros(len(raw_values)),
    })
    target_array = np.asarray(target_states, dtype=float)
    for index, name in enumerate(("target_x", "target_y", "target_z", "target_roll", "target_pitch", "target_yaw")):
        timeseries[name] = target_array[:, index]
    action_array = np.asarray(actions, dtype=float)
    for index in range(action_array.shape[1]):
        timeseries[f"action_{index}"] = action_array[:, index]
        timeseries[f"applied_action_{index}"] = action_array[:, index]
    requested_array = np.asarray(requested_actions, dtype=float)
    for index in range(requested_array.shape[1]):
        timeseries[f"requested_action_{index}"] = requested_array[:, index]
    true_current_array = np.asarray(true_currents, dtype=float)
    estimated_current_array = np.asarray(estimated_currents, dtype=float)
    for index in range(3):
        timeseries[f"true_current_{index}"] = true_current_array[:, index]
        timeseries[f"estimated_current_{index}"] = estimated_current_array[:, index]
    timeseries["actuator_amplitude_clipped_fraction"] = amplitude_clipped_fractions
    timeseries["actuator_rate_limited_fraction"] = rate_limited_fractions
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
