"""Current-version fixed-controller evaluation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pac.simulation.core import AUVSimulator
from pac.simulation.observations import CausalCurrentEstimator
from pac.evaluation.seeds import episode_uid as make_episode_uid
from pac.controllers.presets import (
    build_real10kg_mpc_event,
    build_real10kg_mpc_ltv_v3,
    build_real10kg_smc_steady,
)
from pac.evaluation.metrics import compute_timeseries_engineering_metrics

SCENARIO_NAMES = {1: "constant", 2: "sinusoidal", 3: "step_change"}


def wrap_angle(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def desired_heading(t: float) -> float:
    return float(np.arctan2(0.9 * np.cos(0.6 * float(t)), 0.9 * np.cos(0.3 * float(t))))


def compute_episode_metrics(errors, energies, actions, headings, desired_headings, dt, z_errors=None) -> dict:
    errors = np.asarray(errors, dtype=float)
    energies = np.asarray(energies, dtype=float)
    actions = np.asarray(actions, dtype=float)
    headings = np.asarray(headings, dtype=float) if len(headings) else np.zeros_like(errors)
    desired = np.asarray(desired_headings, dtype=float) if len(desired_headings) else np.zeros_like(errors)
    n = max(int(len(errors)), 1)
    steady_start = int(n * 0.8)
    heading_error = np.asarray([wrap_angle(h - d) for h, d in zip(headings, desired)], dtype=float)
    metrics = {
        "final_error": float(errors[-1]) if len(errors) else float("inf"),
        "success_0.5m": float(errors[-1] < 0.5) if len(errors) else 0.0,
        "success_1.0m": float(errors[-1] < 1.0) if len(errors) else 0.0,
        "success_2.0m": float(errors[-1] < 2.0) if len(errors) else 0.0,
        "mean_error": float(np.mean(errors)) if len(errors) else float("inf"),
        "max_error": float(np.max(errors)) if len(errors) else float("inf"),
        "rmse": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else float("inf"),
        "iae": float(np.sum(np.abs(errors)) * float(dt)),
        "steady_state_error": float(np.mean(errors[steady_start:])),
        "steady_state_rmse": float(np.sqrt(np.mean(errors[steady_start:] ** 2))),
        "energy": float(np.sum(energies) * float(dt)),
        "mean_thrust": float(np.mean(np.sum(actions ** 2, axis=1))) if len(actions) else 0.0,
        "control_tv": float(np.mean(np.sum(np.abs(np.diff(actions, axis=0)), axis=1))) if len(actions) > 1 else 0.0,
        "heading_rmse_deg": float(np.sqrt(np.mean(heading_error ** 2))) * 180.0 / np.pi if len(heading_error) else 0.0,
        "heading_max_deg": float(np.max(np.abs(heading_error))) * 180.0 / np.pi if len(heading_error) else 0.0,
    }
    if z_errors is not None:
        z_errors = np.asarray(z_errors, dtype=float)
        metrics["z_rmse"] = float(np.sqrt(np.mean(z_errors ** 2))) if len(z_errors) else float("inf")
        metrics["z_max_abs_error"] = float(np.max(np.abs(z_errors))) if len(z_errors) else float("inf")
        metrics["rmse_3d"] = metrics["rmse"]
    return metrics


def make_timeseries(
        steps,
        errors,
        energies,
        actions,
        headings,
        desired_headings,
        xs,
        ys,
        rolls=None,
        pitches=None,
        desired_rolls=None,
        desired_pitches=None,
        zs=None,
        target_zs=None,
        z_errors=None,
        times=None,
        requested_actions=None,
        applied_actions=None,
        true_currents=None,
        estimated_currents=None,
        amplitude_clipped_fractions=None,
        rate_limited_fractions=None) -> dict:
    out = {"step": steps, "error": errors, "energy": energies, "heading": headings, "desired_heading": desired_headings, "x": xs, "y": ys}
    if times is not None:
        out["time"] = times
        out["sample_time"] = times
    out["yaw"] = headings
    out["desired_yaw"] = desired_headings
    if rolls is not None:
        out["roll"] = rolls
    if pitches is not None:
        out["pitch"] = pitches
    if desired_rolls is not None:
        out["desired_roll"] = desired_rolls
    if desired_pitches is not None:
        out["desired_pitch"] = desired_pitches
    if zs is not None:
        out["z"] = zs
    if target_zs is not None:
        out["target_z"] = target_zs
    if z_errors is not None:
        out["z_error"] = z_errors
    action_values = actions if applied_actions is None else applied_actions
    if len(action_values):
        action_array = np.asarray(action_values, dtype=float)
        for idx in range(action_array.shape[1]):
            out[f"action_{idx}"] = action_array[:, idx]
            out[f"applied_action_{idx}"] = action_array[:, idx]
    if requested_actions is not None and len(requested_actions):
        requested_array = np.asarray(requested_actions, dtype=float)
        for idx in range(requested_array.shape[1]):
            out[f"requested_action_{idx}"] = requested_array[:, idx]
    if true_currents is not None and len(true_currents):
        current_array = np.asarray(true_currents, dtype=float)
        for idx in range(current_array.shape[1]):
            out[f"true_current_{idx}"] = current_array[:, idx]
    if estimated_currents is not None and len(estimated_currents):
        current_array = np.asarray(estimated_currents, dtype=float)
        for idx in range(current_array.shape[1]):
            out[f"estimated_current_{idx}"] = current_array[:, idx]
    if amplitude_clipped_fractions is not None:
        out["actuator_amplitude_clipped_fraction"] = amplitude_clipped_fractions
    if rate_limited_fractions is not None:
        out["actuator_rate_limited_fraction"] = rate_limited_fractions
    return out


def build_controller(name: str):
    controller_name = str(name or "real10kg_smc_steady").strip().lower()
    if controller_name == "real10kg_smc_steady":
        return "real10kg_smc_steady", build_real10kg_smc_steady()
    if controller_name == "real10kg_mpc_event":
        return "real10kg_mpc_event", build_real10kg_mpc_event()
    if controller_name == "real10kg_mpc_ltv_v3":
        return "real10kg_mpc_ltv_v3", build_real10kg_mpc_ltv_v3()
    raise ValueError(f"Unknown current-version controller: {controller_name}")


def compute_controller_action(controller, controller_name, target, eta, nu, t, dt, current):
    del controller_name
    del dt
    return controller.compute(
        target,
        eta,
        nu,
        t=t,
        current_prediction=current,
    )


def run_fixed_controller_episode(
        *,
        scenario: int,
        seed: int,
        steps: int,
        mass_scale_xy: float,
        damping_scale_xy: float,
        base_controller: str,
        current_amplitude_scale: float = 1.0,
        current_frequency_scale: float = 1.0,
        initial_position_std: float = 0.0,
        initial_velocity_std: float = 0.0,
        vertical_current: float = 0.0,
        vehicle_profile: str = "real_10kg_v1",
        thruster_layout: str = "real_10kg_x",
        save_ts: bool = False,
        episode_spec=None,
        dt: float = 0.01,
        actuator_max_delta_per_step: float | None = None,
        aligned_metrics: bool = False) -> dict:
    aligned = bool(aligned_metrics or episode_spec is not None)
    env = AUVSimulator(
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
    canonical_name, controller = build_controller(base_controller)
    env.reset(
        seed=None if episode_spec is not None else int(seed),
        episode_spec=episode_spec,
    )
    estimator = None
    if episode_spec is not None:
        estimator = CausalCurrentEstimator(
            episode_spec.current_delay_steps,
            episode_spec.current_estimation_noise,
        )
        estimator.reset(env.privileged_state[:3])
    if hasattr(controller, "reset"):
        controller.reset()
    controller.set_trajectory3d(True)
    errors, energies, actions = [], [], []
    requested_actions = []
    true_currents, estimated_currents = [], []
    sample_times = []
    amplitude_clipped_fractions, rate_limited_fractions = [], []
    headings, desired_headings, xs, ys, step_rows = [], [], [], [], []
    target_states = []
    rolls, pitches, desired_rolls, desired_pitches = [], [], [], []
    zs, target_zs, z_errors, thruster_forces = [], [], [], []
    done = False
    while not done:
        dyn = env.dynamics
        t = env.current_step * dyn.dt
        target = env._get_target(t)
        pre_eta = np.asarray(dyn.eta, dtype=float).copy()
        if episode_spec is not None:
            true_current = np.asarray(
                env._current_for_dynamics(env._generate_current(t)),
                dtype=float,
            ).copy()
        else:
            true_current = np.asarray(env.privileged_state[:3], dtype=float).copy()
        if estimator is not None:
            estimated_current = estimator.estimate(true_current, env.current_step)
        else:
            estimated_current = true_current.copy()
        action = np.asarray(
            compute_controller_action(
                controller,
                canonical_name,
                target,
                dyn.eta,
                dyn.nu,
                t,
                dyn.dt,
                estimated_current,
            ),
            dtype=float,
        ).reshape(6)
        done, info = env.step(action)
        sample_time = float(info.get("sample_time", t + dyn.dt))
        if aligned:
            row_target = env._get_target(sample_time)
            row_eta = np.asarray(dyn.eta, dtype=float).copy()
            row_error = row_target - row_eta
            for index in (3, 4, 5):
                row_error[index] = wrap_angle(row_error[index])
            row_dist_error = float(np.linalg.norm(row_error[:3]))
            row_z_error = float(row_error[2])
            row_heading = float(row_eta[5])
            row_desired_heading = float(desired_heading(sample_time))
        else:
            row_target = target
            row_eta = pre_eta
            row_dist_error = float(info["dist_error"])
            row_z_error = float(info.get("z_error", target[2] - dyn.eta[2]))
            row_heading = float(dyn.eta[5])
            row_desired_heading = float(desired_heading(t))
        step_rows.append(int(env.current_step - 1))
        sample_times.append(sample_time if aligned else t)
        rolls.append(float(row_eta[3]))
        pitches.append(float(row_eta[4]))
        headings.append(row_heading)
        desired_rolls.append(float(row_target[3]))
        desired_pitches.append(float(row_target[4]))
        desired_headings.append(row_desired_heading)
        xs.append(float(row_eta[0]))
        ys.append(float(row_eta[1]))
        zs.append(float(row_eta[2]))
        target_zs.append(float(row_target[2]))
        target_states.append(np.asarray(row_target, dtype=float).copy())
        errors.append(row_dist_error)
        energies.append(float(info["energy"]))
        z_errors.append(row_z_error)
        applied_action = np.asarray(info["applied_action"], dtype=float).copy()
        actions.append(applied_action)
        requested_actions.append(np.asarray(info["requested_action"], dtype=float).copy())
        true_currents.append(true_current.copy())
        estimated_currents.append(np.asarray(estimated_current, dtype=float).copy())
        amplitude_clipped_fractions.append(float(info.get("actuator_amplitude_clipped_fraction", 0.0)))
        rate_limited_fractions.append(float(info.get("actuator_rate_limited_fraction", 0.0)))
        if "thruster_forces" in info:
            thruster_forces.append(np.asarray(info["thruster_forces"], dtype=float))
    metrics = compute_episode_metrics(errors, energies, actions, headings, desired_headings, env.dynamics.dt, z_errors=z_errors)
    metrics.update({
        "base_controller": canonical_name,
        "start_time": 0.0,
        "trajectory3d": True,
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
    })
    ts = pd.DataFrame(make_timeseries(
        step_rows,
        errors,
        energies,
        actions,
        headings,
        desired_headings,
        xs,
        ys,
        rolls=rolls,
        pitches=pitches,
        desired_rolls=desired_rolls,
        desired_pitches=desired_pitches,
        zs=zs,
        target_zs=target_zs,
        z_errors=z_errors,
        times=sample_times,
        requested_actions=requested_actions,
        applied_actions=actions,
        true_currents=true_currents,
        estimated_currents=estimated_currents,
        amplitude_clipped_fractions=amplitude_clipped_fractions,
        rate_limited_fractions=rate_limited_fractions,
    ))
    target_array = np.asarray(target_states, dtype=float)
    for index, name in enumerate(("target_x", "target_y", "target_z", "target_roll", "target_pitch", "target_yaw")):
        ts[name] = target_array[:, index]
    ts["target_z"] = target_array[:, 2]
    if thruster_forces:
        force_array = np.asarray(thruster_forces, dtype=float)
        for idx in range(force_array.shape[1]):
            ts[f"thruster_force_{idx}"] = force_array[:, idx]
    ts["vehicle_profile"] = str(vehicle_profile)
    ts["action_mode"] = "thruster"
    ts["thruster_layout"] = str(thruster_layout)
    ts["initial_position_std"] = float(initial_position_std)
    ts["initial_velocity_std"] = float(initial_velocity_std)
    metrics.update(compute_timeseries_engineering_metrics(ts, dt=env.dynamics.dt))
    if save_ts:
        metrics["ts"] = ts
    return metrics
