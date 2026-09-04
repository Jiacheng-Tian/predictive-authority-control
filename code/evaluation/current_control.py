"""Current-version fixed-controller evaluation helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from env.auv_env import AUVTrackingEnv
from evaluation.baseline_presets import (
    build_real10kg_mpc_3d,
    build_real10kg_mpc_event,
    build_real10kg_smc,
    build_real10kg_smc_steady,
)
from evaluation.engineering_metrics import compute_timeseries_engineering_metrics

SCENARIO_NAMES = {1: "constant", 2: "sinusoidal", 3: "step_change"}


def _wrap_angle(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _desired_heading(t: float) -> float:
    return float(np.arctan2(0.9 * np.cos(0.6 * float(t)), 0.9 * np.cos(0.3 * float(t))))


def _compute_metrics(errors, energies, actions, headings, desired_headings, dt, z_errors=None) -> dict:
    errors = np.asarray(errors, dtype=float)
    energies = np.asarray(energies, dtype=float)
    actions = np.asarray(actions, dtype=float)
    headings = np.asarray(headings, dtype=float) if len(headings) else np.zeros_like(errors)
    desired = np.asarray(desired_headings, dtype=float) if len(desired_headings) else np.zeros_like(errors)
    n = max(int(len(errors)), 1)
    steady_start = int(n * 0.8)
    heading_error = np.asarray([_wrap_angle(h - d) for h, d in zip(headings, desired)], dtype=float)
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


def _make_ts(
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
        z_errors=None) -> dict:
    out = {"step": steps, "error": errors, "energy": energies, "heading": headings, "desired_heading": desired_headings, "x": xs, "y": ys}
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
    if actions:
        action_array = np.asarray(actions, dtype=float)
        for idx in range(action_array.shape[1]):
            out[f"action_{idx}"] = action_array[:, idx]
    return out


def _build_base_controller(name: str):
    controller_name = str(name or "real10kg_smc_steady").strip().lower()
    if controller_name in {"real10kg_smc", "smc"}:
        return "real10kg_smc", build_real10kg_smc()
    if controller_name in {"real10kg_smc_steady", "real10kg_steady_smc"}:
        return "real10kg_smc_steady", build_real10kg_smc_steady()
    if controller_name in {"real10kg_mpc_3d", "real10kg_mpc"}:
        return "real10kg_mpc_3d", build_real10kg_mpc_3d()
    if controller_name in {"real10kg_mpc_event", "real10kg_event_mpc", "mpc"}:
        return "real10kg_mpc_event", build_real10kg_mpc_event()
    raise ValueError(f"Unknown current-version controller: {controller_name}")


def _compute_base_action(controller, controller_name, target, eta, nu, t, dt, current):
    del controller_name
    try:
        return controller.compute(target, eta, nu, t=t, dt=dt, current_prediction=current)
    except TypeError:
        return controller.compute(target, eta, nu, t=t, dt=dt)


def run_fixed_controller_episode(
        *,
        scenario: int,
        seed: int,
        steps: int,
        mass_scale_xy: float,
        damping_scale_xy: float,
        base_controller: str,
        start_time: float = 0.0,
        trajectory3d: bool = True,
        current_amplitude_scale: float = 1.0,
        current_frequency_scale: float = 1.0,
        initial_position_std: float = 0.0,
        initial_velocity_std: float = 0.0,
        vertical_current: float = 0.0,
        vehicle_profile: str = "real_10kg_v1",
        action_mode: str = "thruster",
        thruster_layout: str = "real_10kg_x",
        save_ts: bool = False) -> dict:
    env = AUVTrackingEnv(
        scenario=int(scenario),
        max_steps=int(steps),
        trajectory3d=bool(trajectory3d),
        mass_scale_xy=float(mass_scale_xy),
        damping_scale_xy=float(damping_scale_xy),
        start_time=float(start_time),
        current_amplitude_scale=float(current_amplitude_scale),
        current_frequency_scale=float(current_frequency_scale),
        initial_position_std=float(initial_position_std),
        initial_velocity_std=float(initial_velocity_std),
        vertical_current=float(vertical_current),
        vehicle_profile=str(vehicle_profile),
        action_mode=str(action_mode),
        thruster_layout=str(thruster_layout),
    )
    canonical_name, controller = _build_base_controller(base_controller)
    obs, _ = env.reset(seed=int(seed))
    del obs
    if hasattr(controller, "reset"):
        controller.reset()
    if hasattr(controller, "set_trajectory3d"):
        controller.set_trajectory3d(bool(trajectory3d))
    errors, energies, actions = [], [], []
    headings, desired_headings, xs, ys, step_rows = [], [], [], [], []
    rolls, pitches, desired_rolls, desired_pitches = [], [], [], []
    zs, target_zs, z_errors, thruster_forces = [], [], [], []
    done = False
    while not done:
        dyn = env.dynamics
        t = env.start_time + env.current_step * dyn.dt
        target = env._get_target(t)
        action = _compute_base_action(controller, canonical_name, target, dyn.eta, dyn.nu, t, dyn.dt, env.privileged_state[:3])
        step_rows.append(int(env.current_step))
        rolls.append(float(dyn.eta[3]))
        pitches.append(float(dyn.eta[4]))
        headings.append(float(dyn.eta[5]))
        desired_rolls.append(float(target[3]))
        desired_pitches.append(float(target[4]))
        desired_headings.append(float(_desired_heading(t)))
        xs.append(float(dyn.eta[0]))
        ys.append(float(dyn.eta[1]))
        if trajectory3d:
            zs.append(float(dyn.eta[2]))
            target_zs.append(float(target[2]))
        _next_obs, _reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        errors.append(float(info["dist_error"]))
        energies.append(float(info["energy"]))
        if trajectory3d:
            z_errors.append(float(info.get("z_error", target[2] - dyn.eta[2])))
        actions.append(np.asarray(action, dtype=float).copy())
        if "thruster_forces" in info:
            thruster_forces.append(np.asarray(info["thruster_forces"], dtype=float))
    metrics = _compute_metrics(errors, energies, actions, headings, desired_headings, env.dynamics.dt, z_errors=z_errors if trajectory3d else None)
    metrics.update({
        "base_controller": canonical_name,
        "start_time": float(start_time),
        "trajectory3d": bool(trajectory3d),
        "current_amplitude_scale": float(current_amplitude_scale),
        "current_frequency_scale": float(current_frequency_scale),
        "initial_position_std": float(initial_position_std),
        "initial_velocity_std": float(initial_velocity_std),
        "vertical_current": float(vertical_current),
        "vehicle_profile": str(vehicle_profile),
        "action_mode": str(action_mode),
        "thruster_layout": str(thruster_layout),
    })
    ts = pd.DataFrame(_make_ts(
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
        zs=zs if trajectory3d else None,
        target_zs=target_zs if trajectory3d else None,
        z_errors=z_errors if trajectory3d else None,
    ))
    if thruster_forces:
        force_array = np.asarray(thruster_forces, dtype=float)
        for idx in range(force_array.shape[1]):
            ts[f"thruster_force_{idx}"] = force_array[:, idx]
    ts["vehicle_profile"] = str(vehicle_profile)
    ts["action_mode"] = str(action_mode)
    ts["thruster_layout"] = str(thruster_layout)
    ts["initial_position_std"] = float(initial_position_std)
    ts["initial_velocity_std"] = float(initial_velocity_std)
    metrics.update(compute_timeseries_engineering_metrics(ts, dt=env.dynamics.dt))
    if save_ts:
        metrics["ts"] = ts
    return metrics
