"""Oracle data collection and Transformer training for formal PAC runs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pac.authority.evaluation import blend_actions_with_predicted_alpha
from pac.authority.features import build_alpha_feature, wrap_angle
from pac.authority.model import train_alpha_model
from pac.evaluation.episodes import build_controller, compute_controller_action
from pac.simulation.core import AUVSimulator


def tracking_cost(
        target,
        eta,
        action,
        action_delta,
        *,
        saturation_weight: float,
        action_delta_weight: float) -> float:
    """Compute the archived short-horizon oracle objective."""
    target_array = np.asarray(target, dtype=float).reshape(6)
    eta_array = np.asarray(eta, dtype=float).reshape(6)
    error = target_array - eta_array
    error[3:6] = [wrap_angle(value) for value in error[3:6]]
    action_array = np.asarray(action, dtype=float).reshape(6)
    delta = np.asarray(action_delta, dtype=float).reshape(6)
    return (
        float(np.dot(error[:2], error[:2]))
        + 0.7 * float(error[2] ** 2)
        + 0.08 * float(error[5] ** 2)
        + float(saturation_weight)
        * float(np.mean(np.maximum(np.abs(action_array) - 0.92, 0.0) ** 2))
        + float(action_delta_weight) * float(np.mean(delta ** 2))
    )


def choose_oracle_alpha(
        *,
        target,
        eta,
        nu,
        primary_action,
        authority_action,
        current,
        action_to_wrench,
        dynamics_stepper,
        alpha_grid,
        previous_action,
        previous_alpha: float,
        action_saturation_weight: float,
        action_delta_weight: float,
        alpha_delta_weight: float) -> tuple[float, dict[str, float]]:
    """Choose the lowest-cost authority value from the configured grid."""
    grid = np.asarray(alpha_grid, dtype=float).reshape(-1)
    if grid.size == 0:
        raise ValueError("alpha_grid must contain at least one value")
    primary = np.asarray(primary_action, dtype=float).reshape(6)
    authority = np.asarray(authority_action, dtype=float).reshape(6)
    previous = np.asarray(previous_action, dtype=float).reshape(6)
    candidates = []
    for raw_alpha in grid:
        alpha = float(np.clip(raw_alpha, 0.0, 1.0))
        action, _info = blend_actions_with_predicted_alpha(primary, authority, alpha)
        predicted_eta, _predicted_nu = dynamics_stepper(
            np.asarray(eta, dtype=float).copy(),
            np.asarray(nu, dtype=float).copy(),
            action_to_wrench(action),
            current,
        )
        cost = tracking_cost(
            target,
            predicted_eta,
            action,
            action - previous,
            saturation_weight=action_saturation_weight,
            action_delta_weight=action_delta_weight,
        )
        cost += float(alpha_delta_weight) * abs(alpha - float(previous_alpha))
        candidates.append((cost, alpha))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return float(candidates[0][1]), {
        "candidate_count": int(grid.size),
        "best_cost": float(candidates[0][0]),
        "worst_cost": float(candidates[-1][0]),
        "alpha_delta_weight": float(alpha_delta_weight),
    }


def collect_teacher_dataset(
        *,
        scenarios: list[int],
        data_seeds: list[int],
        steps: int,
        mass_scale_xy: float,
        damping_scale_xy: float,
        current_amplitude_scale: float,
        current_frequency_scale: float,
        vertical_current: float,
        primary_controller: str,
        authority_controller: str,
        feature_mode: str,
        oracle_alpha_grid: list[float],
        oracle_horizon_steps: int,
        oracle_action_saturation_weight: float,
        oracle_action_delta_weight: float,
        oracle_alpha_delta_weight: float,
        vehicle_profile: str,
        thruster_layout: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Collect the fixed-start formal oracle dataset."""
    features, labels, rows = [], [], []
    episode_end = float(steps) * 0.01
    alpha_grid = np.asarray(oracle_alpha_grid, dtype=float)
    for scenario in scenarios:
        for seed in data_seeds:
            environment = AUVSimulator(
                scenario=int(scenario),
                max_steps=int(steps),
                mass_scale_xy=float(mass_scale_xy),
                damping_scale_xy=float(damping_scale_xy),
                current_amplitude_scale=float(current_amplitude_scale),
                current_frequency_scale=float(current_frequency_scale),
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
            previous_action = np.zeros(6, dtype=float)
            previous_alpha = 0.0
            done = False

            def action_to_wrench(action):
                return environment._action_to_wrench(action)[0]

            def dynamics_stepper(eta0, nu0, wrench, current):
                saved_eta = environment.dynamics.eta.copy()
                saved_nu = environment.dynamics.nu.copy()
                environment.dynamics.eta = np.asarray(eta0, dtype=float).copy()
                environment.dynamics.nu = np.asarray(nu0, dtype=float).copy()
                predicted_eta, predicted_nu = environment.dynamics.step(wrench, current)
                for _ in range(max(0, int(oracle_horizon_steps) - 1)):
                    predicted_eta, predicted_nu = environment.dynamics.step(wrench, current)
                environment.dynamics.eta = saved_eta
                environment.dynamics.nu = saved_nu
                return predicted_eta, predicted_nu

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
                oracle_target = environment._get_target(
                    t + max(1, int(oracle_horizon_steps)) * dynamics.dt
                )
                alpha, oracle = choose_oracle_alpha(
                    target=oracle_target,
                    eta=dynamics.eta,
                    nu=dynamics.nu,
                    primary_action=primary_action,
                    authority_action=authority_action,
                    current=current,
                    action_to_wrench=action_to_wrench,
                    dynamics_stepper=dynamics_stepper,
                    alpha_grid=alpha_grid,
                    previous_action=previous_action,
                    previous_alpha=previous_alpha,
                    action_saturation_weight=oracle_action_saturation_weight,
                    action_delta_weight=oracle_action_delta_weight,
                    alpha_delta_weight=oracle_alpha_delta_weight,
                )
                features.append(build_alpha_feature(
                    target,
                    dynamics.eta,
                    dynamics.nu,
                    primary_action,
                    authority_action,
                    current[:2],
                    t,
                    episode_end,
                    feature_mode,
                ))
                labels.append(alpha)
                rows.append({
                    "scenario": int(scenario),
                    "seed": int(seed),
                    "start_time": 0.0,
                    "current_amplitude_scale": float(current_amplitude_scale),
                    "current_frequency_scale": float(current_frequency_scale),
                    "step": int(environment.current_step),
                    "time": float(t),
                    "absolute_time": float(t),
                    "teacher_alpha": float(alpha),
                    "oracle_best_cost": float(oracle["best_cost"]),
                    "oracle_worst_cost": float(oracle["worst_cost"]),
                    "error_norm": float(np.linalg.norm((target - dynamics.eta)[:3])),
                    "vehicle_profile": str(vehicle_profile),
                    "action_mode": "thruster",
                    "thruster_layout": str(thruster_layout),
                })
                action, _info = blend_actions_with_predicted_alpha(
                    primary_action,
                    authority_action,
                    alpha,
                )
                done, _step_info = environment.step(action)
                previous_action = np.asarray(action, dtype=float).copy()
                previous_alpha = float(alpha)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        pd.DataFrame(rows),
    )


__all__ = ["choose_oracle_alpha", "collect_teacher_dataset", "train_alpha_model"]
