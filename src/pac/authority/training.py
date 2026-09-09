"""Oracle data collection and Transformer training for formal PAC runs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import sys

from pac.authority.evaluation import blend_actions_with_predicted_alpha
from pac.authority.features import build_alpha_feature, wrap_angle
from pac.authority.model import train_alpha_model, train_alpha_model_v3
from pac.authority.oracle import (
    OracleSettings,
    choose_rollout_oracle_alpha,
    validate_formal_oracle_contract,
)
from pac.authority.dataset import episode_fingerprint
from pac.evaluation.episode_spec import build_episode_spec
from pac.evaluation.episodes import build_controller, compute_controller_action
from pac.controllers.presets import build_v3_controller_pair
from pac.simulation.core import AUVSimulator
from pac.simulation.observations import CausalCurrentEstimator


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
                predicted_eta, predicted_nu = environment.dynamics.predict_step(
                    np.asarray(eta0, dtype=float).copy(),
                    np.asarray(nu0, dtype=float).copy(),
                    wrench,
                    current,
                )
                for _ in range(max(0, int(oracle_horizon_steps) - 1)):
                    predicted_eta, predicted_nu = environment.dynamics.predict_step(
                        predicted_eta, predicted_nu, wrench, current
                    )
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


def _normalize_collection_profile(profile: str | None) -> str:
    normalized = "" if profile is None else str(profile).strip().lower()
    return "short" if not normalized or normalized == "short" else normalized


normalize_oracle_profile = _normalize_collection_profile


def _collection_plan(config, profile: str) -> list[tuple[str, int, int]]:
    profile = _normalize_collection_profile(profile)
    if profile == "short":
        return [("train", 11000, 20), ("val", 12000, 20)]
    if profile == "formal":
        steps = int(config.environment.steps)
        return [
            ("train", int(seed), steps)
            for seed in config.training.oracle_train_seeds
        ] + [
            ("val", int(seed), steps)
            for seed in config.training.oracle_val_seeds
        ]
    raise ValueError("profile must be 'short' or 'formal'")


def collect_teacher_dataset_v3(config, profile: str = "formal"):
    """Collect the true-MPC v3 rollout teacher dataset once per episode."""
    profile = _normalize_collection_profile(profile)
    if profile in {"short", "formal"}:
        validate_formal_oracle_contract(config.oracle)
    plans = _collection_plan(config, profile)
    dt = float(config.environment.dt)
    settings = OracleSettings(
        horizon=config.oracle.horizon,
        alpha_grid=config.oracle.alpha_grid,
        xy_weight=config.oracle.xy_weight,
        z_weight=config.oracle.z_weight,
        heading_weight=config.oracle.heading_weight,
        control_delta_weight=config.oracle.control_delta_weight,
        saturation_weight=config.oracle.saturation_weight,
        terminal_scale=config.oracle.terminal_scale,
    )
    features: list[np.ndarray] = []
    labels: list[float] = []
    rows: list[dict[str, object]] = []
    for split, seed, steps in plans:
        # A new simulator and fresh controller pair per episode keep warm-start
        # and actuator state out of the next seed's teacher labels.
        # Formal collection evaluates all configured scenarios for each seed.
        scenarios = (1,) if profile == "short" else tuple(config.environment.scenarios)
        for scenario in scenarios:
            spec = build_episode_spec(scenario, seed, steps, dt)
            environment = AUVSimulator(
                scenario=scenario,
                max_steps=steps,
                mass_scale_xy=float(config.environment.mass_scale_xy),
                damping_scale_xy=float(config.environment.damping_scale_xy),
                current_amplitude_scale=float(config.environment.current_amplitude_scale),
                current_frequency_scale=float(config.environment.current_frequency_scale),
                vertical_current=float(config.environment.vertical_current),
                vehicle_profile=str(config.environment.vehicle_profile),
                thruster_layout=str(config.environment.thruster_layout),
                dt=dt,
                actuator_command_min=float(config.actuator.command_min),
                actuator_command_max=float(config.actuator.command_max),
                actuator_max_delta_per_step=float(config.actuator.max_delta_per_step),
                max_force=float(config.actuator.max_force_n),
            )
            environment.reset(episode_spec=spec)
            primary_name = config.controller.primary
            authority_name = config.controller.authority
            primary, authority = build_v3_controller_pair(config)
            for controller in (primary, authority):
                controller.reset()
                controller.set_trajectory3d(True)
            estimator = CausalCurrentEstimator(
                spec.current_delay_steps,
                spec.current_estimation_noise,
            )
            estimator.reset(environment.privileged_state[:3])
            fingerprint = episode_fingerprint(spec)
            episode_end = float(steps) * dt
            previous_applied = np.zeros(6, dtype=float)
            previous_plan_generation: int | None = None
            done = False
            while not done:
                dynamics = environment.dynamics
                step = int(environment.current_step)
                stamp = float(step * dt)
                target = environment._get_target(stamp)
                true_current = np.asarray(
                    environment._current_for_dynamics(environment._generate_current(stamp)),
                    dtype=float,
                )
                estimated_current = estimator.estimate(true_current, step)
                primary_action = compute_controller_action(
                    primary,
                    primary_name,
                    target,
                    dynamics.eta,
                    dynamics.nu,
                    stamp,
                    dt,
                    estimated_current,
                )
                authority_action = compute_controller_action(
                    authority,
                    authority_name,
                    target,
                    dynamics.eta,
                    dynamics.nu,
                    stamp,
                    dt,
                    estimated_current,
                )
                plan = getattr(authority, "last_plan", None)
                if callable(plan):
                    plan = plan()
                telemetry = getattr(authority, "last_telemetry", None)
                if telemetry is None:
                    telemetry = getattr(authority, "telemetry", {})
                telemetry = telemetry if isinstance(telemetry, dict) else {}
                plan_valid = False
                if plan is not None:
                    try:
                        plan_array = np.asarray(plan, dtype=float)
                        plan_valid = (
                            plan_array.ndim == 2
                            and plan_array.shape[1] == 6
                            and plan_array.shape[0] == int(config.mpc.horizon)
                            and np.isfinite(plan_array).all()
                        )
                    except (TypeError, ValueError, OverflowError):
                        plan_valid = False
                if (
                    not plan_valid
                    or not bool(telemetry.get("accepted", False))
                    or str(telemetry.get("fallback_mode", "")) != "none"
                ):
                    raise RuntimeError(
                        "fresh accepted MPC plan required "
                        f"episode={spec.episode_uid} step={step}: "
                        f"plan={'present' if plan_valid else 'missing/invalid'} "
                        f"accepted={telemetry.get('accepted', False)} "
                        f"fallback_mode={telemetry.get('fallback_mode', '')}"
                    )
                try:
                    plan_generation = int(telemetry["plan_generation"])
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise RuntimeError(
                        f"fresh accepted MPC plan required episode={spec.episode_uid} "
                        f"step={step}: missing plan_generation"
                    ) from exc
                expected_generation = (
                    1 if previous_plan_generation is None else previous_plan_generation + 1
                )
                if plan_generation != expected_generation:
                    raise RuntimeError(
                        f"fresh accepted MPC plan required episode={spec.episode_uid} "
                        f"step={step}: plan_generation={plan_generation} "
                        f"expected={expected_generation}"
                    )
                previous_plan_generation = plan_generation

                decision = choose_rollout_oracle_alpha(
                    current_eta=dynamics.eta,
                    current_nu=dynamics.nu,
                    t=stamp,
                    previous_applied=previous_applied,
                    primary_controller=primary,
                    mpc_controller=authority,
                    mpc_plan=plan,
                    simulator=environment,
                    current_provider=lambda future_t, env=environment: env._current_for_dynamics(
                        env._generate_current(float(future_t))
                    ),
                    actuator_limits=environment.actuator.limits,
                    settings=settings,
                )
                feature = build_alpha_feature(
                    target,
                    dynamics.eta,
                    dynamics.nu,
                    primary_action,
                    authority_action,
                    estimated_current[:2],
                    stamp,
                    episode_end,
                    "state_phase",
                )
                features.append(feature)
                labels.append(float(decision.alpha))
                action, _ = blend_actions_with_predicted_alpha(
                    primary_action,
                    authority_action,
                    decision.alpha,
                )
                done, info = environment.step(action)
                previous_applied = np.asarray(info["applied_action"], dtype=float).copy()
                rows.append({
                    "split": split,
                    "scenario": scenario,
                    "environment_seed": seed,
                    "episode_uid": spec.episode_uid,
                    "step": step,
                    "sample_time": float(info.get("sample_time", stamp + dt)),
                    "teacher_alpha": float(decision.alpha),
                    "oracle_best": float(decision.best_cost),
                    "oracle_worst": float(decision.worst_cost),
                    "episode_fingerprint": fingerprint,
                    "primary_controller": primary_name,
                    "authority_controller": authority_name,
                })
            if profile == "formal":
                print(
                    f"oracle_episode_complete split={split} episode={spec.episode_uid} "
                    f"samples={steps}",
                    file=sys.stderr,
                    flush=True,
                )
    from pac.authority.dataset import OracleDataset

    return OracleDataset(
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        pd.DataFrame(rows),
    )


__all__ = [
    "build_v3_controller_pair",
    "choose_oracle_alpha",
    "collect_teacher_dataset",
    "collect_teacher_dataset_v3",
    "normalize_oracle_profile",
    "train_alpha_model",
    "train_alpha_model_v3",
]
