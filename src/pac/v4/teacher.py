"""Oracle teacher collection over v4 disturbance episodes.

Mirrors the deployment loop (``run_v4_policy_episode``) step for step --
same controllers, same causal current estimator, same 24-dim feature --
but the blending alpha comes from the privileged rollout oracle and is
executed, so the recorded features stay on the teacher's closed-loop
distribution.  Per step we keep the full alpha-grid cost profile for
soft-label training.

Current handling inside the oracle's H-step lookahead follows the
simulator contract: analytic-current families (structured,
actuator_delay, mass_damping, estimation_delay, random_freq) are polled
exactly; step-advanced stochastic families (ou_current, fast_ou,
colored_noise) are frozen at the current step (documented
approximation, <=18% OU decay over the 20-step horizon at train-range
theta).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pac.authority.evaluation import blend_actions_with_predicted_alpha
from pac.authority.features import build_alpha_feature
from pac.authority.model import build_temporal_feature_window
from pac.authority.oracle import OracleSettings, choose_rollout_oracle_alpha
from pac.evaluation.episodes import build_controller, compute_controller_action
from pac.simulation.observations import CausalCurrentEstimator
from pac.v4.disturbances import (
    STRUCTURED_FAMILY,
    build_v4_episode_spec,
    draw_disturbance_params,
    make_v4_simulator,
)


@dataclass(frozen=True)
class TeacherPlanItem:
    """One teacher episode to collect."""

    family: str
    scenario_id: int
    seed: int
    steps: int
    difficulty: str          # free-form tag recorded in metadata
    params: dict | None      # explicit params override train-range draw


def oracle_settings_from_config(config) -> OracleSettings:
    o = config.oracle
    return OracleSettings(
        horizon=int(o.horizon),
        alpha_grid=tuple(float(a) for a in o.alpha_grid),
        xy_weight=float(o.xy_weight),
        z_weight=float(o.z_weight),
        heading_weight=float(o.heading_weight),
        control_delta_weight=float(o.control_delta_weight),
        saturation_weight=float(o.saturation_weight),
        terminal_scale=float(o.terminal_scale),
    )


def collect_v4_teacher_episode(
        config,
        item: TeacherPlanItem) -> dict[str, Any]:
    """Run one teacher episode and return features/labels/costs/metadata."""
    env_config = config.environment
    steps = int(item.steps)
    if item.params is not None:
        params = dict(item.params)
    elif item.family == STRUCTURED_FAMILY:
        params = None
    else:
        ranges = {
            name: (bound.low, bound.high)
            for name, bound in config.disturbances.profile_ranges(
                item.family, "train"
            ).items()
        }
        params = draw_disturbance_params(item.family, item.seed, ranges)
    spec = build_v4_episode_spec(
        item.family, item.scenario_id, item.seed, steps,
        float(env_config.dt), disturbance_params=params,
    )
    environment = make_v4_simulator(
        spec,
        mass_scale_xy=float(env_config.mass_scale_xy),
        damping_scale_xy=float(env_config.damping_scale_xy),
        current_amplitude_scale=float(env_config.current_amplitude_scale),
        current_frequency_scale=float(env_config.current_frequency_scale),
        vertical_current=float(env_config.vertical_current),
        vehicle_profile=str(env_config.vehicle_profile),
        thruster_layout=str(env_config.thruster_layout),
        dt=float(env_config.dt),
        actuator_command_min=float(config.actuator.command_min),
        actuator_command_max=float(config.actuator.command_max),
        actuator_max_delta_per_step=float(config.actuator.max_delta_per_step),
        max_force=float(config.actuator.max_force_n),
    )
    primary_name, primary = build_controller(config.controller.primary)
    authority_name, authority = build_controller(config.controller.authority)
    for controller in (primary, authority):
        controller.reset()
        controller.set_trajectory3d(True)
    environment.reset(episode_spec=spec)
    estimator = CausalCurrentEstimator(
        spec.current_delay_steps, spec.current_estimation_noise,
    )
    estimator.reset(environment.privileged_state[:3])

    oracle = oracle_settings_from_config(config)
    episode_end = float(steps) * environment.dynamics.dt
    history_len = int(config.authority_model.history_len)
    grid = np.asarray(oracle.alpha_grid, dtype=float)

    features, labels, costs = [], [], []
    rows = []
    feature_history: list[np.ndarray] = []
    previous_applied = np.zeros(6, dtype=float)
    previous_alpha = 0.0
    plan_fallbacks = 0
    last_plan = None
    done = False
    while not done:
        dynamics = environment.dynamics
        step = int(environment.current_step)
        stamp = float(step * dynamics.dt)
        target = environment._get_target(stamp)
        true_current = np.asarray(
            environment._current_for_dynamics(environment._generate_current(stamp)),
            dtype=float,
        ).copy()
        estimated_current = estimator.estimate(true_current, step)
        primary_action = compute_controller_action(
            primary, primary_name, target, dynamics.eta, dynamics.nu,
            stamp, dynamics.dt, estimated_current,
        )
        authority_action = compute_controller_action(
            authority, authority_name, target, dynamics.eta, dynamics.nu,
            stamp, dynamics.dt, estimated_current,
        )
        # Resolve the MPC plan with the same tolerance the evaluation
        # runner applies; the oracle requires an explicit array.
        plan = getattr(authority, "last_plan", None)
        if callable(plan):
            plan = plan()
        if plan is not None and hasattr(plan, "plan"):
            plan = plan.plan
        if plan is None:
            plan = last_plan
        if plan is None:
            plan = np.broadcast_to(
                np.asarray(authority_action, dtype=float).reshape(1, 6),
                (int(oracle.horizon), 6),
            )
            plan_fallbacks += 1
        plan = np.asarray(plan, dtype=float)
        if plan.ndim == 1 and plan.size == 6:
            plan = plan.reshape(1, 6)
        last_plan = plan.copy()
        padded = np.zeros((int(oracle.horizon), 6), dtype=float)
        rows_n = min(plan.shape[0], int(oracle.horizon))
        padded[:rows_n] = plan[:rows_n]
        decision = choose_rollout_oracle_alpha(
            current_eta=np.asarray(dynamics.eta, dtype=float),
            current_nu=np.asarray(dynamics.nu, dtype=float),
            t=stamp,
            previous_applied=previous_applied,
            primary_controller=primary,
            mpc_plan=padded,
            simulator=environment,
            settings=oracle,
        )
        alpha = float(decision.alpha)
        feature = build_alpha_feature(
            target,
            dynamics.eta,
            dynamics.nu,
            primary_action,
            authority_action,
            estimated_current[:2],
            stamp,
            episode_end,
            config.authority_model.feature_mode,
        )
        feature_history.append(feature)
        features.append(feature)
        labels.append(alpha)
        costs.append(np.asarray(decision.costs, dtype=float))
        rows.append({
            "family": item.family,
            "difficulty": item.difficulty,
            "scenario": int(item.scenario_id),
            "seed": int(item.seed),
            "step": step,
            "time": stamp,
            "teacher_alpha": alpha,
            "best_cost": float(decision.best_cost),
            "worst_cost": float(decision.worst_cost),
            "alpha_margin": float(
                np.sort(np.asarray(decision.costs))[1] - decision.best_cost
            ),
        })
        action, _info = blend_actions_with_predicted_alpha(
            primary_action, authority_action, alpha,
        )
        done, step_info = environment.step(action)
        previous_applied = np.asarray(step_info["applied_action"], dtype=float).copy()
        previous_alpha = alpha

    return {
        "features": np.asarray(features, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.float32),
        "costs": np.asarray(costs, dtype=np.float32),
        "alpha_grid": grid,
        "metadata": pd.DataFrame(rows),
        "plan_fallbacks": plan_fallbacks,
        "plan_item": item,
    }


def save_teacher_dataset(episodes: list[dict[str, Any]], out_dir: str | Path,
                         config, plan: list[TeacherPlanItem]) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    features = np.concatenate([e["features"] for e in episodes], axis=0)
    labels = np.concatenate([e["labels"] for e in episodes], axis=0)
    costs = np.concatenate([e["costs"] for e in episodes], axis=0)
    metadata = pd.concat(
        [e["metadata"].assign(episode_index=i) for i, e in enumerate(episodes)],
        ignore_index=True,
    )
    np.savez_compressed(
        out / "teacher.npz",
        features=features, labels=labels, costs=costs,
        alpha_grid=episodes[0]["alpha_grid"],
    )
    metadata.to_csv(out / "metadata.csv", index=False)
    digest = hashlib.sha256(
        np.ascontiguousarray(features).tobytes()
        + np.ascontiguousarray(labels).tobytes()
    ).hexdigest()
    manifest = {
        "config_oracle": {
            "horizon": int(config.oracle.horizon),
            "alpha_grid": [float(a) for a in config.oracle.alpha_grid],
        },
        "episodes": [
            {
                "family": p.family, "scenario_id": p.scenario_id,
                "seed": p.seed, "steps": p.steps,
                "difficulty": p.difficulty, "params": p.params,
            }
            for p in plan
        ],
        "rows": int(features.shape[0]),
        "content_hash": digest,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return out


def build_teacher_windows(features: np.ndarray, metadata: pd.DataFrame,
                          history_len: int) -> np.ndarray:
    """Zero-padded temporal windows that never cross episode boundaries."""
    windows = np.zeros((features.shape[0], history_len, features.shape[1]),
                       dtype=np.float32)
    episode_ids = metadata["episode_index"].to_numpy()
    history: list[np.ndarray] = []
    current = episode_ids[0]
    for index, feature in enumerate(features):
        if episode_ids[index] != current:
            history = []
            current = episode_ids[index]
        history.append(feature)
        window = build_temporal_feature_window(history, history_len)
        windows[index] = window
    return windows


__all__ = [
    "TeacherPlanItem",
    "build_teacher_windows",
    "collect_v4_teacher_episode",
    "oracle_settings_from_config",
    "save_teacher_dataset",
]
