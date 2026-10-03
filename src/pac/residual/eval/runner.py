"""Closed-loop episode runner for every paired evaluation policy.

The loop replicates the archived supervised evaluation semantics exactly (metric
timing, alpha stack, forced-primary handling) while driving a
:class:`~pac.residual.disturbances.DisturbanceSimulator`, so structured episodes evaluated
here reproduce the frozen supervised numbers and stochastic-disturbance episodes
become directly comparable.  Two parity tests pin this contract:

* ``v3_transformer`` mode vs ``run_predictive_alpha_episode``;
* fixed-controller modes vs ``run_fixed_controller_episode``.

Policies return the *raw* alpha (before gain/threshold/smoothing) and the
runner applies the shared safety stack, or they run in ``direct_action``
mode (fixed controllers) where the controller output reaches the actuator
unblended, matching the supervised fixed-controller evaluation path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch

from pac.authority.evaluation import (
    alpha_bias_at_time,
    blend_actions_with_predicted_alpha,
    filter_authority_alpha,
)
from pac.authority.features import build_alpha_feature
from pac.authority.model import build_temporal_feature_window, load_alpha_model_checkpoint
from pac.evaluation.episodes import (
    build_controller,
    compute_controller_action,
    compute_episode_metrics,
    desired_heading,
)
from pac.evaluation.metrics import compute_timeseries_engineering_metrics
from pac.evaluation.seeds import episode_uid as make_episode_uid
from pac.simulation.observations import CausalCurrentEstimator
from pac.residual.disturbances import PairedEpisodeSpec, make_disturbance_simulator
from pac.residual.collector import assemble_transition_feature


@dataclass(frozen=True)
class WMAdvice:
    """World-model candidate-rollout advice for one decision step."""

    best_alpha: float
    best_cost: float
    cost_spread: float
    uncertainty: float
    gate_open: bool
    compute_seconds: float


@dataclass
class StepContext:
    """Everything a policy may condition on at one decision step."""

    step: int
    stamp: float
    episode_end: float
    target: np.ndarray
    eta: np.ndarray
    nu: np.ndarray
    primary_action: np.ndarray
    authority_action: np.ndarray
    estimated_current: np.ndarray
    true_current: np.ndarray
    telemetry: dict
    plan: np.ndarray                  # (20, 6) padded snapshot
    forced_primary: bool
    feature: np.ndarray               # 24-dim supervised alpha feature at this step
    feature_window: np.ndarray        # (L, 24) window ending at this step
    wm_window: np.ndarray             # (L_wm, 32) world-model window ending t-1
    previous_applied: np.ndarray
    previous_alpha: float
    wm: WMAdvice | None = None


class ResidualPolicy(Protocol):
    """Minimal policy contract consumed by :func:`run_residual_policy_episode`."""

    uses_wm: bool
    direct_action: bool

    def reset(self, spec: PairedEpisodeSpec, episode_end: float) -> None: ...

    def select_alpha(self, context: StepContext) -> float: ...


class ConstantAlphaPolicy:
    """Fixed blending coefficient (use ``direct_action`` for pure controllers)."""

    uses_wm = False
    direct_action = False

    def __init__(self, alpha: float):
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("constant alpha must be in [0, 1]")
        self.alpha = float(alpha)

    def reset(self, spec: PairedEpisodeSpec, episode_end: float) -> None:
        del spec, episode_end

    def select_alpha(self, context: StepContext) -> float:
        del context
        return self.alpha


class FixedControllerPolicy(ConstantAlphaPolicy):
    """Pass one controller's action through unblended (supervised fixed-controller path)."""

    def __init__(self, which: str):
        if which not in ("primary", "authority"):
            raise ValueError("fixed controller must be 'primary' or 'authority'")
        super().__init__(0.0 if which == "primary" else 1.0)
        self.direct_action = True


class TransformerAlphaPolicy:
    """Frozen supervised backbone producing the raw alpha, optional SSPO bias."""

    uses_wm = False
    direct_action = False

    def __init__(
            self,
            model,
            history_len: int,
            *,
            alpha_bias_schedule: dict[str, float] | None = None):
        self.model = model
        self.history_len = int(history_len)
        self.alpha_bias_schedule = alpha_bias_schedule
        self.model.eval()

    def reset(self, spec: PairedEpisodeSpec, episode_end: float) -> None:
        del spec, episode_end

    def select_alpha(self, context: StepContext) -> float:
        with torch.no_grad():
            model_input = torch.from_numpy(context.feature_window).unsqueeze(0)
            alpha_raw = float(self.model(model_input).item())
        if self.alpha_bias_schedule:
            alpha_raw += alpha_bias_at_time(
                context.stamp, self.alpha_bias_schedule
            )
        return alpha_raw


def build_backbone_policy(config, model_seed: int, *,
                          alpha_bias_schedule=None) -> TransformerAlphaPolicy:
    """Load one frozen supervised backbone checkpoint as a v4 policy."""
    checkpoint_path = (
        Path(config.backbone.checkpoint_dir)
        / f"pac_train_seed_{int(model_seed)}"
        / "checkpoint.pt"
    )
    model, metadata = load_alpha_model_checkpoint(
        checkpoint_path,
        expected_feature_mode=config.authority_model.feature_mode,
    )
    if int(metadata["history_len"]) != int(config.authority_model.history_len):
        raise ValueError("backbone history_len does not match authority_model config")
    return TransformerAlphaPolicy(
        model,
        int(metadata["history_len"]),
        alpha_bias_schedule=alpha_bias_schedule,
    )


def _pad_plan(plan_array: np.ndarray, horizon: int) -> np.ndarray:
    plan = np.asarray(plan_array, dtype=float)
    if plan.ndim == 1 and plan.size == 6:
        plan = plan.reshape(1, 6)
    if plan.ndim != 2 or plan.shape[1] != 6 or plan.shape[0] < 1:
        return np.zeros((horizon, 6), dtype=float)
    if plan.shape[0] >= horizon:
        return plan[:horizon].copy()
    padded = np.zeros((horizon, 6), dtype=float)
    padded[:plan.shape[0]] = plan
    return padded


def run_residual_policy_episode(
        policy: ResidualPolicy,
        spec: PairedEpisodeSpec,
        config,
        *,
        wm_computer=None,
        save_ts: bool = True,
        transition_sink=None) -> dict[str, Any]:
    """Run one paired v4 episode and return v3-compatible metrics.

    ``transition_sink(context, step_info, next_state, alpha)`` — optional
    per-step callback invoked after the environment step for RL training
    collection; it receives the pre-step decision context, the simulator
    info dict, the post-step state ``[eta, nu]``, and the applied (filtered)
    alpha.
    """
    env_config = config.environment
    environment = make_disturbance_simulator(
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
        spec.current_delay_steps,
        spec.current_estimation_noise,
    )
    estimator.reset(environment.privileged_state[:3])

    steps = int(spec.steps)
    episode_end = float(steps) * environment.dynamics.dt
    policy.reset(spec, episode_end)
    wm_history_len = int(config.world_model.history_len)
    context_base = np.array([
        float(env_config.mass_scale_xy),
        float(env_config.damping_scale_xy),
        float(env_config.current_amplitude_scale),
        float(env_config.current_frequency_scale),
    ], dtype=float)

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
    wm_advice_flags, wm_uncertainties = [], []
    feature_history: list[np.ndarray] = []
    wm_feature_history: list[np.ndarray] = []
    previous_alpha = 0.0
    previous_applied = np.zeros(6, dtype=float)
    plan_horizon = int(config.oracle.horizon)
    refresh_steps = int(config.wm_authority.refresh_steps)
    needs_wm_window = bool(getattr(policy, "uses_wm", False)) or wm_computer is not None

    done = False
    while not done:
        dynamics = environment.dynamics
        step = int(environment.current_step)
        stamp = float(step * dynamics.dt)
        target = environment._get_target(stamp)
        pre_eta = np.asarray(dynamics.eta, dtype=float).copy()
        true_current = np.asarray(
            environment._current_for_dynamics(environment._generate_current(stamp)),
            dtype=float,
        ).copy()
        estimated_current = estimator.estimate(true_current, step)
        state = np.concatenate([
            np.asarray(dynamics.eta, dtype=float),
            np.asarray(dynamics.nu, dtype=float),
        ])

        primary_action = compute_controller_action(
            primary, primary_name, target, dynamics.eta, dynamics.nu,
            stamp, dynamics.dt, estimated_current,
        )
        authority_action = compute_controller_action(
            authority, authority_name, target, dynamics.eta, dynamics.nu,
            stamp, dynamics.dt, estimated_current,
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
        plan = _pad_plan(authority.last_plan, plan_horizon)

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
        feature_window = build_temporal_feature_window(
            feature_history, int(config.authority_model.history_len)
        )
        wm_window = np.zeros((wm_history_len, 32), dtype=np.float32)
        if needs_wm_window and wm_feature_history:
            recent = wm_feature_history[-wm_history_len:]
            wm_window[-len(recent):] = np.asarray(recent, dtype=np.float32)

        wm_advice: WMAdvice | None = None
        if wm_computer is not None and (
                refresh_steps <= 1 or step % refresh_steps == 0):
            wm_advice = wm_computer.advise(
                state=state,
                est_current=estimated_current,
                context=context_base,
                scenario_id=environment.scenario,
                wm_window=wm_window,
                previous_applied=previous_applied,
                plan=plan,
                stamp=stamp,
            )

        context = StepContext(
            step=step,
            stamp=stamp,
            episode_end=episode_end,
            target=target,
            eta=np.asarray(dynamics.eta, dtype=float).copy(),
            nu=np.asarray(dynamics.nu, dtype=float).copy(),
            primary_action=np.asarray(primary_action, dtype=float).copy(),
            authority_action=np.asarray(authority_action, dtype=float).copy(),
            estimated_current=np.asarray(estimated_current, dtype=float).copy(),
            true_current=true_current,
            telemetry=telemetry,
            plan=plan,
            forced_primary=authority_forced_primary,
            feature=feature,
            feature_window=feature_window,
            wm_window=wm_window,
            previous_applied=previous_applied.copy(),
            previous_alpha=previous_alpha,
            wm=wm_advice,
        )

        # Direct-RL policies expose select_action (6-dim thruster command,
        # no alpha stack, no forced-primary override); fixed controllers
        # keep the legacy direct path below.
        select_action = getattr(policy, "select_action", None)
        if select_action is not None:
            alpha_raw = 0.0
            action = np.asarray(select_action(context), dtype=float).reshape(6)
            action = np.clip(action, -1.0, 1.0)
            alpha = 0.0
        else:
            alpha_raw = float(policy.select_alpha(context))
            if policy.direct_action:
                action = np.asarray(
                    primary_action if policy.alpha == 0.0 else authority_action,
                    dtype=float,
                ).reshape(6)
                action = np.clip(action, -1.0, 1.0)
                alpha = 0.0 if policy.alpha == 0.0 else 1.0
            else:
                alpha = float(np.clip(alpha_raw * float(config.authority_model.alpha_gain), 0.0, 1.0))
                if alpha < 0.0:
                    alpha = 0.0
                alpha = filter_authority_alpha(
                    alpha,
                    previous_alpha,
                    smoothing=float(config.authority_model.alpha_smoothing),
                    rate_limit=float(config.authority_model.alpha_rate_limit),
                    deadband=float(config.authority_model.alpha_deadband),
                )
        if authority_forced_primary and select_action is None:
            alpha = 0.0
            action = np.asarray(primary_action, dtype=float).reshape(6)
            action = np.clip(action, -1.0, 1.0)
        elif not policy.direct_action:
            action, blend_info = blend_actions_with_predicted_alpha(
                primary_action, authority_action, alpha,
            )

        active_values.append(float(alpha > 1.0e-4))
        alpha_values.append(float(alpha))
        raw_values.append(float(alpha_raw))
        forced_primary_values.append(bool(authority_forced_primary))
        wm_advice_flags.append(float(wm_advice is not None and wm_advice.gate_open))
        wm_uncertainties.append(
            float(wm_advice.uncertainty) if wm_advice is not None else float("nan")
        )

        pre_step = int(environment.current_step)
        done, step_info = environment.step(action)
        sample_time = float(step_info.get("sample_time", stamp + dynamics.dt))
        applied = np.asarray(step_info["applied_action"], dtype=float)

        row_target = environment._get_target(sample_time)
        row_eta = np.asarray(dynamics.eta, dtype=float).copy()
        row_error = row_target - row_eta
        for index in (3, 4, 5):
            row_error[index] = float(
                (row_error[index] + np.pi) % (2 * np.pi) - np.pi
            )
        row_dist_error = float(np.linalg.norm(row_error[:3]))
        row_z_error = float(row_error[2])
        row_desired_heading = float(desired_heading(sample_time))

        steps_out.append(pre_step)
        times.append(float(sample_time))
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
        actions.append(applied.copy())
        requested_actions.append(
            np.asarray(step_info["requested_action"], dtype=float).copy()
        )
        true_currents.append(true_current.copy())
        estimated_currents.append(np.asarray(estimated_current, dtype=float).copy())
        amplitude_clipped_fractions.append(
            float(step_info.get("actuator_amplitude_clipped_fraction", 0.0))
        )
        rate_limited_fractions.append(
            float(step_info.get("actuator_rate_limited_fraction", 0.0))
        )
        if "thruster_forces" in step_info:
            thruster_forces.append(
                np.asarray(step_info["thruster_forces"], dtype=float).copy()
            )

        if needs_wm_window:
            wm_feature_history.append(assemble_transition_feature(
                state,
                np.asarray(step_info["requested_action"], dtype=float),
                applied,
                estimated_current,
                context_base,
                environment.scenario,
            ))
        if transition_sink is not None:
            transition_sink(
                context,
                step_info,
                np.concatenate([
                    np.asarray(dynamics.eta, dtype=float),
                    np.asarray(dynamics.nu, dtype=float),
                ]),
                float(alpha),
            )
        previous_alpha = float(alpha)
        previous_applied = applied.copy()

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
        "blend_curve": "v4_policy",
        "authority_active_fraction": float(np.mean(active_values)),
        "authority_alpha_mean": float(np.mean(alpha_values)),
        "authority_alpha_std": float(np.std(alpha_values)),
        "authority_forced_primary": bool(any(forced_primary_values)),
        "authority_forced_primary_fraction": float(np.mean(forced_primary_values)),
        "alpha_raw_mean": float(np.mean(raw_values)),
        "alpha_uncertainty_std": 0.0,
        "policy_architecture": "transformer",
        "history_len": int(config.authority_model.history_len),
        "trajectory3d": True,
        "start_time": 0.0,
        "current_amplitude_scale": float(env_config.current_amplitude_scale),
        "current_frequency_scale": float(env_config.current_frequency_scale),
        "initial_position_std": float(env_config.eval_initial_position_std),
        "initial_velocity_std": float(env_config.eval_initial_velocity_std),
        "vertical_current": float(env_config.vertical_current),
        "vehicle_profile": str(env_config.vehicle_profile),
        "action_mode": "thruster",
        "thruster_layout": str(env_config.thruster_layout),
        "episode_uid": str(spec.episode_uid),
        "environment_seed": int(spec.episode_seed),
        "family": str(spec.family),
        "aligned_metrics": True,
        "actuator_rate_limited_fraction_mean": (
            float(np.mean(rate_limited_fractions)) if rate_limited_fractions else 0.0
        ),
        "actuator_rate_limit_episode_mean": (
            float(np.mean(rate_limited_fractions)) if rate_limited_fractions else 0.0
        ),
        "solver_fallback_step_fraction": (
            float(np.mean(solver_fallbacks)) if solver_fallbacks else 0.0
        ),
        "solver_deadline_miss_step_fraction": (
            float(np.mean(solver_deadlines)) if solver_deadlines else 0.0
        ),
        "wm_gate_open_fraction": (
            float(np.mean(wm_advice_flags)) if wm_advice_flags else 0.0
        ),
    })

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
        "alpha_uncertainty": wm_uncertainties,
        "wm_gate_open": wm_advice_flags,
    })
    target_array = np.asarray(target_states, dtype=float)
    for index, name in enumerate(
            ("target_x", "target_y", "target_z", "target_roll", "target_pitch", "target_yaw")):
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
    timeseries["vehicle_profile"] = str(env_config.vehicle_profile)
    timeseries["action_mode"] = "thruster"
    timeseries["thruster_layout"] = str(env_config.thruster_layout)
    metrics.update(
        compute_timeseries_engineering_metrics(timeseries, dt=environment.dynamics.dt)
    )
    if save_ts:
        metrics["ts"] = timeseries
    return metrics


__all__ = [
    "ConstantAlphaPolicy",
    "FixedControllerPolicy",
    "StepContext",
    "TransformerAlphaPolicy",
    "ResidualPolicy",
    "WMAdvice",
    "build_backbone_policy",
    "run_residual_policy_episode",
]
