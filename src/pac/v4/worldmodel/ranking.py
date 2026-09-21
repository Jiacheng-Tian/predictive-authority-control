"""Candidate-alpha rollout evaluation for the v4 world model.

At a decision step ``t`` the evaluator replays the exact authority-selection
problem from the frozen v3 oracle: given the state, the active MPC plan, the
primary SMC expert, and an actuator, score every candidate alpha on the
11-point grid over an H-step constant-alpha rollout.  Three scorers are
provided:

* ``true``: the v3 physics rollout oracle under the *realized* current
  trajectory recorded during collection (perfect-current preview, matching
  the v3 teacher contract);
* ``wm``: the physics + residual ensemble rolling forward with its own
  auxiliary current forecasts;
* ``physics_persistence``: pure physics with the last estimated current held
  constant (the deployable physics-only control).

Ranking agreement is measured with Spearman correlation, top-1 selection
accuracy, and normalized ranking regret on the true costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from pac.authority.oracle import OracleSettings, _tracking_cost
from pac.evaluation.episodes import build_controller, compute_controller_action
from pac.simulation.actuators import ActuatorLimits, SharedActuator
from pac.simulation.dynamics import AUVDynamics
from pac.simulation.thrusters import build_thruster_layout
from pac.v4.worldmodel.data import assemble_wm_feature
from pac.v4.worldmodel.model import EnsembleDynamicsModel
from pac.v4.worldmodel.torch_dynamics import TorchAUVDynamics


@dataclass(frozen=True)
class RankingWindow:
    """One decision-step replay extracted from a transition dataset."""

    episode_uid: str
    family: str
    split: str
    behavior: str
    scenario_id: int
    environment_seed: int
    row: int                 # global dataset row of decision step t
    step: int                # step index t within the episode
    initial_state: np.ndarray  # (12,)
    window: np.ndarray         # (L, 32) rows t-L+1 .. t-1, zero-padded front
    previous_applied: np.ndarray  # (6,) applied action of step t-1
    est_current: np.ndarray    # (3,)
    context: np.ndarray        # (4,) nominal scales used by feature assembly
    plan: np.ndarray           # (H, 6) active MPC plan
    realized_currents: np.ndarray  # (H, 3) true currents for steps t..t+H-1
    sample_time: float         # pre-step time of decision t


@dataclass(frozen=True)
class RankingResult:
    """Agreement metrics between one predicted and the true cost ranking."""

    true_costs: np.ndarray
    predicted_costs: np.ndarray
    scorer: str

    @property
    def true_best_alpha(self) -> float:
        return float(_GRID[int(np.argmin(self.true_costs))])

    @property
    def predicted_best_alpha(self) -> float:
        return float(_GRID[int(np.argmin(self.predicted_costs))])

    @property
    def spearman(self) -> float:
        correlation = _spearman(self.true_costs, self.predicted_costs)
        return float(correlation)

    @property
    def top1_correct(self) -> bool:
        return int(np.argmin(self.predicted_costs)) == int(np.argmin(self.true_costs))

    @property
    def regret(self) -> float:
        """Normalized regret of taking the predicted best alpha on true costs."""
        best = float(np.min(self.true_costs))
        worst = float(np.max(self.true_costs))
        chosen = float(self.true_costs[int(np.argmin(self.predicted_costs))])
        denominator = worst - best
        if denominator <= 1.0e-12:
            return 0.0
        return float((chosen - best) / denominator)


GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_GRID = GRID


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("spearman inputs must be equal-length arrays of size >= 2")
    if np.std(a) <= 1.0e-15 or np.std(b) <= 1.0e-15:
        return float("nan")
    rank_a = pd.Series(a).rank().to_numpy(dtype=float)
    rank_b = pd.Series(b).rank().to_numpy(dtype=float)
    correlation = np.corrcoef(rank_a, rank_b)[0, 1]
    return float(correlation)


def select_ranking_windows(
        dataset,
        *,
        split: str,
        windows_per_episode: int,
        horizon: int,
        history_len: int) -> list[RankingWindow]:
    """Sample decision steps with accepted plans from the dataset."""
    metadata = dataset.metadata
    features = np.asarray(dataset.features, dtype=np.float32)
    states = np.asarray(dataset.state, dtype=np.float32)
    applied = np.asarray(dataset.applied, dtype=np.float32)
    est_current = np.asarray(dataset.est_current, dtype=np.float32)
    true_current = np.asarray(dataset.true_current, dtype=np.float32)
    plans = np.asarray(dataset.plan, dtype=np.float32)
    windows: list[RankingWindow] = []
    uids = metadata["episode_uid"].tolist()
    start = 0
    for index in range(1, len(uids) + 1):
        if index != len(uids) and uids[index] == uids[start]:
            continue
        episode_slice = slice(start, index)
        episode_meta = metadata.iloc[episode_slice]
        split_values = episode_meta["split"].astype(str).to_numpy()
        if split_values[0] != str(split):
            start = index
            continue
        accepted = np.flatnonzero(
            episode_meta["plan_accepted"].astype(bool).to_numpy()
        )
        accepted = np.array([
            row for row in accepted
            if int(episode_meta["step"].to_numpy()[row]) >= 1
            and row + horizon <= episode_meta.shape[0]
        ])
        if accepted.size == 0:
            start = index
            continue
        if accepted.size > int(windows_per_episode):
            positions = np.linspace(0, accepted.size - 1, int(windows_per_episode))
            accepted = accepted[np.unique(np.round(positions).astype(int))]
        episode_context = np.asarray(dataset.context, dtype=np.float32)[start:index]
        for local_row in accepted:
            global_row = start + int(local_row)
            step = int(episode_meta["step"].to_numpy()[local_row])
            window_start = global_row - int(history_len)
            window = np.zeros((int(history_len), features.shape[1]), dtype=np.float32)
            if window_start >= 0:
                window[:] = features[window_start:global_row]
            else:
                window[-global_row:] = features[0:global_row]
            horizon_end = min(global_row + int(horizon), start + episode_meta.shape[0])
            realized = true_current[global_row:horizon_end]
            if realized.shape[0] < int(horizon):
                padding = np.repeat(realized[-1:], int(horizon) - realized.shape[0], axis=0)
                realized = np.concatenate([realized, padding], axis=0)
            windows.append(RankingWindow(
                episode_uid=str(uids[start]),
                family=str(episode_meta["family"].to_numpy()[local_row]),
                split=str(split),
                behavior=str(episode_meta["behavior"].to_numpy()[local_row]),
                scenario_id=int(episode_meta["scenario_id"].to_numpy()[local_row]),
                environment_seed=int(episode_meta["environment_seed"].to_numpy()[local_row]),
                row=global_row,
                step=step,
                initial_state=states[global_row].copy(),
                window=window,
                previous_applied=(
                    applied[global_row - 1].copy() if global_row >= 1
                    else np.zeros(6, dtype=np.float32)
                ),
                est_current=est_current[global_row].copy(),
                context=episode_context[local_row].copy(),
                plan=plans[global_row].copy(),
                realized_currents=realized,
                sample_time=float(step) * 0.01,
            ))
        start = index
    return windows


class CandidateRolloutEvaluator:
    """Score the 11-point alpha grid with the v3 oracle objective."""

    def __init__(
            self,
            config,
            *,
            device: str = "cpu"):
        self.settings = OracleSettings(
            horizon=int(config.oracle.horizon),
            alpha_grid=tuple(float(value) for value in config.oracle.alpha_grid),
            xy_weight=float(config.oracle.xy_weight),
            z_weight=float(config.oracle.z_weight),
            heading_weight=float(config.oracle.heading_weight),
            control_delta_weight=float(config.oracle.control_delta_weight),
            saturation_weight=float(config.oracle.saturation_weight),
            terminal_scale=float(config.oracle.terminal_scale),
        )
        self.dt = float(config.environment.dt)
        self.layout = build_thruster_layout(
            config.environment.thruster_layout,
            max_force=float(config.actuator.max_force_n),
        )
        self.limits = ActuatorLimits(
            command_min=float(config.actuator.command_min),
            command_max=float(config.actuator.command_max),
            max_delta_per_step=float(config.actuator.max_delta_per_step),
        )
        self.oracle_dynamics = AUVDynamics(
            vehicle_profile=config.environment.vehicle_profile,
            dt=self.dt,
        )
        self.torch_dynamics = TorchAUVDynamics(
            vehicle_profile=config.environment.vehicle_profile,
            dt=self.dt,
            device=device,
        )
        self.primary = build_controller(config.controller.primary)[1]
        self.primary.set_trajectory3d(True)
        self.device = torch.device(device)
        from pac.v4.batched import BatchedSMC

        self.batched_smc = BatchedSMC(self.primary)
        self._allocation = np.asarray(self.layout.allocation_matrix, dtype=float)
        self._allocation_t = np.ascontiguousarray(self._allocation.T)
        self._max_force = float(self.layout.max_force)

    # -- shared rollout plumbing ------------------------------------------
    def _target_at(self, stamp: float) -> np.ndarray:
        x = 3.0 * np.sin(0.3 * stamp)
        y = 1.5 * np.sin(0.6 * stamp)
        z = 0.8 * np.sin(0.2 * stamp)
        return np.array([x, y, z, 0.0, 0.0, np.arctan2(0.9 * np.cos(0.6 * stamp), 0.9 * np.cos(0.3 * stamp))])

    def _to_wrench(self, action: np.ndarray) -> np.ndarray:
        forces = self.layout.normalized_action_to_forces(np.asarray(action, dtype=float))
        return self.layout.forces_to_wrench(forces)

    def true_oracle_costs(self, window: RankingWindow) -> np.ndarray:
        """Perfect-preview physics rollout costs on the 11-point grid."""
        settings = self.settings
        realized = np.asarray(window.realized_currents, dtype=float)
        t0 = float(window.sample_time)
        costs: list[float] = []
        for alpha in settings.alpha_grid:
            eta = np.asarray(window.initial_state[:6], dtype=float).copy()
            nu = np.asarray(window.initial_state[6:], dtype=float).copy()
            actuator = SharedActuator(self.limits)
            actuator._previous_applied = np.asarray(window.previous_applied, dtype=float).copy()
            total = 0.0
            for step in range(settings.horizon):
                stamp = t0 + step * self.dt
                target = self._target_at(stamp)
                current = realized[step]
                primary_action = compute_controller_action(
                    self.primary, "real10kg_smc_steady", target, eta, nu, stamp, self.dt, current
                )
                authority_action = window.plan[min(step, window.plan.shape[0] - 1)]
                requested = (1.0 - alpha) * primary_action + alpha * authority_action
                previous_for_delta = actuator._previous_applied.copy()
                actuator_step = actuator.apply(requested)
                applied = np.asarray(actuator_step.applied, dtype=float)
                delta = applied - previous_for_delta
                saturation = float(np.mean((requested - actuator_step.amplitude_clipped) ** 2))
                if self.limits.max_delta_per_step is not None:
                    saturation += float(np.mean(
                        np.maximum(
                            np.abs(actuator_step.amplitude_clipped - previous_for_delta)
                            - self.limits.max_delta_per_step,
                            0.0,
                        ) ** 2
                    ))
                tau = self._to_wrench(applied)
                eta, nu = self.oracle_dynamics.predict_step(eta, nu, tau, current)
                next_target = self._target_at(stamp + self.dt)
                total += _tracking_cost(settings, next_target, eta)
                total += settings.control_delta_weight * float(np.mean(delta ** 2))
                total += settings.saturation_weight * saturation
            total += settings.terminal_scale * _tracking_cost(settings, next_target, eta)
            costs.append(float(total))
        return np.asarray(costs, dtype=float)

    def single_model_costs(self, model: Any, window: RankingWindow) -> np.ndarray:
        """Rollout costs for one predictor with per-candidate batch of ``A``.

        ``model`` must expose ``predict_delta(windows (A, L, F)) ->
        (delta_state (A, 12), delta_current (A, 2))``.
        """
        return self._single_model_rollout(model, window)

    def _single_model_rollout(self, model: Any, window: RankingWindow) -> np.ndarray:
        """Batched candidate rollout: one vectorized SMC call, vectorized
        actuator/blend/tracking, and torch RK4 physics per step."""
        from pac.v4.batched import batched_actuator_apply

        settings = self.settings
        alphas = np.asarray(settings.alpha_grid, dtype=float)
        candidate_count = alphas.size
        t0 = float(window.sample_time)
        state_np = np.repeat(
            np.asarray(window.initial_state, dtype=float)[None, :], candidate_count, axis=0
        )
        current_forecast = np.repeat(
            np.asarray(window.est_current, dtype=float)[None, :], candidate_count, axis=0
        )
        previous_applied = np.repeat(
            np.asarray(window.previous_applied, dtype=float)[None, :], candidate_count, axis=0
        )
        window_tensor = torch.tensor(
            np.repeat(np.asarray(window.window)[None, :, :], candidate_count, axis=0),
            dtype=torch.float32,
        )
        costs = np.zeros(candidate_count, dtype=float)
        plans = np.repeat(
            np.asarray(window.plan, dtype=float)[None, :, :], candidate_count, axis=0
        )
        context = np.asarray(window.context, dtype=float)
        scenario_id = window.scenario_id
        xy_weight = float(settings.xy_weight)
        z_weight = float(settings.z_weight)
        heading_weight = float(settings.heading_weight)

        for step in range(settings.horizon):
            stamp = t0 + step * self.dt
            target = self._target_at(stamp)
            next_target = self._target_at(stamp + self.dt)
            primary_actions = self.batched_smc.compute_batch(
                target, state_np[:, :6], state_np[:, 6:], stamp, current_forecast
            )
            plan_step = plans[:, min(step, plans.shape[1] - 1)]
            requested = np.clip(
                (1.0 - alphas)[:, None] * primary_actions
                + alphas[:, None] * plan_step,
                -1.0, 1.0,
            )
            applied, _amplitude = batched_actuator_apply(
                requested, previous_applied,
                command_min=self.limits.command_min,
                command_max=self.limits.command_max,
                max_delta_per_step=self.limits.max_delta_per_step,
            )
            deltas = applied - previous_applied
            saturation = np.mean((requested - applied) ** 2, axis=1)
            if self.limits.max_delta_per_step is not None:
                saturation += np.mean(
                    np.maximum(
                        np.abs(deltas) - self.limits.max_delta_per_step, 0.0
                    ) ** 2,
                    axis=1,
                )
            previous_applied = applied.copy()
            wrench = (applied * self._max_force) @ self._allocation_t
            physics_eta, physics_nu = self.torch_dynamics.predict_step(
                torch.tensor(state_np[:, :6]),
                torch.tensor(state_np[:, 6:]),
                torch.tensor(wrench),
                torch.tensor(current_forecast),
            )
            features = np.stack([
                assemble_wm_feature(
                    state_np[index], requested[index], applied[index],
                    current_forecast[index], context, scenario_id,
                )
                for index in range(candidate_count)
            ])
            window_tensor = torch.cat([
                window_tensor[:, 1:, :],
                torch.tensor(features[:, None, :], dtype=torch.float32),
            ], dim=1)
            with torch.no_grad():
                delta_state, delta_current = model.predict_delta(window_tensor)
            eta_next = physics_eta.numpy() + delta_state.numpy()[:, :6]
            nu_next = physics_nu.numpy() + delta_state.numpy()[:, 6:]
            state_np = np.concatenate([eta_next, nu_next], axis=1)
            current_forecast = current_forecast.copy()
            current_forecast[:, :2] += delta_current.numpy()
            error = next_target[None, :] - eta_next
            yaw_error = (error[:, 5] + np.pi) % (2.0 * np.pi) - np.pi
            tracking = (
                xy_weight * (error[:, 0] ** 2 + error[:, 1] ** 2)
                + z_weight * error[:, 2] ** 2
                + heading_weight * yaw_error ** 2
            )
            costs += tracking
            costs += settings.control_delta_weight * np.mean(deltas ** 2, axis=1)
            costs += settings.saturation_weight * saturation
        terminal_target = self._target_at(t0 + settings.horizon * self.dt)
        terminal_error = terminal_target[None, :] - state_np[:, :6]
        terminal_yaw = (terminal_error[:, 5] + np.pi) % (2.0 * np.pi) - np.pi
        terminal = (
            xy_weight * (terminal_error[:, 0] ** 2 + terminal_error[:, 1] ** 2)
            + z_weight * terminal_error[:, 2] ** 2
            + heading_weight * terminal_yaw ** 2
        )
        costs += settings.terminal_scale * terminal
        return costs


class _ZeroModel:
    """Deployable physics-only baseline: zero residual, frozen current."""

    def predict_delta(self, windows: torch.Tensor):
        batch = windows.shape[0]
        zeros_state = torch.zeros(batch, 12)
        zeros_current = torch.zeros(batch, 2)
        return zeros_state, zeros_current


class EnsembleAdapter:
    """Adapt an :class:`EnsembleDynamicsModel` to (M, A) cost rollouts."""

    def __init__(self, ensemble: EnsembleDynamicsModel):
        self.ensemble = ensemble

    def costs(self, evaluator: CandidateRolloutEvaluator, window: RankingWindow) -> np.ndarray:
        return np.stack([
            evaluator.single_model_costs(member, window)
            for member in self.ensemble.members
        ], axis=0)

    def mean_costs(self, evaluator: CandidateRolloutEvaluator, window: RankingWindow) -> np.ndarray:
        return self.costs(evaluator, window).mean(axis=0)


def physics_persistence_costs(evaluator: CandidateRolloutEvaluator, window: RankingWindow) -> np.ndarray:
    """Deployable physics-only control: zero residual, persistence current."""
    return evaluator.single_model_costs(_ZeroModel(), window)


def ensemble_candidate_costs(
        evaluator: CandidateRolloutEvaluator,
        ensemble: EnsembleDynamicsModel,
        window: RankingWindow) -> np.ndarray:
    return EnsembleAdapter(ensemble).costs(evaluator, window)


def ranking_row(result: RankingResult, **identifiers) -> dict[str, Any]:
    row = dict(identifiers)
    row.update({
        "scorer": result.scorer,
        "spearman": result.spearman,
        "top1_correct": float(result.top1_correct),
        "regret": result.regret,
        "true_best_alpha": result.true_best_alpha,
        "predicted_best_alpha": result.predicted_best_alpha,
        "true_best_cost": float(np.min(result.true_costs)),
        "true_worst_cost": float(np.max(result.true_costs)),
        "predicted_cost_spread": float(np.max(result.predicted_costs) - np.min(result.predicted_costs)),
    })
    return row


__all__ = [
    "GRID",
    "CandidateRolloutEvaluator",
    "EnsembleAdapter",
    "RankingResult",
    "RankingWindow",
    "ensemble_candidate_costs",
    "physics_persistence_costs",
    "ranking_row",
    "select_ranking_windows",
]
