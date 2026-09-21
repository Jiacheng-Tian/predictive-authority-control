"""Training-time collection glue between the runner and the TD3 trainer."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from pac.evaluation.episodes import desired_heading
from pac.simulation.core import AUVSimulator
from pac.v4.eval.runner import run_v4_policy_episode
from pac.v4.rl.policy import ResidualAlphaPolicy, build_wm_features
from pac.v4.rl.reward import compute_step_reward


class LiveResidualPolicy:
    """Drives the runner with the current residual actor (plus exploration)."""

    uses_wm: bool
    direct_action = False

    def __init__(
            self,
            policy: ResidualAlphaPolicy,
            *,
            explore_std: float = 0.0,
            seed: int = 0):
        self.policy = policy
        self.explore_std = float(explore_std)
        self.rng = np.random.default_rng(int(seed))
        self.uses_wm = bool(policy.wm_feature_dim)
        self.last_delta = 0.0
        self.last_base = 0.0

    def reset(self, spec, episode_end: float) -> None:
        del spec, episode_end
        self.last_delta = 0.0
        self.last_base = 0.0

    def select_alpha(self, context) -> float:
        windows = torch.from_numpy(
            np.asarray(context.feature_window, dtype=np.float32)[None]
        )
        wm_features = None
        if self.policy.wm_feature_dim:
            wm_features = torch.from_numpy(build_wm_features(context.wm)[None])
        with torch.no_grad():
            base = float(self.policy.backbone_alpha(windows).item())
            delta = float(self.policy(windows, wm_features).item())
        if self.explore_std > 0.0:
            delta += float(
                self.rng.normal(0.0, self.explore_std * self.policy.delta_max)
            )
        delta = float(np.clip(delta, -self.policy.delta_max, self.policy.delta_max))
        self.last_base = base
        self.last_delta = delta
        raw = base + self.policy.lambda_blend * delta
        if not np.isfinite(raw):
            # Defensive fallback: any numerical failure in the head must
            # degrade to the frozen backbone behavior, never to NaN.
            raw = base
        return float(np.clip(raw, 0.0, 1.0))


class TransitionRecorder:
    """Runner ``transition_sink`` building v1-rewarded transitions."""

    def __init__(self, config, *, wm_feature_dim: int, delta_max: float):
        self.config = config
        self.wm_feature_dim = int(wm_feature_dim)
        self.delta_max = float(delta_max)
        self.entries: list[dict[str, Any]] = []

    def __call__(self, context, step_info, next_state, alpha: float) -> None:
        if self.entries:
            self.entries[-1]["next_window"] = np.array(
                context.feature_window, dtype=np.float32, copy=True
            )
            self.entries[-1]["next_wm_features"] = (
                build_wm_features(context.wm)
                if self.wm_feature_dim else np.zeros(0, dtype=np.float32)
            )
        sample_time = float(step_info["sample_time"])
        target = AUVSimulator._get_target(sample_time)
        position_error = float(np.linalg.norm(target[:3] - next_state[:3]))
        heading_error = abs(float(
            (desired_heading(sample_time) - next_state[5] + np.pi)
            % (2.0 * np.pi) - np.pi
        ))
        requested = np.asarray(step_info["requested_action"], dtype=float)
        applied = np.asarray(step_info["applied_action"], dtype=float)
        saturation = float(np.mean(np.abs(requested - applied) > 1.0e-12))
        telemetry = context.telemetry if isinstance(context.telemetry, dict) else {}
        reward = compute_step_reward(
            self.config.reward,
            position_error=position_error,
            heading_error=heading_error,
            applied_action=applied,
            previous_applied=context.previous_applied,
            alpha=float(alpha),
            previous_alpha=float(context.previous_alpha),
            saturation_fraction=saturation,
            deadline_missed=bool(telemetry.get("deadline_missed", False)),
            constraint_violated=position_error > 20.0,
        )
        self.entries.append({
            "window": np.array(context.feature_window, dtype=np.float32, copy=True),
            "wm_features": (
                build_wm_features(context.wm)
                if self.wm_feature_dim else np.zeros(0, dtype=np.float32)
            ),
            "action": float(np.clip(context.policy_delta, -self.delta_max, self.delta_max)),
            "reward": float(reward),
            "next_window": None,
            "next_wm_features": None,
            "done": False,
            "position_error": position_error,
        })
        self._last_window = self.entries[-1]["window"]
        self._last_wm_features = self.entries[-1]["wm_features"]

    def close(self) -> None:
        if not self.entries:
            return
        last = self.entries[-1]
        last["next_window"] = np.array(self._last_window, dtype=np.float32, copy=True)
        last["next_wm_features"] = np.array(
            self._last_wm_features, dtype=np.float32, copy=True
        )
        last["done"] = True

    @property
    def transitions(self) -> list[dict[str, Any]]:
        incomplete = [index for index, entry in enumerate(self.entries)
                      if entry["next_window"] is None]
        if incomplete:
            raise ValueError(f"transitions not closed: {len(incomplete)} entries")
        return self.entries


class _RecordingPolicyWrapper:
    """Binds the recorder to the live policy's chosen delta."""

    def __init__(self, live: LiveResidualPolicy, recorder: TransitionRecorder):
        self._live = live
        self._recorder = recorder

    def __getattr__(self, name):
        return getattr(self._live, name)

    @property
    def uses_wm(self) -> bool:
        return self._live.uses_wm

    direct_action = False

    def reset(self, spec, episode_end: float) -> None:
        self._live.reset(spec, episode_end)

    def select_alpha(self, context) -> float:
        alpha = self._live.select_alpha(context)
        context.policy_delta = self._live.last_delta
        return alpha


def collect_training_episode(
        policy: ResidualAlphaPolicy,
        spec,
        config,
        *,
        explore_std: float,
        episode_seed: int,
        wm_computer=None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one training episode and return v1-rewarded transitions."""
    recorder = TransitionRecorder(
        config,
        wm_feature_dim=int(policy.wm_feature_dim),
        delta_max=float(policy.delta_max),
    )
    live = LiveResidualPolicy(
        policy, explore_std=explore_std, seed=int(episode_seed)
    )
    wrapper = _RecordingPolicyWrapper(live, recorder)
    metrics = run_v4_policy_episode(
        wrapper,
        spec,
        config,
        wm_computer=wm_computer if policy.wm_feature_dim else None,
        save_ts=False,
        transition_sink=recorder,
    )
    recorder.close()
    return recorder.transitions, metrics


__all__ = [
    "LiveResidualPolicy",
    "TransitionRecorder",
    "collect_training_episode",
]
