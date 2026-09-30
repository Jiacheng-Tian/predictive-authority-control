"""Constrained TD3 trainer for the v4 residual alpha policy.

Standard TD3 (twin critics, target-policy smoothing, delayed actor
updates) with two v4-specific constraints:

* behavior regularization: the actor loss adds ``behavior_reg * mean(delta^2)``
  keeping residuals near the frozen-backbone behavior;
* the actor update only touches parameters with ``requires_grad`` (the
  backbone stays frozen in round 1).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class ReplayBatch:
    windows: torch.Tensor        # (B, L, 24)
    wm_features: torch.Tensor    # (B, W) or (B, 0)
    actions: torch.Tensor        # (B,)
    rewards: torch.Tensor        # (B,)
    next_windows: torch.Tensor   # (B, L, 24)
    next_wm_features: torch.Tensor
    dones: torch.Tensor          # (B,)


class ReplayBuffer:
    """Pre-allocated ring buffer over fixed-shape transition tensors."""

    def __init__(self, capacity: int, history_len: int, wm_feature_dim: int):
        self.capacity = int(capacity)
        self.history_len = int(history_len)
        self.wm_feature_dim = int(wm_feature_dim)
        self.windows = np.zeros((self.capacity, self.history_len, 24), dtype=np.float32)
        self.wm_features = np.zeros((self.capacity, self.wm_feature_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_windows = np.zeros_like(self.windows)
        self.next_wm_features = np.zeros_like(self.wm_features)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.index = 0
        self.size = 0

    def add(
            self,
            window: np.ndarray,
            wm_features: np.ndarray,
            action: float,
            reward: float,
            next_window: np.ndarray,
            next_wm_features: np.ndarray,
            done: bool) -> None:
        slot = self.index
        self.windows[slot] = window
        self.wm_features[slot] = wm_features
        self.actions[slot] = float(action)
        self.rewards[slot] = float(reward)
        self.next_windows[slot] = next_window
        self.next_wm_features[slot] = next_wm_features
        self.dones[slot] = float(bool(done))
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def extend(self, arrays: dict[str, Any]) -> None:
        """Bulk-load warm-start arrays (windows/actions/... keys)."""
        count = arrays["windows"].shape[0]
        if count > self.capacity:
            stride = count // self.capacity
            arrays = {
                key: (value[::stride][: self.capacity] if key != "delta_max"
                      and hasattr(value, "shape") else value)
                for key, value in arrays.items()
            }
            count = arrays["windows"].shape[0]
        self.windows[:count] = arrays["windows"]
        self.next_windows[:count] = arrays["next_windows"]
        self.actions[:count] = arrays["actions"]
        self.rewards[:count] = arrays["rewards"]
        self.dones[:count] = arrays["dones"]
        if self.wm_feature_dim:
            self.wm_features[:count] = 0.0
            self.next_wm_features[:count] = 0.0
        self.index = count % self.capacity
        self.size = count

    def sample(self, batch_size: int, generator: torch.Generator) -> ReplayBatch:
        if self.size < batch_size:
            raise ValueError("replay buffer has fewer samples than the batch")
        indices = torch.randint(
            0, self.size, (int(batch_size),), generator=generator
        ).numpy()
        return ReplayBatch(
            windows=torch.from_numpy(self.windows[indices]),
            wm_features=torch.from_numpy(self.wm_features[indices]),
            actions=torch.from_numpy(self.actions[indices]),
            rewards=torch.from_numpy(self.rewards[indices]),
            next_windows=torch.from_numpy(self.next_windows[indices]),
            next_wm_features=torch.from_numpy(self.next_wm_features[indices]),
            dones=torch.from_numpy(self.dones[indices]),
        )


def _critic_input(
        windows: torch.Tensor,
        wm_features: torch.Tensor,
        actions: torch.Tensor) -> torch.Tensor:
    flat = windows.flatten(1)
    if wm_features.shape[1] > 0:
        flat = torch.cat([flat, wm_features], dim=1)
    return torch.cat([flat, actions.unsqueeze(1)], dim=1)


class TwinCritic(nn.Module):
    def __init__(self, history_len: int, wm_feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = history_len * 24 + wm_feature_dim + 1
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, windows, wm_features, actions):
        state_action = _critic_input(windows, wm_features, actions)
        return self.q1(state_action), self.q2(state_action)

    def q_min(self, windows, wm_features, actions):
        q1, q2 = self.forward(windows, wm_features, actions)
        return torch.min(q1, q2).squeeze(-1)


class TD3Trainer:
    """TD3 updates over the residual policy with a frozen backbone."""

    def __init__(
            self,
            policy: nn.Module,
            *,
            gamma: float,
            tau: float,
            policy_delay: int,
            target_noise: float,
            noise_clip: float,
            behavior_reg: float,
            actor_lr: float,
            critic_lr: float,
            history_len: int,
            wm_feature_dim: int,
            device: torch.device,
            seed: int):
        self.policy = policy.to(device)
        # The target actor must own an independent copy of the backbone:
        # sharing the module object made the soft update alias target and
        # source onto the same tensors, scaling the frozen weights by
        # (1 - tau^2) on every delayed update (~0.0825x after round 1).
        self.policy_target = type(policy)(
            copy.deepcopy(policy.backbone),
            delta_max=policy.delta_max,
            lambda_blend=policy.lambda_blend,
            wm_feature_dim=policy.wm_feature_dim,
        ).to(device)
        self.policy_target.load_state_dict(self.policy.state_dict())
        for parameter in self.policy_target.parameters():
            parameter.requires_grad_(False)
        self.critic = TwinCritic(history_len, wm_feature_dim).to(device)
        self.critic_target = TwinCritic(history_len, wm_feature_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for parameter in self.critic_target.parameters():
            parameter.requires_grad_(False)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.policy_delay = int(policy_delay)
        self.target_noise = float(target_noise)
        self.noise_clip = float(noise_clip)
        self.behavior_reg = float(behavior_reg)
        actor_parameters = [p for p in self.policy.parameters() if p.requires_grad]
        if not actor_parameters:
            raise ValueError("no trainable actor parameters (backbone fully frozen and no head?)")
        self.actor_opt = torch.optim.Adam(actor_parameters, lr=float(actor_lr))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(critic_lr))
        self.device = device
        self.update_count = 0
        self.seed = int(seed)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.last_critic_loss = float("nan")
        self.last_actor_loss = float("nan")

    def update(self, batch: ReplayBatch) -> dict[str, float]:
        batch = ReplayBatch(
            windows=batch.windows.to(self.device),
            wm_features=batch.wm_features.to(self.device),
            actions=batch.actions.to(self.device),
            rewards=batch.rewards.to(self.device),
            next_windows=batch.next_windows.to(self.device),
            next_wm_features=batch.next_wm_features.to(self.device),
            dones=batch.dones.to(self.device),
        )
        with torch.no_grad():
            next_alpha, _delta = self.policy_target.raw_alpha(
                batch.next_windows, batch.next_wm_features
            )
            noise = torch.randn_like(next_alpha) * self.target_noise
            if self.noise_clip > 0.0:
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            smoothing_scale = self.policy.delta_max * self.policy.lambda_blend
            next_delta_action = torch.clamp(
                (next_alpha - self.policy_target.backbone_alpha(batch.next_windows))
                + noise * smoothing_scale,
                -self.policy.delta_max, self.policy.delta_max,
            )
            q_next = self.critic_target.q_min(
                batch.next_windows, batch.next_wm_features, next_delta_action
            )
            target_q = batch.rewards + (1.0 - batch.dones) * self.gamma * q_next

        q1, q2 = self.critic(
            batch.windows, batch.wm_features, batch.actions
        )
        critic_loss = torch.mean((q1.squeeze(-1) - target_q) ** 2) + torch.mean(
            (q2.squeeze(-1) - target_q) ** 2
        )
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.last_critic_loss = float(critic_loss.item())

        actor_metrics = {"actor_updated": 0.0}
        if self.update_count % self.policy_delay == 0:
            delta = self.policy(batch.windows, batch.wm_features)
            q = self.critic.q1(
                _critic_input(batch.windows, batch.wm_features, delta)
            ).squeeze(-1)
            actor_loss = -torch.mean(q) + self.behavior_reg * torch.mean(delta ** 2)
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            self._soft_update()
            self.last_actor_loss = float(actor_loss.item())
            actor_metrics["actor_updated"] = 1.0
        self.update_count += 1
        return {
            "critic_loss": self.last_critic_loss,
            "actor_loss": self.last_actor_loss,
            **actor_metrics,
        }

    def _soft_update(self) -> None:
        with torch.no_grad():
            for target, source in zip(
                    self.critic_target.parameters(), self.critic.parameters()
            ):
                target.mul_(1.0 - self.tau).add_(self.tau * source)
            for target, source in zip(
                    self.policy_target.parameters(), self.policy.parameters()
            ):
                # Only trainable actor parameters are tracked: frozen
                # backbone blocks must stay identical to the checkpoint.
                if source.requires_grad:
                    target.mul_(1.0 - self.tau).add_(self.tau * source)


__all__ = ["ReplayBatch", "ReplayBuffer", "TD3Trainer", "TwinCritic"]
