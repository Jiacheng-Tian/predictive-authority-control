"""Direct-thruster RL control arm (no backbone, no alpha, no safety filter).

This is the negative-control arm for the residual architecture: a plain
TD3 agent maps the same 16-step x 24-dim observation window the PAC stack
sees directly to six thruster commands in ``[-1, 1]^6``.  It carries no
frozen backbone, produces no authority coefficient, and bypasses the
alpha gain/smoothing/rate-limit stack entirely — the tanh output is the
actuator command.  Its single-step reward drops only the ``w_delta_alpha``
term (no alpha exists); every other frozen weight is shared with the
residual arm so the two remain directly comparable.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from pac.simulation.core import AUVSimulator
from pac.v4.eval.runner import run_v4_policy_episode
from pac.evaluation.episodes import desired_heading
from pac.v4.rl.reward import compute_step_reward
from pac.v4.rl.td3 import ReplayBuffer, TwinCritic, _critic_input

DIRECT_RL_CHECKPOINT_KIND = "direct_rl_policy"
DIRECT_ACTION_DIM = 6


class DirectRLPolicy(nn.Module):
    """MLP from the flattened observation window to bounded thruster commands."""

    def __init__(
            self,
            history_len: int,
            *,
            input_dim: int = 24,
            hidden_dim: int = 256,
            action_dim: int = DIRECT_ACTION_DIM):
        super().__init__()
        self.history_len = int(history_len)
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        # LayerNorm first: raw feature magnitudes span orders of magnitude
        # (cm-scale errors to tens of meters under disturbance failures).
        self.net = nn.Sequential(
            nn.LayerNorm(self.history_len * self.input_dim),
            nn.Linear(self.history_len * self.input_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), self.action_dim),
            nn.Tanh(),
        )
        # Zero-initialized output layer (net[-2]; net[-1] is the Tanh): an
        # untrained policy commands no thrust rather than random thrashing.
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3 or windows.shape[-1] != self.input_dim:
            raise ValueError("expected windows with shape (B, L, 24)")
        if windows.shape[1] > self.history_len:
            windows = windows[:, -self.history_len:, :]
        return self.net(windows.flatten(1))


class DirectTD3Trainer:
    """Standard TD3 over the 6-dim bounded action space."""

    def __init__(
            self,
            policy: nn.Module,
            *,
            gamma: float,
            tau: float,
            policy_delay: int,
            target_noise: float,
            noise_clip: float,
            actor_lr: float,
            critic_lr: float,
            history_len: int,
            action_dim: int = DIRECT_ACTION_DIM,
            hidden_dim: int = 256,
            device: torch.device,
            seed: int):
        self.policy = policy.to(device)
        self.policy_target = copy.deepcopy(policy).to(device)
        for parameter in self.policy_target.parameters():
            parameter.requires_grad_(False)
        self.critic = TwinCritic(
            history_len, 0, hidden_dim=hidden_dim, action_dim=int(action_dim)
        ).to(device)
        self.critic_target = TwinCritic(
            history_len, 0, hidden_dim=hidden_dim, action_dim=int(action_dim)
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for parameter in self.critic_target.parameters():
            parameter.requires_grad_(False)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.policy_delay = int(policy_delay)
        self.target_noise = float(target_noise)
        self.noise_clip = float(noise_clip)
        trainable = [p for p in self.policy.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("direct policy has no trainable parameters")
        self.actor_opt = torch.optim.Adam(trainable, lr=float(actor_lr))
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=float(critic_lr)
        )
        self.device = device
        self.update_count = 0
        self.seed = int(seed)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.last_critic_loss = float("nan")
        self.last_actor_loss = float("nan")

    def update(self, batch) -> dict[str, float]:
        windows = batch.windows.to(self.device)
        wm_features = batch.wm_features.to(self.device)
        actions = batch.actions.to(self.device)
        rewards = batch.rewards.to(self.device)
        next_windows = batch.next_windows.to(self.device)
        next_wm_features = batch.next_wm_features.to(self.device)
        dones = batch.dones.to(self.device)
        with torch.no_grad():
            next_action = self.policy_target(next_windows)
            noise = torch.randn_like(next_action) * self.target_noise
            if self.noise_clip > 0.0:
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_action = torch.clamp(next_action + noise, -1.0, 1.0)
            q_next = self.critic_target.q_min(
                next_windows, next_wm_features, next_action
            )
            target_q = rewards + (1.0 - dones) * self.gamma * q_next
        q1, q2 = self.critic(windows, wm_features, actions)
        critic_loss = torch.mean((q1.squeeze(-1) - target_q) ** 2) + torch.mean(
            (q2.squeeze(-1) - target_q) ** 2
        )
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.last_critic_loss = float(critic_loss.item())

        actor_metrics = {"actor_updated": 0.0}
        if self.update_count % self.policy_delay == 0:
            action = self.policy(windows)
            q = self.critic.q1(
                _critic_input(windows, wm_features, action)
            ).squeeze(-1)
            actor_loss = -torch.mean(q)
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
                if source.requires_grad:
                    target.mul_(1.0 - self.tau).add_(self.tau * source)


class DirectLivePolicy:
    """Runner adapter driving the loop with the direct actor (plus exploration)."""

    uses_wm: bool = False
    direct_action: bool = True

    def __init__(self, policy: DirectRLPolicy, *, explore_std: float = 0.0,
                 seed: int = 0):
        self.policy = policy
        self.explore_std = float(explore_std)
        self.rng = np.random.default_rng(int(seed))
        self.last_action = np.zeros(DIRECT_ACTION_DIM, dtype=float)

    def reset(self, spec, episode_end: float) -> None:
        del spec, episode_end
        self.last_action = np.zeros(DIRECT_ACTION_DIM, dtype=float)

    def select_action(self, context) -> np.ndarray:
        windows = torch.from_numpy(
            np.asarray(context.feature_window, dtype=np.float32)[None]
        )
        with torch.no_grad():
            action = self.policy(windows).numpy()[0].astype(float)
        if self.explore_std > 0.0:
            action = action + self.rng.normal(
                0.0, self.explore_std, size=self.policy.action_dim
            )
        action = np.clip(action, -1.0, 1.0)
        if not np.all(np.isfinite(action)):
            # Defensive fallback: any numerical failure must degrade to a
            # zero (thrusters-off) command, never to NaN reaching the plant.
            action = np.zeros(self.policy.action_dim, dtype=float)
        self.last_action = action
        return action


class DirectTransitionRecorder:
    """Runner ``transition_sink`` building v1-rewarded direct transitions."""

    def __init__(self, config):
        self.config = config
        self.entries: list[dict[str, Any]] = []

    def __call__(self, context, step_info, next_state, alpha: float) -> None:
        del alpha  # no authority coefficient in this arm
        if self.entries:
            self.entries[-1]["next_window"] = np.array(
                context.feature_window, dtype=np.float32, copy=True
            )
            self.entries[-1]["next_wm_features"] = np.zeros(0, dtype=np.float32)
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
            alpha=0.0,
            previous_alpha=0.0,
            saturation_fraction=saturation,
            deadline_missed=bool(telemetry.get("deadline_missed", False)),
            constraint_violated=position_error > 20.0,
            include_delta_alpha=False,
        )
        self.entries.append({
            "window": np.array(context.feature_window, dtype=np.float32, copy=True),
            "wm_features": np.zeros(0, dtype=np.float32),
            "action": np.array(context.policy_action, dtype=float, copy=True),
            "reward": float(reward),
            "next_window": None,
            "next_wm_features": None,
            "done": False,
            "position_error": position_error,
        })
        self._last_window = self.entries[-1]["window"]

    def close(self) -> None:
        if not self.entries:
            return
        last = self.entries[-1]
        last["next_window"] = np.array(self._last_window, dtype=np.float32, copy=True)
        last["next_wm_features"] = np.zeros(0, dtype=np.float32)
        last["done"] = True

    @property
    def transitions(self) -> list[dict[str, Any]]:
        incomplete = [index for index, entry in enumerate(self.entries)
                      if entry["next_window"] is None]
        if incomplete:
            raise ValueError(f"transitions not closed: {len(incomplete)} entries")
        return self.entries


class _DirectRecordingPolicyWrapper:
    """Binds the recorder to the live policy's commanded action."""

    def __init__(self, live: DirectLivePolicy, recorder: DirectTransitionRecorder):
        self._live = live
        self._recorder = recorder

    def __getattr__(self, name):
        return getattr(self._live, name)

    @property
    def uses_wm(self) -> bool:
        return False

    direct_action: bool = True

    def reset(self, spec, episode_end: float) -> None:
        self._live.reset(spec, episode_end)

    def select_action(self, context) -> np.ndarray:
        action = self._live.select_action(context)
        context.policy_action = action
        return action


def collect_direct_episode(
        policy: DirectRLPolicy,
        spec,
        config,
        *,
        explore_std: float,
        episode_seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one training episode and return rewarded direct transitions."""
    recorder = DirectTransitionRecorder(config)
    live = DirectLivePolicy(policy, explore_std=explore_std, seed=int(episode_seed))
    wrapper = _DirectRecordingPolicyWrapper(live, recorder)
    metrics = run_v4_policy_episode(wrapper, spec, config, save_ts=False,
                                    transition_sink=recorder)
    recorder.close()
    return recorder.transitions, metrics


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[4]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_direct_rl_checkpoint(
        checkpoint_path: str | Path,
        policy: DirectRLPolicy,
        *,
        model_seed: int,
        reward_version: str,
        training_summary: dict[str, Any]) -> dict[str, Any]:
    target = Path(checkpoint_path)
    if target.exists():
        raise FileExistsError(f"direct RL checkpoint already exists: {target}")
    payload = {
        "protocol_version": "predictive_authority_v4",
        "kind": DIRECT_RL_CHECKPOINT_KIND,
        "model_seed": int(model_seed),
        "history_len": int(policy.history_len),
        "input_dim": int(policy.input_dim),
        "action_dim": int(policy.action_dim),
        "reward_version": str(reward_version),
        "reward_variant": "no_delta_alpha",
        "controller_pair": "real10kg_smc_steady+real10kg_mpc_ltv_v3",
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
        },
        "param_count": int(sum(p.numel() for p in policy.parameters())),
        "training_summary": dict(training_summary),
        "git_commit": _git_commit(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return payload


def load_direct_rl_checkpoint(
        checkpoint_path: str | Path,
        *,
        expected_model_seed: int | None = None) -> tuple[DirectRLPolicy, dict[str, Any]]:
    path = Path(checkpoint_path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, ValueError) as exc:
        raise ValueError(f"invalid direct RL checkpoint: {path}") from exc
    if not isinstance(payload, dict) or payload.get("kind") != DIRECT_RL_CHECKPOINT_KIND:
        raise ValueError("checkpoint is not a v4 direct RL policy")
    for key in ("model_seed", "history_len", "action_dim", "state_dict",
                "training_summary", "reward_version"):
        if key not in payload:
            raise ValueError(f"direct RL checkpoint missing field: {key}")
    if expected_model_seed is not None and int(payload["model_seed"]) != int(expected_model_seed):
        raise ValueError("direct RL checkpoint model_seed mismatch")
    policy = DirectRLPolicy(
        int(payload["history_len"]),
        input_dim=int(payload.get("input_dim", 24)),
        action_dim=int(payload["action_dim"]),
    )
    policy.load_state_dict(payload["state_dict"], strict=True)
    policy.eval()
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    metadata["checkpoint_path"] = str(path.resolve())
    return policy, metadata


def train_direct_rl(
        config,
        output_dir: str | Path,
        *,
        model_seed: int,
        max_env_steps: int | None = None,
        device: str = "cpu",
        log=lambda message: None) -> dict[str, Any]:
    """Train one model seed of the direct-thruster control arm.

    Mirrors ``train_residual_rl`` except that no backbone is loaded, the
    replay buffer starts empty (the stage-one dataset holds scalar alpha
    actions with different semantics), and collection goes through
    :class:`DirectLivePolicy`.
    """
    from pac.v4.rl.train import training_episode_plan
    from pac.v4.collector import build_v4_spec_for_plan

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(device)
    rl = config.rl
    total_steps = int(max_env_steps or rl.max_steps_per_round)
    policy = DirectRLPolicy(int(config.authority_model.history_len))
    trainer = DirectTD3Trainer(
        policy,
        gamma=float(rl.gamma),
        tau=float(rl.tau),
        policy_delay=int(rl.policy_delay),
        target_noise=float(rl.target_noise),
        noise_clip=float(rl.noise_clip),
        actor_lr=float(rl.actor_lr),
        critic_lr=float(rl.critic_lr),
        history_len=int(config.authority_model.history_len),
        device=resolved_device,
        seed=int(model_seed),
    )
    capacity = total_steps + 4096
    replay = ReplayBuffer(
        capacity,
        int(config.authority_model.history_len),
        0,
        action_dim=DIRECT_ACTION_DIM,
    )
    log(
        f"[direct-rl-train seed={model_seed}] no warmstart "
        f"(direct actions) target_env_steps={total_steps}"
    )

    history_rows: list[dict[str, float]] = []
    env_steps = 0
    updates = 0
    episode_index = 0
    episode_rewards: list[float] = []
    started = time.perf_counter()
    partial_path = output / f"partial_seed_{int(model_seed)}.pt"
    if partial_path.exists():
        state = torch.load(partial_path, map_location="cpu", weights_only=True)
        policy.load_state_dict(state["policy"])
        trainer.critic.load_state_dict(state["critic"])
        trainer.critic_target.load_state_dict(state["critic_target"])
        trainer.actor_opt.load_state_dict(state["actor_opt"])
        trainer.critic_opt.load_state_dict(state["critic_opt"])
        trainer.policy_target.load_state_dict(state["policy_target"])
        trainer.update_count = int(state["update_count"])
        episode_index = int(state["episode_index"])
        env_steps = int(state["env_steps"])
        updates = int(state["updates"])
        log(
            f"[direct-rl-train seed={model_seed}] resumed partial checkpoint "
            f"at episode={episode_index} env_steps={env_steps}"
        )

    def save_partial() -> None:
        temporary = partial_path.with_suffix(".tmp")
        torch.save({
            "policy": policy.state_dict(),
            "policy_target": trainer.policy_target.state_dict(),
            "critic": trainer.critic.state_dict(),
            "critic_target": trainer.critic_target.state_dict(),
            "actor_opt": trainer.actor_opt.state_dict(),
            "critic_opt": trainer.critic_opt.state_dict(),
            "update_count": trainer.update_count,
            "episode_index": episode_index,
            "env_steps": env_steps,
            "updates": updates,
        }, temporary)
        temporary.replace(partial_path)

    while env_steps < total_steps:
        plan = training_episode_plan(config, episode_index)
        spec = build_v4_spec_for_plan(config, plan)
        transitions, metrics = collect_direct_episode(
            policy,
            spec,
            config,
            explore_std=float(rl.expl_noise),
            episode_seed=int(spec.episode_seed) + 7919 * episode_index,
        )
        for transition in transitions:
            replay.add(
                transition["window"],
                transition["wm_features"],
                transition["action"],
                transition["reward"],
                transition["next_window"],
                transition["next_wm_features"],
                transition["done"],
            )
            env_steps += 1
            if env_steps >= int(rl.warmup_steps):
                for _update in range(int(rl.updates_per_step)):
                    batch = replay.sample(int(rl.batch_size), trainer.generator)
                    losses = trainer.update(batch)
                    updates += 1
        episode_rewards.append(float(np.mean(
            [transition["reward"] for transition in transitions]
        )))
        if episode_index % 2 == 0 or env_steps >= total_steps:
            history_rows.append({
                "episode": episode_index,
                "env_steps": env_steps,
                "updates": updates,
                "mean_episode_reward": float(np.mean(episode_rewards[-3:])),
                "critic_loss": trainer.last_critic_loss,
                "actor_loss": trainer.last_actor_loss,
                "elapsed_s": time.perf_counter() - started,
            })
            log(
                f"[direct-rl-train seed={model_seed}] ep={episode_index} "
                f"steps={env_steps}/{total_steps} "
                f"reward={history_rows[-1]['mean_episode_reward']:.3f} "
                f"critic={trainer.last_critic_loss:.4f} "
                f"elapsed={history_rows[-1]['elapsed_s']:.0f}s"
            )
        episode_index += 1
        if episode_index % 20 == 0:
            save_partial()

    pd.DataFrame(history_rows).to_csv(
        output / f"training_history_seed_{int(model_seed)}.csv", index=False
    )
    checkpoint_path = output / f"direct_rl_seed_{int(model_seed)}.pt"
    summary = save_direct_rl_checkpoint(
        checkpoint_path,
        policy,
        model_seed=int(model_seed),
        reward_version=str(config.reward.version),
        training_summary={
            "env_steps": int(env_steps),
            "gradient_updates": int(updates),
            "episodes": int(episode_index),
            "warmstart_rows": 0,
            "final_critic_loss": trainer.last_critic_loss,
            "final_actor_loss": trainer.last_actor_loss,
            "mean_episode_reward_last3": float(np.mean(episode_rewards[-3:]))
            if episode_rewards else float("nan"),
            "expl_noise": float(rl.expl_noise),
            "device": str(resolved_device),
            "reward_variant": "no_delta_alpha",
        },
    )
    summary["checkpoint_path"] = str(checkpoint_path)
    (output / f"training_summary_seed_{int(model_seed)}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    log(f"[direct-rl-train seed={model_seed}] checkpoint: {checkpoint_path}")
    return summary


__all__ = [
    "DIRECT_ACTION_DIM",
    "DIRECT_RL_CHECKPOINT_KIND",
    "DirectLivePolicy",
    "DirectRLPolicy",
    "DirectTD3Trainer",
    "collect_direct_episode",
    "load_direct_rl_checkpoint",
    "save_direct_rl_checkpoint",
    "train_direct_rl",
]
