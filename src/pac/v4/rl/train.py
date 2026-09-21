"""Round-based constrained TD3 training for the v4 residual policy."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from pac.authority.model import load_alpha_model_checkpoint
from pac.v4.collector import build_v4_spec_for_plan, load_transition_dataset
from pac.v4.disturbances import STRUCTURED_FAMILY
from pac.v4.rl.collect import collect_training_episode
from pac.v4.rl.policy import (
    ResidualAlphaPolicy,
    load_rl_checkpoint,
    save_rl_checkpoint,
)
from pac.v4.rl.td3 import ReplayBuffer, TD3Trainer
from pac.v4.rl.warmstart import materialize_warmstart
from pac.v4.worldmodel.train import dataset_content_hash

_STRUCTURES = (1, 2, 3)


def training_episode_plan(config, episode_index: int) -> dict:
    """Rotate structured scenarios and train-range disturbance families."""
    families = (STRUCTURED_FAMILY, *config.disturbances.families)
    family = families[episode_index % len(families)]
    seed = int(config.seeds.wm_train[episode_index % len(config.seeds.wm_train)])
    if family == STRUCTURED_FAMILY:
        scenario = _STRUCTURES[episode_index % len(_STRUCTURES)]
    else:
        scenario = int(config.disturbances.base_scenario[family])
    return {
        "split": "train",
        "seed": seed,
        "family": family,
        "scenario_id": scenario,
        "behavior": "current_policy",
        "behavior_alpha": float("nan"),
        "steps": int(config.environment.steps),
    }


def train_residual_rl(
        config,
        dataset_dir: str | Path,
        output_dir: str | Path,
        *,
        model_seed: int,
        round_index: int = 1,
        max_env_steps: int | None = None,
        resume_checkpoint: str | Path | None = None,
        wm_checkpoint: str | Path | None = None,
        device: str = "cpu",
        log=lambda message: None) -> dict[str, Any]:
    """Train one model seed of the residual policy under the frozen protocol."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(device)
    rl = config.rl
    backbone_path = (
        Path(config.backbone.checkpoint_dir)
        / f"pac_train_seed_{int(model_seed)}"
        / "checkpoint.pt"
    )
    backbone, _metadata = load_alpha_model_checkpoint(backbone_path)
    wm_dim = 0
    wm_computer = None
    wm_version = "none"
    if wm_checkpoint is not None:
        from pac.v4.wmauthority import build_wm_authority_computer

        wm_computer, wm_version = build_wm_authority_computer(
            config, wm_checkpoint, dataset_dir, device=resolved_device
        )
        wm_dim = 5

    if resume_checkpoint is not None:
        policy, _resume_metadata = load_rl_checkpoint(
            resume_checkpoint, backbone=backbone, expected_model_seed=model_seed
        )
    else:
        policy = ResidualAlphaPolicy(
            backbone,
            delta_max=float(rl.delta_max),
            lambda_blend=float(rl.lambda_blend),
            wm_feature_dim=wm_dim,
            freeze_backbone=bool(rl.freeze_backbone_round1) if round_index == 1
            else False,
        )
    if int(policy.wm_feature_dim) != wm_dim:
        raise ValueError("resumed policy wm_feature_dim does not match this run")

    trainer = TD3Trainer(
        policy,
        gamma=float(rl.gamma),
        tau=float(rl.tau),
        policy_delay=int(rl.policy_delay),
        target_noise=float(rl.target_noise),
        noise_clip=float(rl.noise_clip),
        behavior_reg=float(rl.behavior_reg),
        actor_lr=float(rl.actor_lr),
        critic_lr=float(rl.critic_lr),
        history_len=int(config.authority_model.history_len),
        wm_feature_dim=wm_dim,
        device=resolved_device,
        seed=int(model_seed),
    )

    dataset_hash = dataset_content_hash(dataset_dir)
    dataset = load_transition_dataset(dataset_dir)
    warmstart = materialize_warmstart(
        dataset, config, policy.backbone, limit=int(rl.warmstart_transitions)
    )
    total_steps = int(max_env_steps or rl.max_steps_per_round)
    capacity = max(int(rl.warmstart_transitions), total_steps) + 4096
    replay = ReplayBuffer(
        capacity,
        int(config.authority_model.history_len),
        wm_dim,
    )
    replay.extend(warmstart)
    log(
        f"[rl-train seed={model_seed}] warmstart rows={replay.size} "
        f"target_env_steps={total_steps}"
    )

    history_rows: list[dict[str, float]] = []
    env_steps = 0
    updates = 0
    episode_index = 0
    episode_rewards: list[float] = []
    started = time.perf_counter()
    while env_steps < total_steps:
        plan = training_episode_plan(config, episode_index)
        spec = build_v4_spec_for_plan(config, plan)
        transitions, metrics = collect_training_episode(
            policy,
            spec,
            config,
            explore_std=float(rl.expl_noise),
            episode_seed=int(spec.episode_seed) + 7919 * episode_index,
            wm_computer=wm_computer,
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
            if env_steps >= int(rl.warmup_steps) and env_steps % 1 == 0:
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
                f"[rl-train seed={model_seed}] ep={episode_index} "
                f"steps={env_steps}/{total_steps} "
                f"reward={history_rows[-1]['mean_episode_reward']:.3f} "
                f"critic={trainer.last_critic_loss:.4f} "
                f"elapsed={history_rows[-1]['elapsed_s']:.0f}s"
            )
        episode_index += 1

    import pandas as pd

    pd.DataFrame(history_rows).to_csv(output / "training_history.csv", index=False)
    checkpoint_path = output / f"residual_rl_seed_{int(model_seed)}_round{round_index}.pt"
    summary = save_rl_checkpoint(
        checkpoint_path,
        policy,
        model_seed=int(model_seed),
        round_index=int(round_index),
        backbone_checkpoint=str(backbone_path),
        reward_version=str(config.reward.version),
        wm_version=wm_version,
        training_summary={
            "env_steps": int(env_steps),
            "gradient_updates": int(updates),
            "episodes": int(episode_index),
            "warmstart_rows": int(warmstart["valid_rows"]),
            "warmstart_limit": int(rl.warmstart_transitions),
            "dataset_hash": dataset_hash,
            "dataset_dir": str(dataset_dir),
            "final_critic_loss": trainer.last_critic_loss,
            "final_actor_loss": trainer.last_actor_loss,
            "mean_episode_reward_last3": float(np.mean(episode_rewards[-3:]))
            if episode_rewards else float("nan"),
            "wm_checkpoint": str(wm_checkpoint) if wm_checkpoint else "none",
            "delta_max": float(rl.delta_max),
            "lambda_blend": float(rl.lambda_blend),
            "behavior_reg": float(rl.behavior_reg),
            "expl_noise": float(rl.expl_noise),
            "device": str(resolved_device),
        },
    )
    (output / f"training_summary_seed_{int(model_seed)}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    log(f"[rl-train seed={model_seed}] checkpoint: {checkpoint_path}")
    return summary


__all__ = ["train_residual_rl", "training_episode_plan"]
