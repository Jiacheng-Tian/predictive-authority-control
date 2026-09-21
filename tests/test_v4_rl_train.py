"""Tests for the residual policy, TD3 trainer, and training collection.

The zero-initialization parity test is the stage-2 safety contract: with a
freshly initialized residual head the live policy must reproduce the frozen
backbone episode bit-for-bit on a structured episode.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pac.authority.model import load_alpha_model_checkpoint
from pac.v4.config import load_v4_config
from pac.v4.disturbances import build_v4_episode_spec
from pac.v4.eval.runner import build_backbone_policy, run_v4_policy_episode
from pac.v4.rl.collect import collect_training_episode
from pac.v4.rl.policy import (
    ResidualAlphaPolicy,
    load_rl_checkpoint,
    save_rl_checkpoint,
)
from pac.v4.rl.td3 import ReplayBuffer, TD3Trainer
from pac.v4.rl.warmstart import materialize_warmstart

try:
    from tests.test_v4_ranking import _collect_single_plan, _plan, _to_transition_arrays
except ImportError:  # pragma: no cover
    from test_v4_ranking import _collect_single_plan, _plan, _to_transition_arrays

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "pac_v4.yaml"
PARITY_SEED = 41235
PARITY_STEPS = 300


def _float_metrics(metrics: dict) -> dict[str, float]:
    return {
        key: value
        for key, value in metrics.items()
        if key != "ts" and isinstance(value, (int, float, bool))
    }


class ResidualPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_v4_config(_CONFIG_PATH)
        cls.backbone, _meta = load_alpha_model_checkpoint(
            Path(cls.config.backbone.checkpoint_dir)
            / "pac_train_seed_31000"
            / "checkpoint.pt"
        )

    def test_zero_init_head_matches_backbone_episode(self):
        policy = ResidualAlphaPolicy(
            self.backbone,
            delta_max=self.config.rl.delta_max,
            lambda_blend=self.config.rl.lambda_blend,
        )
        spec = build_v4_episode_spec(
            "structured", 1, PARITY_SEED, PARITY_STEPS, 0.01
        )
        backbone_metrics = run_v4_policy_episode(
            build_backbone_policy(self.config, 31000), spec, self.config,
            save_ts=False,
        )
        from pac.v4.rl.collect import LiveResidualPolicy

        live = LiveResidualPolicy(policy, explore_std=0.0, seed=1)
        residual_metrics = run_v4_policy_episode(
            live, spec, self.config, save_ts=False
        )
        base = _float_metrics(backbone_metrics)
        residual = _float_metrics(residual_metrics)
        timing = {"solver_deadline_miss_step_fraction", "solver_fallback_step_fraction"}
        for key, expected in base.items():
            if key in timing:
                continue
            if np.isnan(float(expected)) or np.isnan(float(residual[key])):
                self.assertTrue(
                    np.isnan(float(expected)) and np.isnan(float(residual[key])),
                    f"metric {key} NaN mismatch",
                )
                continue
            self.assertAlmostEqual(
                residual[key], expected,
                delta=max(1.0e-9, abs(expected) * 1.0e-9),
                msg=f"metric {key} diverged",
            )
        self.assertAlmostEqual(
            residual["rmse_3d"], base["rmse_3d"], delta=1.0e-12
        )

    def test_residual_bounded_and_backbone_frozen(self):
        policy = ResidualAlphaPolicy(
            self.backbone,
            delta_max=0.1,
            lambda_blend=1.0,
            wm_feature_dim=5,
        )
        self.assertFalse(any(
            parameter.requires_grad for parameter in policy.backbone.parameters()
        ))
        self.assertTrue(any(
            parameter.requires_grad for parameter in policy.head.parameters()
        ))
        windows = torch.randn(4, 16, 24)
        wm = torch.randn(4, 5)
        delta = policy(windows, wm)
        self.assertEqual(tuple(delta.shape), (4,))
        self.assertLessEqual(float(delta.abs().max()), 0.1 + 1e-9)
        delta_no_wm = policy(windows)
        self.assertEqual(tuple(delta_no_wm.shape), (4,))
        raw, delta = policy.raw_alpha(windows, wm)
        self.assertTrue(torch.all(raw >= 0.0))

    def test_checkpoint_roundtrip(self):
        policy = ResidualAlphaPolicy(
            self.backbone,
            delta_max=0.1,
            lambda_blend=1.0,
            wm_feature_dim=0,
        )
        with torch.no_grad():
            policy.head[0].weight.fill_(0.05)
        windows = torch.randn(3, 16, 24)
        before = policy(windows)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rl.pt"
            save_rl_checkpoint(
                path,
                policy,
                model_seed=31000,
                round_index=1,
                backbone_checkpoint="runs/train-v3-formal-baseline",
                reward_version="pac_v4_reward_v1",
                wm_version="none",
                training_summary={"env_steps": 10},
            )
            with self.assertRaises(FileExistsError):
                save_rl_checkpoint(
                    path, policy, model_seed=31000, round_index=1,
                    backbone_checkpoint="x", reward_version="r",
                    wm_version="none", training_summary={},
                )
            backbone2, _meta = load_alpha_model_checkpoint(
                Path(self.config.backbone.checkpoint_dir)
                / "pac_train_seed_31000"
                / "checkpoint.pt"
            )
            loaded, metadata = load_rl_checkpoint(
                path, backbone=backbone2, expected_model_seed=31000
            )
            self.assertEqual(metadata["reward_version"], "pac_v4_reward_v1")
            after = loaded(windows)
            np.testing.assert_allclose(
                before.detach().numpy(), after.detach().numpy(), atol=1.0e-12
            )
            with self.assertRaises(ValueError):
                load_rl_checkpoint(path, backbone=backbone2, expected_model_seed=31001)


class TD3TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_v4_config(_CONFIG_PATH)
        cls.backbone, _meta = load_alpha_model_checkpoint(
            Path(cls.config.backbone.checkpoint_dir)
            / "pac_train_seed_31000"
            / "checkpoint.pt"
        )

    def test_updates_produce_finite_losses(self):
        policy = ResidualAlphaPolicy(
            self.backbone, delta_max=0.1, lambda_blend=1.0
        )
        trainer = TD3Trainer(
            policy,
            gamma=0.99, tau=0.005, policy_delay=2,
            target_noise=0.2, noise_clip=0.5, behavior_reg=0.01,
            actor_lr=1e-4, critic_lr=1e-3,
            history_len=16, wm_feature_dim=0,
            device=torch.device("cpu"), seed=31000,
        )
        rng = np.random.default_rng(0)
        count = 128
        windows = rng.normal(0, 0.3, size=(count, 16, 24)).astype(np.float32)
        replay = ReplayBuffer(count, 16, 0)
        replay.extend({
            "windows": windows,
            "next_windows": windows,
            "actions": rng.uniform(-0.1, 0.1, size=count).astype(np.float32),
            "rewards": rng.uniform(-2, 0, size=count).astype(np.float32),
            "dones": np.zeros(count, dtype=np.float32),
        })
        for _ in range(12):
            batch = replay.sample(32, trainer.generator)
            metrics = trainer.update(batch)
        self.assertTrue(np.isfinite(metrics["critic_loss"]))
        self.assertTrue(np.isfinite(metrics["actor_loss"]))
        delta = policy(torch.from_numpy(windows[:4]))
        self.assertTrue(torch.isfinite(delta).all())


class TrainingCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_v4_config(_CONFIG_PATH)
        cls.backbone, _meta = load_alpha_model_checkpoint(
            Path(cls.config.backbone.checkpoint_dir)
            / "pac_train_seed_31000"
            / "checkpoint.pt"
        )

    def test_collect_training_episode_transitions(self):
        policy = ResidualAlphaPolicy(
            self.backbone, delta_max=0.1, lambda_blend=1.0
        )
        spec = build_v4_episode_spec(
            "ou_current", 2, 45321, 80, 0.01,
            disturbance_params={"theta": 0.5, "sigma": 0.1, "mean_scale": 0.3},
        )
        transitions, metrics = collect_training_episode(
            policy, spec, self.config, explore_std=0.1, episode_seed=5,
        )
        self.assertEqual(len(transitions), 80)
        for transition in transitions:
            self.assertEqual(transition["window"].shape, (16, 24))
            self.assertEqual(transition["next_window"].shape, (16, 24))
            self.assertTrue(np.isfinite(transition["reward"]))
            self.assertLess(transition["reward"], 0.0)
        self.assertTrue(transitions[-1]["done"])
        self.assertFalse(transitions[0]["done"])
        self.assertTrue(np.isfinite(metrics["rmse_3d"]))

    def test_warmstart_feeds_td3(self):
        dataset = _to_transition_arrays(
            _collect_single_plan(
                self.config,
                _plan(steps=60, behavior="constant_alpha", behavior_alpha=0.5),
            )
        )
        warmstart = materialize_warmstart(
            dataset, self.config, self.backbone, limit=64
        )
        replay = ReplayBuffer(1024, 16, 0)
        replay.extend(warmstart)
        policy = ResidualAlphaPolicy(
            self.backbone, delta_max=0.1, lambda_blend=1.0
        )
        trainer = TD3Trainer(
            policy,
            gamma=0.99, tau=0.005, policy_delay=2,
            target_noise=0.2, noise_clip=0.5, behavior_reg=0.01,
            actor_lr=1e-4, critic_lr=1e-3,
            history_len=16, wm_feature_dim=0,
            device=torch.device("cpu"), seed=1,
        )
        for _ in range(5):
            batch = replay.sample(16, trainer.generator)
            metrics = trainer.update(batch)
        self.assertTrue(np.isfinite(metrics["critic_loss"]))


if __name__ == "__main__":
    unittest.main()
