"""Tests for the direct-thruster RL control arm (P3)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pac.v4.config import load_v4_config
from pac.v4.disturbances import build_v4_episode_spec
from pac.v4.eval.runner import run_v4_policy_episode
from pac.v4.rl.direct import (
    DIRECT_ACTION_DIM,
    DirectLivePolicy,
    DirectRLPolicy,
    DirectTD3Trainer,
    collect_direct_episode,
    load_direct_rl_checkpoint,
    save_direct_rl_checkpoint,
)
from pac.v4.rl.policy import ResidualAlphaPolicy, unfreeze_round2_blocks
from pac.v4.rl.td3 import ReplayBuffer
from pac.v4.rl.td3 import TD3Trainer  # noqa: F401  (regression import)

try:
    from pac.authority.model import load_alpha_model_checkpoint
except ImportError:  # pragma: no cover
    load_alpha_model_checkpoint = None

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "pac_v4.yaml"


class DirectRLPolicyTests(unittest.TestCase):
    def test_output_shape_and_bounds(self):
        policy = DirectRLPolicy(16)
        windows = torch.randn(5, 16, 24)
        action = policy(windows)
        self.assertEqual(tuple(action.shape), (5, DIRECT_ACTION_DIM))
        self.assertTrue(torch.all(action >= -1.0 - 1e-9))
        self.assertTrue(torch.all(action <= 1.0 + 1e-9))

    def test_zero_init_commands_no_thrust(self):
        policy = DirectRLPolicy(16)
        action = policy(torch.randn(3, 16, 24))
        torch.testing.assert_close(
            action, torch.zeros(3, DIRECT_ACTION_DIM)
        )

    def test_history_window_truncated(self):
        policy = DirectRLPolicy(4)
        long_window = torch.randn(2, 9, 24)
        short_window = long_window[:, -4:, :]
        torch.testing.assert_close(policy(long_window), policy(short_window))


class DirectTD3Tests(unittest.TestCase):
    def test_replay_and_updates_finite(self):
        policy = DirectRLPolicy(16)
        trainer = DirectTD3Trainer(
            policy,
            gamma=0.99, tau=0.005, policy_delay=2,
            target_noise=0.2, noise_clip=0.5,
            actor_lr=1e-4, critic_lr=1e-3,
            history_len=16, device=torch.device("cpu"), seed=31000,
        )
        rng = np.random.default_rng(1)
        count = 256
        windows = rng.normal(0, 0.3, size=(count, 16, 24)).astype(np.float32)
        replay = ReplayBuffer(count, 16, 0, action_dim=DIRECT_ACTION_DIM)
        replay.extend({
            "windows": windows,
            "next_windows": windows,
            "actions": rng.uniform(-1, 1, size=(count, DIRECT_ACTION_DIM)).astype(np.float32),
            "rewards": rng.uniform(-2, 0, size=count).astype(np.float32),
            "dones": np.zeros(count, dtype=np.float32),
        })
        self.assertEqual(replay.actions.shape, (count, DIRECT_ACTION_DIM))
        for _ in range(12):
            batch = replay.sample(32, trainer.generator)
            self.assertEqual(tuple(batch.actions.shape), (32, DIRECT_ACTION_DIM))
            metrics = trainer.update(batch)
        self.assertTrue(np.isfinite(metrics["critic_loss"]))
        self.assertTrue(np.isfinite(metrics["actor_loss"]))
        self.assertTrue(
            torch.isfinite(policy(torch.from_numpy(windows[:4]))).all()
        )

    def test_residual_replay_still_accepts_flat_actions(self):
        """The scalar residual path keeps working through the shared buffer."""
        rng = np.random.default_rng(2)
        count = 64
        replay = ReplayBuffer(count, 16, 0)
        replay.extend({
            "windows": rng.normal(0, 0.3, size=(count, 16, 24)).astype(np.float32),
            "next_windows": rng.normal(0, 0.3, size=(count, 16, 24)).astype(np.float32),
            "actions": rng.uniform(-0.1, 0.1, size=count).astype(np.float32),
            "rewards": rng.uniform(-2, 0, size=count).astype(np.float32),
            "dones": np.zeros(count, dtype=np.float32),
        })
        batch = replay.sample(8, torch.Generator().manual_seed(0))
        self.assertEqual(batch.actions.ndim, 2)
        self.assertEqual(tuple(batch.actions.shape), (8, 1))


class DirectCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_v4_config(_CONFIG_PATH)

    def test_collect_direct_episode_transitions(self):
        policy = DirectRLPolicy(
            int(self.config.authority_model.history_len)
        )
        spec = build_v4_episode_spec(
            "ou_current", 2, 45321, 80, 0.01,
            disturbance_params={"theta": 0.5, "sigma": 0.1, "mean_scale": 0.3},
        )
        transitions, metrics = collect_direct_episode(
            policy, spec, self.config, explore_std=0.2, episode_seed=5,
        )
        self.assertEqual(len(transitions), 80)
        for transition in transitions:
            self.assertEqual(transition["window"].shape, (16, 24))
            self.assertEqual(transition["action"].shape, (DIRECT_ACTION_DIM,))
            self.assertTrue(np.all(np.abs(transition["action"]) <= 1.0 + 1e-9))
            self.assertTrue(np.isfinite(transition["reward"]))
            self.assertLess(transition["reward"], 0.0)
        self.assertTrue(transitions[-1]["done"])
        self.assertFalse(transitions[0]["done"])
        self.assertTrue(np.isfinite(metrics["rmse_3d"]))

    def test_direct_episode_runs_without_alpha_stack(self):
        config = load_v4_config(_CONFIG_PATH)
        policy = DirectRLPolicy(int(config.authority_model.history_len))
        live = DirectLivePolicy(policy, explore_std=0.0, seed=1)
        spec = build_v4_episode_spec("structured", 1, 41235, 120, 0.01)
        metrics = run_v4_policy_episode(live, spec, config, save_ts=False)
        self.assertTrue(np.isfinite(metrics["rmse_3d"]))
        # zero-initialized policy commands no thrust: alpha stays 0 and
        # the vehicle simply drifts along the open-loop trajectory
        self.assertEqual(metrics["authority_alpha_mean"], 0.0)
        self.assertGreater(metrics["rmse_3d"], 0.0)


class DirectCheckpointTests(unittest.TestCase):
    def test_roundtrip(self):
        config = load_v4_config(_CONFIG_PATH)
        policy = DirectRLPolicy(int(config.authority_model.history_len))
        with torch.no_grad():
            policy.net[-2].bias.fill_(0.25)
        windows = torch.randn(3, 16, 24)
        before = policy(windows)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "direct_rl.pt"
            save_direct_rl_checkpoint(
                path,
                policy,
                model_seed=31000,
                reward_version=str(config.reward.version),
                training_summary={"env_steps": 10},
            )
            with self.assertRaises(FileExistsError):
                save_direct_rl_checkpoint(
                    path, policy, model_seed=31000,
                    reward_version="r", training_summary={},
                )
            loaded, metadata = load_direct_rl_checkpoint(
                path, expected_model_seed=31000
            )
            self.assertEqual(metadata["action_dim"], DIRECT_ACTION_DIM)
            self.assertEqual(metadata["reward_variant"], "no_delta_alpha")
            after = loaded(windows)
            np.testing.assert_allclose(
                before.detach().numpy(), after.detach().numpy(), atol=1e-12
            )
            with self.assertRaises(ValueError):
                load_direct_rl_checkpoint(path, expected_model_seed=1)


class Round2UnfreezeTests(unittest.TestCase):
    """P2: only encoder_last + head thaw in round 2."""

    @classmethod
    def setUpClass(cls):
        cls.config = load_v4_config(_CONFIG_PATH)
        cls.backbone, _meta = load_alpha_model_checkpoint(
            Path(cls.config.backbone.checkpoint_dir)
            / "pac_train_seed_31000"
            / "checkpoint.pt"
        )

    def _trainable_names(self, policy):
        return {
            name
            for name, parameter in policy.backbone.named_parameters()
            if parameter.requires_grad
        }

    def test_round2_unfreezes_exactly_encoder_last_and_head(self):
        policy = ResidualAlphaPolicy(
            self.backbone,
            delta_max=self.config.rl.delta_max,
            lambda_blend=self.config.rl.lambda_blend,
            freeze_backbone=True,
        )
        unfreeze_round2_blocks(policy, self.config.rl.unfreeze_blocks_round2)
        trainable = self._trainable_names(policy)
        last_layer_prefix = (
            f"encoder.layers.{len(policy.backbone.encoder.layers) - 1}."
        )
        expected = {
            name
            for name, _ in policy.backbone.named_parameters()
            if name.startswith(last_layer_prefix) or name.startswith("head.")
        }
        self.assertEqual(trainable, expected)
        # residual head stays trainable
        self.assertTrue(
            all(p.requires_grad for p in policy.head.parameters())
        )

    def test_unknown_block_rejected(self):
        policy = ResidualAlphaPolicy(
            self.backbone, delta_max=0.1, lambda_blend=1.0
        )
        with self.assertRaises(ValueError):
            unfreeze_round2_blocks(policy, ("encoder_all",))


if __name__ == "__main__":
    unittest.main()
