"""Tests for S5: n-step replay aggregation and critic input norm."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from pac.v4.rl.td3 import ReplayBatch, ReplayBuffer, TwinCritic


def _transition(index, reward, done=False):
    window = np.full((2, 24), float(index), dtype=np.float32)
    return dict(
        window=window,
        wm_features=np.zeros(0, dtype=np.float32),
        action=np.array([float(index)], dtype=np.float32),
        reward=float(reward),
        next_window=np.full((2, 24), float(index) + 0.5, dtype=np.float32),
        next_wm_features=np.zeros(0, dtype=np.float32),
        done=done,
    )


class NStepReplayTests(unittest.TestCase):
    def test_chain_aggregation_math(self):
        buffer = ReplayBuffer(32, 2, 0, n_step=3, gamma=0.5)
        for index, reward in enumerate((1.0, 2.0, 4.0)):
            buffer.add(**_transition(index, reward))
        # after 3 adds the chain (0,1,2) is complete: R = 1 + 2*.5 + 4*.25
        self.assertEqual(buffer.size, 1)
        self.assertAlmostEqual(buffer.rewards[0], 1.0 + 1.0 + 1.0, places=6)
        self.assertEqual(buffer.horizons[0], 3.0)
        np.testing.assert_allclose(
            buffer.next_windows[0], np.full((2, 24), 2.5, dtype=np.float32)
        )

    def test_done_truncation(self):
        buffer = ReplayBuffer(32, 2, 0, n_step=5, gamma=0.9)
        buffer.add(**_transition(0, 1.0))
        buffer.add(**_transition(1, 2.0, done=True))
        # two truncated chains: (0,1) and (1)
        self.assertEqual(buffer.size, 2)
        self.assertAlmostEqual(
            buffer.rewards[0], 1.0 + 0.9 * 2.0, places=6
        )
        self.assertEqual(buffer.horizons[0], 2.0)
        self.assertAlmostEqual(buffer.rewards[1], 2.0, places=6)
        self.assertEqual(buffer.dones[1], 1.0)
        self.assertEqual(buffer.horizons[1], 1.0)  # terminal lone chain

    def test_n1_is_bitwise_legacy(self):
        buffer = ReplayBuffer(8, 2, 0, n_step=1, gamma=0.99)
        buffer.add(**_transition(0, -0.5))
        self.assertEqual(buffer.size, 1)
        self.assertAlmostEqual(buffer.rewards[0], -0.5, places=7)
        self.assertEqual(buffer.horizons[0], 1.0)

    def test_extend_resets_pending_and_marks_horizon_one(self):
        buffer = ReplayBuffer(16, 2, 0, n_step=4, gamma=0.99)
        buffer.add(**_transition(0, 1.0))  # leaves an open pending chain
        buffer.extend({
            "windows": np.zeros((3, 2, 24), dtype=np.float32),
            "next_windows": np.zeros((3, 2, 24), dtype=np.float32),
            "actions": np.zeros(3, dtype=np.float32),
            "rewards": np.ones(3, dtype=np.float32),
            "dones": np.zeros(3, dtype=np.float32),
        })
        self.assertEqual(buffer.size, 3)
        np.testing.assert_array_equal(buffer.horizons[:3], np.ones(3))
        buffer.add(**_transition(9, 1.0, done=True))  # no stale chaining
        self.assertEqual(buffer.size, 4)

    def test_gamma_power_bootstrap_target(self):
        gamma = 0.9
        batch = ReplayBatch(
            windows=torch.zeros(2, 2, 24),
            wm_features=torch.zeros(2, 0),
            actions=torch.zeros(2, 1),
            rewards=torch.tensor([-1.0, -2.0]),
            next_windows=torch.zeros(2, 2, 24),
            next_wm_features=torch.zeros(2, 0),
            dones=torch.tensor([0.0, 0.0]),
            horizons=torch.tensor([1.0, 20.0]),
        )
        q_next = torch.tensor([-10.0, -10.0])
        targets = batch.rewards + (1.0 - batch.dones) * torch.pow(
            gamma, batch.horizons
        ) * q_next
        self.assertAlmostEqual(float(targets[0]), -1.0 - gamma * 10.0, places=5)
        self.assertAlmostEqual(
            float(targets[1]), -2.0 - (gamma ** 20) * 10.0, places=5
        )


class CriticNormTests(unittest.TestCase):
    def test_input_norm_changes_head_only(self):
        plain = TwinCritic(2, 0)
        normed = TwinCritic(2, 0, input_norm=True)
        self.assertIsInstance(normed.q1[0], torch.nn.LayerNorm)
        self.assertEqual(normed.q1[0].normalized_shape[0], 2 * 24 + 1)
        self.assertEqual(len(plain.q1), 5)
        self.assertEqual(len(normed.q1), 6)
        out = normed(torch.randn(4, 2, 24), torch.zeros(4, 0), torch.zeros(4, 1))
        self.assertEqual(out[0].shape, (4, 1))


if __name__ == "__main__":
    unittest.main()
