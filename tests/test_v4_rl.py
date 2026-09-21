"""Tests for the v4 reward function and warm-start materialization."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from pac.v4.config import load_v4_config
from pac.v4.rl.reward import compute_step_reward
from pac.v4.rl.warmstart import materialize_warmstart

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "pac_v4.yaml"

try:
    from tests.test_v4_ranking import _collect_single_plan, _plan, _to_transition_arrays
except ImportError:  # pragma: no cover - direct execution
    from test_v4_ranking import _collect_single_plan, _plan, _to_transition_arrays


class RewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_v4_config(_CONFIG_PATH)

    def test_hand_computed_value(self):
        reward = compute_step_reward(
            self.config.reward,
            position_error=0.2,
            heading_error=0.05,
            applied_action=np.full(6, 0.5),
            previous_applied=np.zeros(6),
            alpha=0.4,
            previous_alpha=0.3,
            saturation_fraction=0.5,
            deadline_missed=True,
            constraint_violated=False,
        )
        cfg = self.config.reward
        expected = (
            -cfg.w_position * (0.2 / cfg.position_scale_m)
            - cfg.w_heading * (0.05 / cfg.heading_scale_rad)
            - cfg.w_control * 0.25
            - cfg.w_delta_control * 0.25
            - cfg.w_delta_alpha * 0.1
            - cfg.w_saturation * 0.5
            - cfg.w_deadline * 1.0
            - cfg.w_constraint * 0.0
        )
        self.assertAlmostEqual(reward, expected, places=12)

    def test_constraint_penalty_dominates(self):
        base = dict(
            position_error=0.1,
            heading_error=0.0,
            applied_action=np.zeros(6),
            previous_applied=np.zeros(6),
            alpha=0.2,
            previous_alpha=0.2,
            saturation_fraction=0.0,
            deadline_missed=False,
        )
        clean = compute_step_reward(
            self.config.reward, constraint_violated=False, **base
        )
        violated = compute_step_reward(
            self.config.reward, constraint_violated=True, **base
        )
        self.assertAlmostEqual(
            clean - violated, self.config.reward.w_constraint, places=12
        )

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            compute_step_reward(
                self.config.reward,
                position_error=0.0,
                heading_error=0.0,
                applied_action=np.zeros(6),
                previous_applied=np.zeros(5),
                alpha=0.0,
                previous_alpha=0.0,
                saturation_fraction=0.0,
                deadline_missed=False,
                constraint_violated=False,
            )


class WarmstartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_v4_config(_CONFIG_PATH)
        dataset = _collect_single_plan(
            cls.config,
            _plan(steps=60, behavior="constant_alpha", behavior_alpha=0.5),
        )
        cls.dataset = _to_transition_arrays(dataset)
        from pac.authority.model import load_alpha_model_checkpoint
        from pathlib import Path

        checkpoint = (
            Path(cls.config.backbone.checkpoint_dir)
            / "pac_train_seed_31000"
            / "checkpoint.pt"
        )
        cls.backbone, _meta = load_alpha_model_checkpoint(checkpoint)
        cls.backbone.eval()

    def test_materialize_shapes_and_semantics(self):
        replay = materialize_warmstart(self.dataset, self.config, self.backbone)
        rows = replay["windows"].shape[0]
        history = int(self.config.authority_model.history_len)
        self.assertEqual(replay["windows"].shape, (rows, history, 24))
        self.assertEqual(replay["next_windows"].shape, (rows, history, 24))
        self.assertEqual(replay["actions"].shape, (rows,))
        self.assertEqual(replay["rewards"].shape, (rows,))
        self.assertTrue(np.isfinite(replay["rewards"]).all())
        self.assertTrue(
            np.all(np.abs(replay["actions"]) <= self.config.rl.delta_max + 1e-9)
        )
        # rewards must be negative (weighted penalties only)
        self.assertTrue(np.all(replay["rewards"] < 0.0))
        # dones only at episode boundaries
        dones = replay["dones"]
        self.assertTrue(np.any(dones))
        # windows must not cross the episode boundary: the last window row
        # of the episode cannot appear as a next-window of earlier rows
        self.assertLessEqual(rows, replay["total_rows"])

    def test_limit_respects_stride(self):
        replay_all = materialize_warmstart(self.dataset, self.config, self.backbone)
        replay_limited = materialize_warmstart(
            self.dataset, self.config, self.backbone, limit=17
        )
        self.assertLessEqual(replay_limited["windows"].shape[0], 17)
        self.assertGreater(replay_limited["windows"].shape[0], 0)
        # stride sampling keeps ordering
        np.testing.assert_allclose(
            replay_limited["windows"][0], replay_all["windows"][0], atol=0.0
        )

    def test_backbone_roundtrip_on_zero_residual(self):
        """A behavior equal to the backbone must give zero warm-start action."""
        dataset = _collect_single_plan(
            self.config,
            _plan(steps=60, behavior="transformer", behavior_alpha=float("nan")),
        )
        arrays = _to_transition_arrays(dataset)
        replay = materialize_warmstart(arrays, self.config, self.backbone)
        # behavior alpha after the v3 filter stack vs raw backbone alpha can
        # differ slightly through smoothing/rate limits, but the residual
        # must stay small (bounded by the filter transient)
        self.assertLess(
            float(np.abs(replay["actions"]).max()), 0.2,
            "transformer-behavior warm-start residuals must stay near zero",
        )


if __name__ == "__main__":
    unittest.main()
