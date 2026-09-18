"""Tests for v4 transition collection and candidate-alpha ranking."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

from pac.v4.collector import (
    FEATURE_DIM,
    PLAN_HORIZON,
    TransitionArrays,
    _collect_single_plan,
    load_transition_dataset,
    resolve_collection_plans,
    save_transition_dataset,
)
from pac.v4.config import load_v4_config
from pac.v4.worldmodel.ranking import (
    CandidateRolloutEvaluator,
    RankingResult,
    physics_persistence_costs,
    select_ranking_windows,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "pac_v4.yaml"


def _plan(**overrides) -> dict:
    values = {
        "split": "train",
        "seed": 21000,
        "family": "structured",
        "scenario_id": 1,
        "behavior": "constant_alpha",
        "behavior_alpha": 0.5,
        "steps": 40,
    }
    values.update(overrides)
    return values


def _to_transition_arrays(result: dict) -> TransitionArrays:
    return TransitionArrays(
        features=result["features"],
        state=result["state"],
        next_state=result["next_state"],
        physics_next_state=result["physics_next_state"],
        requested=result["requested"],
        applied=result["applied"],
        est_current=result["est_current"],
        true_current=result["true_current"],
        next_true_current=result["next_true_current"],
        context=result["context"],
        solver=result["solver"],
        plan=result["plan"],
        alpha=result["alpha"],
        metadata=pd.DataFrame(result["metadata"]),
    )


class CollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_v4_config(CONFIG_PATH)

    def test_collect_single_plan_shapes_and_finiteness(self):
        result = _collect_single_plan(self.config, _plan(steps=30))
        self.assertEqual(result["features"].shape[1], FEATURE_DIM)
        self.assertEqual(result["plan"].shape[1:], (PLAN_HORIZON, 6))
        for name in (
            "features", "state", "next_state", "physics_next_state", "requested",
            "applied", "est_current", "true_current", "next_true_current",
            "context", "solver", "plan", "alpha",
        ):
            self.assertTrue(np.isfinite(result[name]).all(), f"{name} must be finite")
        self.assertEqual(len(result["metadata"]), 30)
        accepted = np.array(
            [bool(row["plan_accepted"]) for row in result["metadata"]]
        )
        self.assertGreater(
            accepted.mean(), 0.3,
            "online MPC should accept most plans on the nominal scenario",
        )

    def test_physics_target_tracks_realized_state(self):
        result = _collect_single_plan(self.config, _plan(steps=30))
        residual = result["next_state"] - result["physics_next_state"]
        # Physics is near-exact on the nominal scenario: residuals stay tiny.
        self.assertLess(float(np.abs(residual).max()), 5.0e-2)

    def test_ou_current_collection_is_finite(self):
        result = _collect_single_plan(
            self.config,
            _plan(family="ou_current", scenario_id=2, steps=30, behavior_alpha=1.0),
        )
        self.assertTrue(np.isfinite(result["features"]).all())
        self.assertGreater(
            float(np.std(result["true_current"][:, 0])), 0.0,
            "OU currents must vary over the episode",
        )

    def test_dataset_save_load_roundtrip(self):
        result = _collect_single_plan(self.config, _plan(steps=20))
        arrays = _to_transition_arrays(result)
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "dataset"
            manifest = save_transition_dataset(
                arrays, out_dir, provenance={"profile": "test"}
            )
            self.assertIn("transitions.npz", manifest["file_sha256"])
            loaded = load_transition_dataset(out_dir)
            self.assertEqual(loaded.features.shape, arrays.features.shape)
            np.testing.assert_allclose(loaded.features, result["features"], atol=0.0)
            self.assertEqual(
                list(loaded.metadata["episode_uid"].unique()), [result["episode_uid"]]
            )
            self.assertEqual(loaded.metadata["plan_accepted"].dtype, bool)
            with self.assertRaises(FileExistsError):
                save_transition_dataset(arrays, out_dir, provenance={"profile": "test"})

    def test_short_profile_plan_resolution(self):
        plans = resolve_collection_plans(self.config, "short")
        self.assertTrue(all(plan["steps"] == 60 for plan in plans))
        families = {plan["family"] for plan in plans}
        self.assertEqual(families, {"structured", "ou_current"})
        with self.assertRaises(ValueError):
            resolve_collection_plans(self.config, "unknown")


class RankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_v4_config(CONFIG_PATH)
        result = _collect_single_plan(
            cls.config,
            _plan(steps=90, behavior="transformer", behavior_alpha=float("nan")),
        )
        cls.dataset = _to_transition_arrays(result)

    def test_ranking_result_metrics(self):
        true_costs = np.array(
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        )
        perfect = RankingResult(true_costs, true_costs.copy(), "wm")
        self.assertTrue(perfect.top1_correct)
        self.assertAlmostEqual(perfect.spearman, 1.0)
        self.assertAlmostEqual(perfect.regret, 0.0)
        reversed_costs = true_costs[::-1].copy()
        worst = RankingResult(true_costs, reversed_costs, "wm")
        self.assertFalse(worst.top1_correct)
        self.assertAlmostEqual(worst.regret, 1.0)
        noisy = true_costs.copy()
        noisy[10] += 0.5  # degrade the true best so the argmin moves to index 9
        middle = RankingResult(true_costs, noisy, "wm")
        self.assertFalse(middle.top1_correct)
        self.assertAlmostEqual(middle.regret, 0.1)

    def test_select_ranking_windows_alignment(self):
        windows = select_ranking_windows(
            self.dataset,
            split="train",
            windows_per_episode=4,
            horizon=20,
            history_len=16,
        )
        self.assertTrue(windows)
        for window in windows:
            self.assertEqual(window.window.shape, (16, FEATURE_DIM))
            self.assertEqual(window.plan.shape, (20, 6))
            if window.row >= 1:
                np.testing.assert_allclose(
                    window.window[-1, :],
                    self.dataset.features[window.row - 1, :],
                    atol=0.0,
                )
            np.testing.assert_allclose(
                window.initial_state, self.dataset.state[window.row], atol=0.0
            )
            self.assertEqual(window.realized_currents.shape, (20, 3))

    def test_true_oracle_costs_match_persistence_on_constant_current(self):
        windows = select_ranking_windows(
            self.dataset,
            split="train",
            windows_per_episode=1,
            horizon=20,
            history_len=16,
        )
        evaluator = CandidateRolloutEvaluator(self.config, device="cpu")
        window = windows[0]
        true_costs = evaluator.true_oracle_costs(window)
        persistence_costs = physics_persistence_costs(evaluator, window)
        self.assertEqual(true_costs.shape, (11,))
        self.assertTrue(np.isfinite(true_costs).all())
        self.assertTrue(np.isfinite(persistence_costs).all())
        # Scenario 1 has a constant current: only current-estimation noise
        # separates the persistence rollout from the realized rollout.
        scale = float(np.max(np.abs(true_costs)) + 1.0e-9)
        self.assertLess(
            float(np.max(np.abs(true_costs - persistence_costs))) / scale, 5.0e-3
        )

    def test_single_model_cost_shape(self):
        windows = select_ranking_windows(
            self.dataset,
            split="train",
            windows_per_episode=1,
            horizon=20,
            history_len=16,
        )
        evaluator = CandidateRolloutEvaluator(self.config, device="cpu")

        class TinyModel:
            def predict_delta(self, batch_windows):
                batch = batch_windows.shape[0]
                return torch.zeros(batch, 12), torch.zeros(batch, 2)

        costs = evaluator.single_model_costs(TinyModel(), windows[0])
        self.assertEqual(costs.shape, (11,))
        self.assertTrue(np.isfinite(costs).all())


if __name__ == "__main__":
    unittest.main()
