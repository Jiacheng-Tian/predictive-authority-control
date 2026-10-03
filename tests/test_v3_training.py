from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import random
import torch


ROOT = Path(__file__).resolve().parents[1]


def _dataset(*, validation_extreme: bool = False):
    from pac.authority.dataset import OracleDataset

    rows = []
    features = []
    labels = []
    for split, episode, base in (("train", "train-a", 1.0), ("train", "train-b", 2.0), ("val", "val-a", 100.0)):
        for step in range(4):
            value = base + step / 10.0
            if validation_extreme and split == "val":
                value = 1000.0 + step
            features.append(np.full(24, value, dtype=np.float32))
            labels.append(float((step % 2) * 0.5))
            rows.append({
                "split": split,
                "scenario": 1,
                "environment_seed": 100 if split == "train" else 200,
                "episode_uid": episode,
                "step": step,
                "sample_time": step * 0.01,
                "teacher_alpha": labels[-1],
                "oracle_best": 0.0,
                "oracle_worst": 1.0,
                "episode_fingerprint": f"fingerprint-{episode}",
            })
    return OracleDataset(np.asarray(features), np.asarray(labels), pd.DataFrame(rows))


class SupervisedTrainingTest(unittest.TestCase):
    def test_training_restores_all_global_rng_and_determinism_state(self):
        from pac.authority.model import train_alpha_model_supervised
        from pac.experiment_config import load_supervised_config

        config = load_supervised_config(ROOT / "config" / "pac_supervised.yaml").authority_model
        random.seed(8123)
        np.random.seed(8123)
        torch.manual_seed(8123)
        torch.use_deterministic_algorithms(False)
        before_python = random.getstate()
        before_numpy = np.random.get_state()
        before_torch = torch.get_rng_state().clone()
        expected_python = random.Random()
        expected_python.setstate(before_python)
        expected_python_value = expected_python.random()
        expected_numpy = np.random.RandomState()
        expected_numpy.set_state(before_numpy)
        expected_numpy_value = float(expected_numpy.random_sample())
        expected_torch = torch.Generator()
        expected_torch.set_state(before_torch)
        expected_torch_value = float(torch.rand((), generator=expected_torch).item())

        train_alpha_model_supervised(_dataset(), config, model_seed=31000, max_epochs=1, patience=1)

        self.assertEqual(random.random(), expected_python_value)
        self.assertEqual(float(np.random.random_sample()), expected_numpy_value)
        self.assertEqual(float(torch.rand(()).item()), expected_torch_value)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch.get_rng_state()))
        self.assertFalse(torch.are_deterministic_algorithms_enabled())

        random.seed(8123)
        np.random.seed(8123)
        torch.manual_seed(8123)
        before_python = random.getstate()
        before_numpy = np.random.get_state()
        before_torch = torch.get_rng_state().clone()
        with self.assertRaises(ValueError):
            train_alpha_model_supervised(_dataset(), config, model_seed=31000, max_epochs=0, patience=1)
        self.assertEqual(random.getstate(), before_python)
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0], before_numpy[0])
        np.testing.assert_array_equal(after_numpy[1], before_numpy[1])
        self.assertEqual(after_numpy[2:], before_numpy[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), before_torch))

    def test_train_uses_train_only_normalization_and_episode_windows(self):
        from pac.authority.model import train_alpha_model_supervised
        from pac.experiment_config import load_supervised_config

        config = load_supervised_config(ROOT / "config" / "pac_supervised.yaml").authority_model
        model, history, metrics = train_alpha_model_supervised(
            _dataset(validation_extreme=True), config, model_seed=31000, max_epochs=2, patience=1
        )

        np.testing.assert_allclose(model.feature_mean.detach().numpy(), np.full(24, 1.65), atol=1e-6)
        self.assertNotAlmostEqual(float(model.feature_mean[0]), 500.0, places=2)
        self.assertEqual(list(history.columns), ["epoch", "train_mse", "val_mse"])
        self.assertGreaterEqual(len(history), 1)
        self.assertEqual(metrics["param_count"], 14113)
        self.assertIn("best_epoch", metrics)
        self.assertIn("val_mse", metrics)

    def test_training_is_deterministic_for_same_seed(self):
        from pac.authority.model import train_alpha_model_supervised
        from pac.experiment_config import load_supervised_config

        config = load_supervised_config(ROOT / "config" / "pac_supervised.yaml").authority_model
        first = train_alpha_model_supervised(_dataset(), config, model_seed=31000, max_epochs=2, patience=1)
        second = train_alpha_model_supervised(_dataset(), config, model_seed=31000, max_epochs=2, patience=1)
        self.assertTrue(first[1].equals(second[1]))
        for name, tensor in first[0].state_dict().items():
            np.testing.assert_allclose(tensor.detach().numpy(), second[0].state_dict()[name].detach().numpy(), rtol=1e-6, atol=1e-7)

    def test_early_stopping_metrics_point_to_restored_best_epoch(self):
        from pac.authority.model import train_alpha_model_supervised
        from pac.experiment_config import load_supervised_config

        config = load_supervised_config(ROOT / "config" / "pac_supervised.yaml").authority_model
        _model, history, metrics = train_alpha_model_supervised(
            _dataset(validation_extreme=True), config, model_seed=31001, max_epochs=8, patience=1
        )
        best_row = history.loc[history["val_mse"].idxmin()]
        self.assertEqual(metrics["best_epoch"], int(best_row["epoch"]))
        self.assertAlmostEqual(metrics["val_mse"], float(best_row["val_mse"]), places=8)
        self.assertLessEqual(metrics["epochs_ran"], 8)


if __name__ == "__main__":
    unittest.main()
