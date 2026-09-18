"""Tests for the v4 world model: physics parity, model, dataset, training."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

from pac.simulation.dynamics import AUVDynamics
from pac.v4.worldmodel.data import (
    FEATURE_DIM,
    STATE_DIM,
    TransitionWindows,
    assemble_wm_feature,
    build_transition_windows,
)
from pac.v4.worldmodel.model import (
    EnsembleDynamicsModel,
    ResidualDynamicsModel,
    load_world_model,
    save_world_model,
)
from pac.v4.worldmodel.torch_dynamics import TorchAUVDynamics


class TorchPhysicsParityTests(unittest.TestCase):
    def test_batched_rk4_matches_numpy(self):
        """Strict parity inside the physical pitch envelope.

        Outside |theta| < pi/2 the frozen v3 dynamics clip negative
        ``cos(theta)`` to +1e-6, amplifying any last-ulp trigonometry
        difference by ~1e6; the vehicle never reaches that envelope
        (restoring stiffness keeps pitch near 0), so parity is asserted on
        the operating domain.
        """
        numpy_dynamics = AUVDynamics(dt=0.01)
        torch_dynamics = TorchAUVDynamics(dt=0.01)
        rng = np.random.default_rng(0)
        count = 128
        eta = rng.normal(0.0, 0.4, size=(count, 6))
        eta[:, 3] = rng.uniform(-np.pi, np.pi, size=count)
        eta[:, 4] = rng.uniform(-1.2, 1.2, size=count)
        eta[:, 5] = rng.uniform(-np.pi, np.pi, size=count)
        nu = rng.normal(0.0, 0.4, size=(count, 6))
        tau = rng.normal(0.0, 5.0, size=(count, 6))
        current = rng.normal(0.0, 0.3, size=(count, 3))
        expected = np.empty((count, 12))
        for index in range(count):
            next_eta, next_nu = numpy_dynamics.predict_step(
                eta[index], nu[index], tau[index], current[index]
            )
            expected[index] = np.concatenate([next_eta, next_nu])
        next_eta, next_nu = torch_dynamics.predict_step(
            torch.tensor(eta), torch.tensor(nu), torch.tensor(tau), torch.tensor(current)
        )
        actual = torch.cat([next_eta, next_nu], dim=1).numpy()
        self.assertLess(float(np.max(np.abs(actual - expected))), 1.0e-9)

    def test_angle_wrap_matches(self):
        numpy_dynamics = AUVDynamics(dt=0.01)
        torch_dynamics = TorchAUVDynamics(dt=0.01)
        eta = np.array([[0.0, 0.0, 0.0, 3.0, -3.0, 3.1]])
        nu = np.zeros((1, 6))
        tau = np.zeros((1, 6))
        expected_eta, _ = numpy_dynamics.predict_step(eta[0], nu[0], tau[0], np.zeros(3))
        next_eta, _ = torch_dynamics.predict_step(
            torch.tensor(eta), torch.tensor(nu), torch.tensor(tau), torch.zeros((1, 3))
        )
        np.testing.assert_allclose(next_eta.numpy()[0], expected_eta, atol=1.0e-12)


class ResidualModelTests(unittest.TestCase):
    def _model(self, **overrides):
        values = dict(
            history_len=4, hidden_dim=8, num_layers=1, residual_hidden_dim=8
        )
        values.update(overrides)
        return ResidualDynamicsModel(**values)

    def test_zero_initialized_head_predicts_zero(self):
        model = self._model()
        windows = torch.randn(3, 4, FEATURE_DIM)
        delta_state, delta_current = model.predict_delta(windows)
        self.assertEqual(delta_state.shape, (3, STATE_DIM))
        self.assertEqual(delta_current.shape, (3, 2))
        self.assertLess(float(delta_state.abs().max()), 1.0e-8)
        self.assertLess(float(delta_current.abs().max()), 1.0e-8)

    def test_forward_validates_shape(self):
        model = self._model()
        with self.assertRaises(ValueError):
            model(torch.randn(2, 4, FEATURE_DIM + 1))

    def test_normalization_contract(self):
        model = self._model()
        with torch.no_grad():
            model.head[-1].weight.fill_(0.1)
            model.head[-1].bias.fill_(0.0)
        model.configure_normalization(
            feature_mean=np.full(FEATURE_DIM, 0.1, dtype=np.float32),
            feature_scale=np.full(FEATURE_DIM, 2.0, dtype=np.float32),
            residual_mean=np.zeros(STATE_DIM + 2, dtype=np.float32),
            residual_scale=np.full(STATE_DIM + 2, 0.5, dtype=np.float32),
        )
        normalized = model(torch.full((1, 4, FEATURE_DIM), 1.1))
        self.assertGreater(float(normalized.abs().max()), 0.0)
        with self.assertRaises(ValueError):
            model.configure_normalization(
                feature_mean=np.zeros(FEATURE_DIM - 1, dtype=np.float32),
                feature_scale=np.ones(FEATURE_DIM, dtype=np.float32),
                residual_mean=np.zeros(STATE_DIM + 2, dtype=np.float32),
                residual_scale=np.ones(STATE_DIM + 2, dtype=np.float32),
            )

    def test_ensemble_mean_and_std(self):
        first = self._model()
        second = self._model()
        with torch.no_grad():
            second.head[-1].weight.fill_(0.01)
            second.head[-1].bias.fill_(0.02)
        ensemble = EnsembleDynamicsModel([first, second], [1, 2])
        windows = torch.randn(5, 4, FEATURE_DIM)
        prediction = ensemble.predict_delta(windows)
        self.assertEqual(prediction.mean_delta_state.shape, (5, STATE_DIM))
        self.assertEqual(prediction.std_delta_state.shape, (5, STATE_DIM))
        self.assertTrue(
            torch.all(prediction.std_delta_current >= 0.0),
            "ensemble std must be non-negative",
        )
        with self.assertRaises(ValueError):
            EnsembleDynamicsModel([], [])

    def test_save_load_roundtrip(self):
        model = self._model()
        ensemble = EnsembleDynamicsModel([model], [91000])
        windows = torch.randn(2, 4, FEATURE_DIM)
        before = ensemble.predict_delta(windows)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "wm.pt"
            save_world_model(
                path,
                ensemble,
                dataset_hash="abc123",
                member_training_metrics=[{"best_val_loss": 0.5, "member_seed": 91000}],
                config_hash="cfg456",
            )
            with self.assertRaises(FileExistsError):
                save_world_model(
                    path,
                    ensemble,
                    dataset_hash="abc123",
                    member_training_metrics=[{"best_val_loss": 0.5}],
                    config_hash="cfg456",
                )
            loaded, metadata = load_world_model(path)
            self.assertEqual(metadata["dataset_hash"], "abc123")
            after = loaded.predict_delta(windows)
            np.testing.assert_allclose(
                before.mean_delta_state.numpy(),
                after.mean_delta_state.numpy(),
                atol=1.0e-12,
            )
            with self.assertRaises(ValueError):
                load_world_model(path, expected_dataset_hash="wrong")


class FeatureAndWindowTests(unittest.TestCase):
    def test_assemble_feature_layout(self):
        feature = assemble_wm_feature(
            np.arange(12, dtype=float),
            np.zeros(6),
            np.ones(6),
            np.array([0.1, 0.2, 0.3]),
            np.array([1.0, 1.0, 2.5, 1.0]),
            2,
        )
        self.assertEqual(feature.shape, (FEATURE_DIM,))
        np.testing.assert_allclose(feature[:12], np.arange(12), atol=0.0)
        self.assertAlmostEqual(float(feature[12]), 0.0)
        self.assertAlmostEqual(float(feature[18]), 1.0)
        self.assertAlmostEqual(float(feature[24]), 0.1)
        self.assertAlmostEqual(float(feature[27]), 1.0)
        self.assertAlmostEqual(float(feature[31]), 2.0 / 3.0)

    def _fake_dataset(self):
        class FakeDataset:
            pass

        dataset = FakeDataset()
        dataset.features = np.random.default_rng(1).normal(
            0.0, 0.1, size=(10, FEATURE_DIM)
        ).astype(np.float32)
        dataset.state = np.zeros((10, STATE_DIM), dtype=np.float32)
        dataset.next_state = np.zeros((10, STATE_DIM), dtype=np.float32)
        dataset.physics_next_state = np.zeros((10, STATE_DIM), dtype=np.float32)
        dataset.true_current = np.zeros((10, 3), dtype=np.float32)
        dataset.next_true_current = np.zeros((10, 3), dtype=np.float32)
        dataset.metadata = pd.DataFrame({
            "episode_uid": ["a"] * 4 + ["b"] * 6,
            "split": ["train"] * 10,
            "step": list(range(4)) + list(range(6)),
        })
        return dataset

    def test_windows_do_not_cross_episodes_and_pad_front(self):
        dataset = self._fake_dataset()
        split = build_transition_windows(dataset, history_len=4)
        self.assertEqual(split.windows.shape, (10, 4, FEATURE_DIM))
        # episode 'a' has only 4 rows: every window ends at its own row.
        for offset in range(4):
            np.testing.assert_allclose(
                split.windows[offset, -1, :], dataset.features[offset], atol=0.0
            )
            padding_rows = 4 - (offset + 1)
            if padding_rows > 0:
                np.testing.assert_allclose(
                    split.windows[offset, :padding_rows, :], 0.0, atol=0.0
                )
        # row 4 starts episode 'b': its window must not contain episode-a rows.
        np.testing.assert_allclose(split.windows[4, :3, :], 0.0, atol=0.0)
        np.testing.assert_allclose(
            split.windows[4, -1, :], dataset.features[4], atol=0.0
        )
        np.testing.assert_allclose(
            split.windows[9, :, :], dataset.features[6:10], atol=0.0
        )

    def test_residual_target(self):
        dataset = self._fake_dataset()
        dataset.next_state = np.full((10, STATE_DIM), 1.0, dtype=np.float32)
        dataset.physics_next_state = np.full((10, STATE_DIM), 0.25, dtype=np.float32)
        split = build_transition_windows(dataset, history_len=2)
        np.testing.assert_allclose(split.residual_target, 0.75, atol=0.0)


if __name__ == "__main__":
    unittest.main()
