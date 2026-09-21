"""Parity and performance tests for the batched SMC / actuator primitives."""

from __future__ import annotations

import time
import unittest

import numpy as np

from pac.controllers.presets import build_real10kg_smc_steady
from pac.simulation.actuators import ActuatorLimits, SharedActuator
from pac.v4.batched import BatchedSMC, batched_actuator_apply, batched_blend


def _random_states(count: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    eta = rng.normal(0.0, 0.4, size=(count, 6))
    eta[:, 3] = rng.uniform(-np.pi, np.pi, size=count)
    eta[:, 4] = rng.uniform(-0.9, 0.9, size=count)
    eta[:, 5] = rng.uniform(-np.pi, np.pi, size=count)
    nu = rng.normal(0.0, 0.4, size=(count, 6))
    current = rng.normal(0.0, 0.3, size=(count, 3))
    return eta, nu, current


class BatchedSMCParityTests(unittest.TestCase):
    def setUp(self):
        self.controller = build_real10kg_smc_steady()
        self.controller.set_trajectory3d(True)
        self.batched = BatchedSMC(self.controller)

    def _target(self, t):
        return np.array([
            3.0 * np.sin(0.3 * t),
            1.5 * np.sin(0.6 * t),
            0.8 * np.sin(0.2 * t),
            0.0,
            0.0,
            np.arctan2(0.9 * np.cos(0.6 * t), 0.9 * np.cos(0.3 * t)),
        ])

    def test_batch_matches_scalar_compute(self):
        eta, nu, current = _random_states(64)
        for t in (0.0, 3.7, 12.915):
            target = self._target(t)
            batched = self.batched.compute_batch(target, eta, nu, t, current)
            for index in range(eta.shape[0]):
                scalar = self.controller.compute(
                    target, eta[index], nu[index], t=t,
                    current_prediction=current[index],
                )
                np.testing.assert_allclose(
                    batched[index], scalar, rtol=0.0, atol=1.0e-12,
                    err_msg=f"t={t} row={index}",
                )

    def test_batch_matches_scalar_without_current(self):
        eta, nu, _current = _random_states(32, seed=11)
        t = 5.5
        target = self._target(t)
        batched = self.batched.compute_batch(target, eta, nu, t, None)
        for index in range(eta.shape[0]):
            scalar = self.controller.compute(target, eta[index], nu[index], t=t)
            np.testing.assert_allclose(batched[index], scalar, atol=1.0e-12)

    def test_shape_validation(self):
        eta, nu, current = _random_states(4)
        target = self._target(0.0)
        with self.assertRaises(ValueError):
            self.batched.compute_batch(target, eta[:, :5], nu, 0.0, current)
        with self.assertRaises(ValueError):
            self.batched.compute_batch(target, eta, nu, 0.0, current[:2])


class BatchedActuatorParityTests(unittest.TestCase):
    def test_matches_shared_actuator(self):
        rng = np.random.default_rng(3)
        requested = rng.uniform(-1.6, 1.6, size=(48, 6))
        previous = rng.uniform(-1.0, 1.0, size=(48, 6))
        limits = ActuatorLimits(-1.0, 1.0, 0.10)
        applied, amplitude = batched_actuator_apply(
            requested, previous,
            command_min=limits.command_min,
            command_max=limits.command_max,
            max_delta_per_step=limits.max_delta_per_step,
        )
        for index in range(requested.shape[0]):
            actuator = SharedActuator(limits)
            actuator._previous_applied = previous[index].copy()
            step = actuator.apply(requested[index])
            np.testing.assert_allclose(
                applied[index], step.applied, atol=0.0
            )
            np.testing.assert_allclose(
                amplitude[index], step.amplitude_clipped, atol=0.0
            )

    def test_no_slew_limit_path(self):
        rng = np.random.default_rng(4)
        requested = rng.uniform(-1.0, 1.0, size=(8, 6))
        previous = np.zeros((8, 6))
        applied, amplitude = batched_actuator_apply(
            requested, previous,
            command_min=-1.0, command_max=1.0, max_delta_per_step=None,
        )
        np.testing.assert_allclose(applied, np.clip(requested, -1.0, 1.0), atol=0.0)
        np.testing.assert_allclose(amplitude, np.clip(requested, -1.0, 1.0), atol=0.0)

    def test_blend_matches_scalar(self):
        rng = np.random.default_rng(5)
        primary = rng.uniform(-1, 1, size=(11, 6))
        authority = rng.uniform(-1, 1, size=(11, 6))
        alphas = np.linspace(0.0, 1.0, 11)
        blended = batched_blend(primary, authority, alphas)
        for index in range(11):
            expected = np.clip(
                (1.0 - alphas[index]) * primary[index] + alphas[index] * authority[index],
                -1.0, 1.0,
            )
            np.testing.assert_allclose(blended[index], expected, atol=0.0)


class BatchedRolloutSpeedTests(unittest.TestCase):
    def test_candidate_rollout_primitive_budget(self):
        """Soft deployment gate: one batched SMC+actuator rollout step for
        11 candidates must stay far below the 10 ms control period."""
        controller = build_real10kg_smc_steady()
        controller.set_trajectory3d(True)
        batched = BatchedSMC(controller)
        eta, nu, current = _random_states(11)
        target = np.array([0.5, 0.2, 0.1, 0.0, 0.0, 0.3])
        previous = np.zeros((11, 6))
        alphas = np.linspace(0.0, 1.0, 11)
        # warm up caches, then measure 50 averaged steps
        for _ in range(5):
            primary = batched.compute_batch(target, eta, nu, 0.0, current)
            blended = batched_blend(primary, np.zeros_like(primary), alphas)
            applied, _ = batched_actuator_apply(
                blended, previous, command_min=-1.0, command_max=1.0,
                max_delta_per_step=0.10,
            )
        started = time.perf_counter()
        repeats = 50
        for _ in range(repeats):
            primary = batched.compute_batch(target, eta, nu, 0.0, current)
            blended = batched_blend(primary, np.zeros_like(primary), alphas)
            applied, _ = batched_actuator_apply(
                blended, previous, command_min=-1.0, command_max=1.0,
                max_delta_per_step=0.10,
            )
        per_step_ms = (time.perf_counter() - started) * 1000.0 / repeats
        self.assertLess(
            per_step_ms, 2.0,
            f"batched SMC + actuator step took {per_step_ms:.3f} ms",
        )


if __name__ == "__main__":
    unittest.main()
