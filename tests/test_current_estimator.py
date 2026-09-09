from __future__ import annotations

import unittest

import numpy as np


class CurrentEstimatorTest(unittest.TestCase):
    def test_delay_uses_only_past_current_and_pregenerated_noise(self):
        from pac.simulation.observations import CausalCurrentEstimator

        noise = np.zeros((4, 3), dtype=float)
        estimator = CausalCurrentEstimator(delay_steps=2, noise=noise)
        estimator.reset(np.array([9.0, 9.0, 9.0]))
        values = [
            estimator.estimate(np.array([1.0, 0.0, 0.0]), 0),
            estimator.estimate(np.array([2.0, 0.0, 0.0]), 1),
            estimator.estimate(np.array([3.0, 0.0, 0.0]), 2),
        ]
        np.testing.assert_array_equal(values[0], [9.0, 9.0, 9.0])
        np.testing.assert_array_equal(values[1], [9.0, 9.0, 9.0])
        np.testing.assert_array_equal(values[2], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(estimator.telemetry["true_current"][:, 0], [1.0, 2.0, 3.0])
        self.assertFalse(estimator.telemetry["estimate"].flags.writeable)

    def test_estimator_requires_contiguous_integer_steps_without_mutation_on_failure(self):
        from pac.simulation.observations import CausalCurrentEstimator

        for delay_steps in (0, 2):
            estimator = CausalCurrentEstimator(
                delay_steps=delay_steps,
                noise=np.zeros((3, 3), dtype=float),
            )
            estimator.reset(np.array([9.0, 9.0, 9.0]))
            self.assertEqual(estimator.expected_step, 0)
            estimator.estimate(np.array([1.0, 0.0, 0.0]), 0)
            before = {
                key: value.copy() for key, value in estimator.telemetry.items()
            }
            self.assertEqual(estimator.expected_step, 1)
            for invalid_step in (2, -1, 3, 0, 0.0):
                with self.assertRaises(ValueError):
                    estimator.estimate(np.array([2.0, 0.0, 0.0]), invalid_step)
                self.assertEqual(estimator.expected_step, 1)
                for key, value in before.items():
                    np.testing.assert_array_equal(estimator.telemetry[key], value)

            estimator.estimate(np.array([2.0, 0.0, 0.0]), 1)
            with self.assertRaises(ValueError):
                estimator.estimate(np.array([2.5, 0.0, 0.0]), 1)
            estimator.estimate(np.array([3.0, 0.0, 0.0]), 2)
            self.assertEqual(estimator.expected_step, 3)
            with self.assertRaises(ValueError):
                estimator.estimate(np.array([4.0, 0.0, 0.0]), 3)


if __name__ == "__main__":
    unittest.main()
