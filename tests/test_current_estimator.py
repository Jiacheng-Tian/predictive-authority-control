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


if __name__ == "__main__":
    unittest.main()
