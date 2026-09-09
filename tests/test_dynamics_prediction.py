from __future__ import annotations

import unittest

import numpy as np


class DynamicsPredictionTest(unittest.TestCase):
    def test_predict_step_matches_step_without_mutating_state(self):
        from pac.simulation.dynamics import AUVDynamics

        dynamics = AUVDynamics()
        eta = np.array([0.3, -0.2, 0.1, 0.05, -0.03, 0.4])
        nu = np.array([0.7, -0.4, 0.2, 0.1, -0.3, 0.5])
        tau = np.array([1.0, -2.0, 3.0, 0.4, -0.5, 0.6])
        current = np.array([0.2, -0.1, 0.05])
        dynamics.eta = eta.copy()
        dynamics.nu = nu.copy()

        predicted_eta, predicted_nu = dynamics.predict_step(eta, nu, tau, current)
        np.testing.assert_array_equal(dynamics.eta, eta)
        np.testing.assert_array_equal(dynamics.nu, nu)

        stepped_eta, stepped_nu = dynamics.step(tau, current)
        np.testing.assert_allclose(predicted_eta, stepped_eta, rtol=0.0, atol=1.0e-14)
        np.testing.assert_allclose(predicted_nu, stepped_nu, rtol=0.0, atol=1.0e-14)

    def test_custom_dt_is_validated_and_used(self):
        from pac.simulation.core import AUVSimulator
        from pac.simulation.dynamics import AUVDynamics

        self.assertAlmostEqual(AUVDynamics(dt=0.02).dt, 0.02)
        simulator = AUVSimulator(scenario=1, dt=0.02)
        self.assertAlmostEqual(simulator.dynamics.dt, 0.02)
        for value in (0.0, -0.1, np.nan, np.inf, 10**1000):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    AUVDynamics(dt=value)


if __name__ == "__main__":
    unittest.main()
