from __future__ import annotations

import unittest

import numpy as np


class SimulationCoreTest(unittest.TestCase):
    def test_formal_simulator_rejects_nonformal_scenarios(self):
        from pac.simulation.core import AUVSimulator

        with self.assertRaisesRegex(ValueError, "scenario must be one of"):
            AUVSimulator(scenario=4)

    def test_thruster_layout_documents_rank_five_underactuation(self):
        from pac.simulation.thrusters import build_real_10kg_x_layout

        layout = build_real_10kg_x_layout()
        self.assertEqual(np.linalg.matrix_rank(layout.allocation_matrix), 5)
        np.testing.assert_array_equal(layout.allocation_matrix[4], np.zeros(6))

    def test_gym_wrapper_uses_formal_observation_and_truncation(self):
        from pac.simulation.environment import AUVTrackingEnv

        environment = AUVTrackingEnv(scenario=1, max_steps=1)
        observation, info = environment.reset(seed=0)
        self.assertEqual(observation.shape, (29,))
        self.assertEqual(info, {})

        next_observation, reward, terminated, truncated, step_info = environment.step(
            np.zeros(6),
        )
        self.assertEqual(next_observation.shape, (29,))
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertIn("dist_error", step_info)

    def test_gym_wrapper_rejects_removed_legacy_options(self):
        from pac.simulation.environment import AUVTrackingEnv

        with self.assertRaisesRegex(TypeError, "unsupported simulator options"):
            AUVTrackingEnv(scenario=1, action_mode="wrench")


if __name__ == "__main__":
    unittest.main()
