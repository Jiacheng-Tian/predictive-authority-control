import unittest

import numpy as np


class SeededEvaluationVariationTest(unittest.TestCase):
    def test_fixed_controller_eval_seed_changes_closed_loop_trajectory(self):
        from pac.evaluation.episodes import run_fixed_controller_episode

        common = {
            "scenario": 2,
            "steps": 120,
            "mass_scale_xy": 1.0,
            "damping_scale_xy": 1.0,
            "base_controller": "real10kg_smc_steady",
            "current_amplitude_scale": 2.5,
            "current_frequency_scale": 1.0,
            "initial_position_std": 0.03,
            "initial_velocity_std": 0.01,
            "vertical_current": 0.75,
            "vehicle_profile": "real_10kg_v1",
            "thruster_layout": "real_10kg_x",
            "save_ts": True,
        }
        seed0 = run_fixed_controller_episode(seed=0, **common)["ts"]
        seed1 = run_fixed_controller_episode(seed=1, **common)["ts"]
        rerun0 = run_fixed_controller_episode(seed=0, **common)["ts"]

        cols = ["x", "y", "z", "error"]
        seed_delta = np.nanmax(np.abs(seed0[cols].to_numpy() - seed1[cols].to_numpy()))
        rerun_delta = np.nanmax(np.abs(seed0[cols].to_numpy() - rerun0[cols].to_numpy()))

        self.assertGreater(seed_delta, 1.0e-6)
        self.assertAlmostEqual(rerun_delta, 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
