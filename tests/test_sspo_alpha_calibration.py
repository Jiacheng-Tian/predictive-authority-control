import unittest
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class SSPOAlphaCalibrationTest(unittest.TestCase):
    def test_alpha_bias_schedule_is_added_by_evaluation_window_and_clipped(self):
        from evaluation import predictive_3d_authority_alpha as pac

        schedule = {
            "startup_0_3s": 0.20,
            "pre_step_3_10s": -0.15,
            "step_recovery_10_13s": 0.50,
            "post_step_13_20p9s": -0.20,
        }

        self.assertAlmostEqual(pac.calibrate_alpha_with_bias_schedule(0.70, 1.0, schedule), 0.90)
        self.assertAlmostEqual(pac.calibrate_alpha_with_bias_schedule(0.10, 4.0, schedule), 0.0)
        self.assertAlmostEqual(pac.calibrate_alpha_with_bias_schedule(0.70, 11.0, schedule), 1.0)
        self.assertAlmostEqual(pac.calibrate_alpha_with_bias_schedule(0.70, 15.0, schedule), 0.50)

    def test_positive_window_regret_compares_against_best_fixed_controller(self):
        from scripts import sspo_alpha_calibration as sspo

        candidate = pd.DataFrame([
            {"scenario": "constant", "window": "startup_0_3s", "rmse_3d": 0.11},
            {"scenario": "constant", "window": "pre_step_3_10s", "rmse_3d": 0.18},
        ])
        fixed = pd.DataFrame([
            {"scenario": "constant", "window": "startup_0_3s", "controller": "real10kg_smc_steady", "rmse_3d": 0.10},
            {"scenario": "constant", "window": "startup_0_3s", "controller": "real10kg_mpc_event", "rmse_3d": 0.12},
            {"scenario": "constant", "window": "pre_step_3_10s", "controller": "real10kg_smc_steady", "rmse_3d": 0.20},
            {"scenario": "constant", "window": "pre_step_3_10s", "controller": "real10kg_mpc_event", "rmse_3d": 0.16},
        ])

        regret = sspo.mean_positive_window_regret(candidate, fixed)

        self.assertAlmostEqual(regret, (0.01 / 0.10 + 0.02 / 0.16) / 2.0)

    def test_candidate_score_penalizes_rmse_guard_violation(self):
        from scripts import sspo_alpha_calibration as sspo

        lower_regret_but_worse_rmse = sspo.score_candidate(
            overall_rmse=0.11,
            positive_window_regret=0.0,
            baseline_rmse=0.09,
            window_regret_weight=0.01,
            rmse_guard_weight=5.0,
        )
        higher_regret_but_baseline_rmse = sspo.score_candidate(
            overall_rmse=0.09,
            positive_window_regret=1.0,
            baseline_rmse=0.09,
            window_regret_weight=0.01,
            rmse_guard_weight=5.0,
        )

        self.assertGreater(lower_regret_but_worse_rmse, higher_regret_but_baseline_rmse)


if __name__ == "__main__":
    unittest.main()
