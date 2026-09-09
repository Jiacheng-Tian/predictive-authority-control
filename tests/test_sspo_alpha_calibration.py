import unittest
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class SSPOAlphaCalibrationTest(unittest.TestCase):
    def test_dry_run_reports_archived_search_configuration_without_writing(self):
        from scripts import sspo_alpha_calibration as sspo

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "sspo"
            with redirect_stdout(io.StringIO()) as output:
                status = sspo.main([
                    "--config",
                    str(ROOT / "config" / "pac.yaml"),
                    "--out-dir",
                    str(out_dir),
                    "--dry-run",
                ])
            manifest = json.loads(output.getvalue())

        self.assertEqual(status, 0)
        self.assertFalse(out_dir.exists())
        self.assertEqual(manifest["bias_grid"], [-0.05, 0.0, 0.1, 0.2, 0.3])
        self.assertEqual(manifest["window_regret_weight"], 0.02)
        self.assertEqual(manifest["search_eval_overlap_seeds"], [20000, 20001, 20002])

    def test_nonempty_output_directory_is_rejected(self):
        from scripts import sspo_alpha_calibration as sspo

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "sspo"
            output.mkdir()
            (output / "existing.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "output directory is not empty"):
                sspo.require_empty_output_directory(output)

    def test_alpha_bias_schedule_is_added_by_evaluation_window_and_clipped(self):
        from pac.authority import evaluation as pac

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
