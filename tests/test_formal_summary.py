from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FormalSummaryTest(unittest.TestCase):
    def test_summary_recomputes_archived_headline_values(self):
        from pac.evaluation.summary import summarize_formal_results

        summary = summarize_formal_results(ROOT / "results" / "formal_seeded_v2")

        self.assertAlmostEqual(summary["pac"]["mean_rmse_3d"], 0.0902230450212123)
        self.assertAlmostEqual(summary["pac"]["pooled_sample_sd"], 0.00979868270936307)
        self.assertAlmostEqual(summary["pac"]["training_seed_mean_sd"], 0.00128272608284843)
        self.assertAlmostEqual(summary["smc"]["mean_rmse_3d"], 0.14374800885215)
        self.assertAlmostEqual(summary["predictive"]["mean_rmse_3d"], 0.145712776770692)

    def test_summary_cli_is_read_only_json(self):
        completed = subprocess.run(
            [sys.executable, "-B", "scripts/summarize_formal_results.py"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = json.loads(completed.stdout)
        self.assertEqual(summary["pac"]["rollouts"], 150)
        self.assertEqual(summary["pac"]["training_seeds"], 5)


if __name__ == "__main__":
    unittest.main()
