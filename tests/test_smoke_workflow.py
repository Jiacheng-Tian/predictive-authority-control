from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pac.authority.model import load_alpha_model_checkpoint


ROOT = Path(__file__).resolve().parents[1]


class SmokeWorkflowTest(unittest.TestCase):
    def test_short_training_evaluation_and_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "run"
            figures = Path(temp_dir) / "figures"
            completed = subprocess.run([
                sys.executable,
                "-B",
                "-m",
                "pac.authority.pipeline",
                "--config",
                str(ROOT / "config" / "pac.yaml"),
                "--out-dir",
                str(output),
                "--figure-dir",
                str(figures),
                "--steps",
                "20",
                "--train-scenarios",
                "1",
                "--train-seeds",
                "0",
                "--eval-scenarios",
                "1",
                "--eval-seeds",
                "90000",
                "--epochs",
                "1",
                "--train-seed",
                "99",
                "--oracle-horizon-steps",
                "2",
                "--no-save-timeseries",
                "--progress-every",
                "0",
            ], cwd=ROOT, text=True, capture_output=True, check=False)

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            raw = pd.read_csv(output / "raw_metrics.csv")
            self.assertEqual(len(raw), 1)
            self.assertFalse((output / "timeseries" / "timeseries_3d.csv").exists())
            model, metadata = load_alpha_model_checkpoint(output / "predictive_alpha_model.pt")
            self.assertEqual(metadata["policy_architecture"], "transformer")
            self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 14113)


if __name__ == "__main__":
    unittest.main()
