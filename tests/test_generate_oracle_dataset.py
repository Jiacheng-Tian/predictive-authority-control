from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GenerateOracleDatasetTest(unittest.TestCase):
    def test_dry_prints_plan_without_writing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "dry"
            completed = subprocess.run(
                [sys.executable, "scripts/generate_oracle_dataset.py", "--profile", "dry", "--out-dir", str(output)],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertFalse(output.exists())
            self.assertEqual(json.loads(completed.stdout)["profile"], "dry")

    def test_short_writes_two_episodes_and_at_most_forty_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "short"
            completed = subprocess.run(
                [sys.executable, "scripts/generate_oracle_dataset.py", "--profile", "short", "--out-dir", str(output)],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            import pandas as pd
            metadata = pd.read_csv(output / "metadata.csv")
            self.assertLessEqual(len(metadata), 40)
            self.assertEqual(len(set(metadata["episode_uid"])), 2)
            self.assertEqual(metadata["split"].tolist()[:1], ["train"])
            self.assertTrue((output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
