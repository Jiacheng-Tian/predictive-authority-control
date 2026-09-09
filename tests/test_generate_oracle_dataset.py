from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class GenerateOracleDatasetTest(unittest.TestCase):
    def test_profile_normalization_maps_blank_and_whitespace_to_short(self):
        from pac.authority.training import normalize_oracle_profile

        self.assertEqual(normalize_oracle_profile(""), "short")
        self.assertEqual(normalize_oracle_profile("  SHORT  "), "short")
        self.assertEqual(normalize_oracle_profile(" formal "), "formal")

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
            from pac.authority.dataset import load_oracle_dataset
            loaded = load_oracle_dataset(output)
            self.assertEqual(loaded.features.shape[1], 24)
            self.assertLessEqual(len(metadata), 40)
            self.assertEqual(len(set(metadata["episode_uid"])), 2)
            self.assertEqual(metadata["split"].tolist()[:1], ["train"])
            self.assertTrue((output / "manifest.json").exists())

    def test_short_can_generate_two_fresh_outputs(self):
        import pandas as pd
        from pac.authority.dataset import load_oracle_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            for index in (1, 2):
                output = Path(temp_dir) / f"short-{index}"
                completed = subprocess.run(
                    [
                        sys.executable,
                        "scripts/generate_oracle_dataset.py",
                        "--profile",
                        "short",
                        "--out-dir",
                        str(output),
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                self.assertEqual(len(pd.read_csv(output / "metadata.csv")), 40)
                self.assertEqual(load_oracle_dataset(output).sample_count, 40)

    def test_formal_contract_is_required_by_dry_and_short(self):
        import yaml

        with tempfile.TemporaryDirectory() as temp_dir:
            source = yaml.safe_load((ROOT / "config" / "pac_v3.yaml").read_text(encoding="utf-8"))
            source["oracle"]["horizon"] = 19
            config_path = Path(temp_dir) / "invalid.yaml"
            config_path.write_text(yaml.safe_dump(source), encoding="utf-8")
            for profile in ("dry", "short", "formal"):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "scripts/generate_oracle_dataset.py",
                        "--config",
                        str(config_path),
                        "--profile",
                        profile,
                        "--out-dir",
                        str(Path(temp_dir) / profile),
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("horizon", completed.stderr + completed.stdout)

    def test_formal_episode_progress_uses_stderr_and_stdout_stays_quiet(self):
        from pac.authority.training import collect_teacher_dataset_v3
        from pac.experiment_config import load_v3_config

        config = load_v3_config(ROOT / "config" / "pac_v3.yaml")

        class FakeController:
            def __init__(self, authority=False):
                self.last_plan = np.zeros((config.mpc.horizon, 6)) if authority else None
                self.last_telemetry = {"accepted": False, "fallback_mode": "none", "plan_generation": 0}
                self.authority = authority

            def reset(self):
                self.last_telemetry["plan_generation"] = 0

            def set_trajectory3d(self, enabled=True):
                pass

            def compute(self, target, eta, nu, *, t=None, current_prediction=None):
                if self.authority:
                    self.last_telemetry.update(accepted=True, plan_generation=self.last_telemetry["plan_generation"] + 1)
                return np.zeros(6)

        stdout = StringIO()
        stderr = StringIO()
        with patch("pac.authority.training._collection_plan", return_value=[("train", 11000, 1)]):
            with patch(
                "pac.authority.training.build_v3_controller_pair",
                return_value=(FakeController(), FakeController(authority=True)),
            ):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    collect_teacher_dataset_v3(config, "formal")
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue().count("oracle_episode_complete"), 3)


if __name__ == "__main__":
    unittest.main()
