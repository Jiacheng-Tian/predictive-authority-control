from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _write_dataset(path: Path) -> None:
    from pac.authority.dataset import OracleDataset, save_oracle_dataset

    rows = []
    features = []
    labels = []
    for split, episode, seed in (("train", "train-a", 1), ("val", "val-a", 2)):
        for step in range(4):
            features.append(np.full(24, step + 1, dtype=np.float32))
            labels.append(float(step % 2))
            rows.append({
                "split": split,
                "scenario": 1,
                "environment_seed": seed,
                "episode_uid": episode,
                "step": step,
                "sample_time": step * 0.01,
                "teacher_alpha": labels[-1],
                "oracle_best": 0.0,
                "oracle_worst": 1.0,
                "episode_fingerprint": f"fingerprint-{episode}",
            })
    save_oracle_dataset(
        OracleDataset(np.asarray(features), np.asarray(labels), pd.DataFrame(rows)), path,
        provenance={"protocol_version": "formal_true_mpc_v3", "profile": "short"},
    )


class TrainPacScriptTest(unittest.TestCase):
    def test_dry_lists_all_model_seeds_without_writing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            output = root / "dry-output"
            _write_dataset(dataset)
            completed = subprocess.run(
                [sys.executable, "scripts/train_pac.py", "--config", "pac_v3", "--dataset-dir", str(dataset), "--out-dir", str(output), "--profile", "dry"],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(result["model_seeds"], [31000, 31001, 31002, 31003, 31004])
            self.assertTrue(result["dataset_hash"])
            self.assertFalse(output.exists())

    def test_short_writes_one_checkpoint_and_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            output = root / "short-output"
            _write_dataset(dataset)
            completed = subprocess.run(
                [sys.executable, "scripts/train_pac.py", "--config", "pac_v3", "--dataset-dir", str(dataset), "--out-dir", str(output), "--profile", "short"],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertTrue((output / "pac_train_seed_31000" / "checkpoint.pt").exists())
            self.assertTrue((output / "pac_train_seed_31000" / "training_history.csv").exists())
            self.assertTrue((output / "pac_train_seed_31000" / "training_summary.json").exists())
            self.assertTrue((output / "manifest.json").exists())
            self.assertEqual(len(list(output.glob("pac_train_seed_*"))), 1)

    def test_dry_rejects_nonformal_model_seeds_without_writing(self):
        import yaml

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            output = root / "dry-output"
            config_path = root / "invalid-seeds.yaml"
            _write_dataset(dataset)
            config = yaml.safe_load((ROOT / "config" / "pac_v3.yaml").read_text(encoding="utf-8"))
            config["training"]["model_seeds"] = [999]
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "scripts/train_pac.py", "--config", str(config_path), "--dataset-dir", str(dataset), "--out-dir", str(output), "--profile", "dry"],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("31000", completed.stderr + completed.stdout)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
