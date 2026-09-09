from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch


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
    def test_formal_and_short_pass_configured_epoch_limits(self):
        import scripts.train_pac as train_pac
        from pac.authority.model import TemporalAlphaTransformer
        from pac.experiment_config import load_v3_config

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            _write_dataset(dataset)
            config_source = json.loads(json.dumps({}))
            import yaml
            config_source = yaml.safe_load((ROOT / "config" / "pac_v3.yaml").read_text(encoding="utf-8"))
            config_source["training"]["max_epochs"] = 7
            config_source["training"]["patience"] = 3
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config_source), encoding="utf-8")
            model = TemporalAlphaTransformer(24, 16, 32, 4, 1, 0.1)
            history = pd.DataFrame({"epoch": [1], "train_mse": [0.1], "val_mse": [0.2]})
            metrics = {
                "best_val_mse": 0.2,
                "best_epoch": 1,
                "epochs_ran": 1,
                "param_count": 14113,
            }
            clean_git = {"git_commit": "HEAD", "git_dirty": False, "git_diff_sha256": "clean"}
            with patch.object(train_pac, "train_alpha_model_v3", return_value=(model, history, metrics)) as trainer:
                with patch.object(train_pac, "_default_git_provenance", return_value=clean_git):
                    train_pac.main([
                        "--config", str(config_path), "--dataset-dir", str(dataset),
                        "--out-dir", str(root / "formal"), "--profile", "formal",
                    ])
                    self.assertEqual(trainer.call_args.kwargs["max_epochs"], 7)
                    self.assertEqual(trainer.call_args.kwargs["patience"], 3)
            trainer.reset_mock()
            with patch.object(train_pac, "train_alpha_model_v3", return_value=(model, history, metrics)) as trainer:
                with patch.object(train_pac, "_default_git_provenance", return_value=clean_git):
                    train_pac.main([
                        "--config", str(config_path), "--dataset-dir", str(dataset),
                        "--out-dir", str(root / "short"), "--profile", "short",
                    ])
                    self.assertEqual(trainer.call_args.kwargs["max_epochs"], 3)
                    self.assertEqual(trainer.call_args.kwargs["patience"], 3)

    def test_rejects_equal_or_nested_dataset_and_output_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            _write_dataset(dataset)
            for output in (dataset, dataset / "nested-output", root / "outer-output"):
                if output == root / "outer-output":
                    nested_dataset = output / "nested-dataset"
                    _write_dataset(nested_dataset)
                    dataset_arg = nested_dataset
                else:
                    dataset_arg = dataset
                completed = subprocess.run(
                    [sys.executable, "scripts/train_pac.py", "--config", "pac_v3", "--dataset-dir", str(dataset_arg), "--out-dir", str(output), "--profile", "dry"],
                    cwd=ROOT, text=True, capture_output=True, check=False,
                )
                self.assertNotEqual(completed.returncode, 0)

    def test_atomic_output_cleanup_on_checkpoint_failure(self):
        import scripts.train_pac as train_pac
        from pac.authority.model import TemporalAlphaTransformer

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            output = root / "output"
            _write_dataset(dataset)
            model = TemporalAlphaTransformer(24, 16, 32, 4, 1, 0.1)
            history = pd.DataFrame({"epoch": [1], "train_mse": [0.1], "val_mse": [0.2]})
            metrics = {
                "best_val_mse": 0.2,
                "best_epoch": 1,
                "epochs_ran": 1,
                "param_count": 14113,
            }
            with patch.object(train_pac, "train_alpha_model_v3", return_value=(model, history, metrics)):
                with patch.object(train_pac, "save_v3_checkpoint", side_effect=RuntimeError("injected write failure")):
                    with patch.object(train_pac, "_default_git_provenance", return_value={"git_commit": "HEAD", "git_dirty": False, "git_diff_sha256": "clean"}):
                        with self.assertRaisesRegex(RuntimeError, "injected write failure"):
                            train_pac.main([
                                "--config", "pac_v3", "--dataset-dir", str(dataset),
                                "--out-dir", str(output), "--profile", "short",
                            ])
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".output.tmp-*")), [])

    def test_formal_rejects_dirty_worktree_without_debug_override(self):
        import scripts.train_pac as train_pac

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            output = root / "formal-output"
            _write_dataset(dataset)
            dirty = {"git_commit": "HEAD", "git_dirty": True, "git_diff_sha256": "dirty"}
            with patch.object(train_pac, "_default_git_provenance", return_value=dirty):
                with self.assertRaisesRegex(ValueError, "clean git worktree"):
                    train_pac.main([
                        "--config", "pac_v3", "--dataset-dir", str(dataset),
                        "--out-dir", str(output), "--profile", "formal",
                    ])
            self.assertFalse(output.exists())
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
