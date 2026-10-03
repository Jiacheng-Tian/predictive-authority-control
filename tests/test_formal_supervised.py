from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _write_dataset(path: Path) -> None:
    from pac.authority.dataset import OracleDataset, save_oracle_dataset

    rows = []
    for split, episode, seed in (("train", "train-a", 1), ("val", "val-a", 2)):
        for step in range(2):
            rows.append({
                "split": split,
                "scenario": 1,
                "environment_seed": seed,
                "episode_uid": episode,
                "step": step,
                "sample_time": (step + 1) * 0.01,
                "teacher_alpha": 0.0,
                "oracle_best": 0.0,
                "oracle_worst": 1.0,
                "episode_fingerprint": f"fingerprint-{episode}",
            })
    save_oracle_dataset(
        OracleDataset(
            np.zeros((4, 24), dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            pd.DataFrame(rows),
        ),
        path,
        {"protocol_version": "formal_supervised", "profile": "short"},
    )


def _write_checkpoint(path: Path, dataset_hash: str) -> None:
    from pac.authority.model import TemporalAlphaTransformer, save_supervised_checkpoint
    from pac.experiment_config import load_supervised_config

    config = load_supervised_config(ROOT / "config" / "pac_supervised.yaml")
    save_supervised_checkpoint(
        path / "pac_train_seed_31000" / "checkpoint.pt",
        TemporalAlphaTransformer(24, 16, 32, 4, 1, 0.1),
        config.authority_model,
        model_seed=31000,
        dataset_hash=dataset_hash,
        git_commit="test",
        git_dirty=False,
        git_diff_sha256="test-diff",
        dependency_versions={"torch": "test"},
        training_metrics={
            "best_val_mse": 0.1,
            "best_epoch": 1,
            "epochs_ran": 1,
            "param_count": 14113,
        },
    )


class FormalSupervisedTest(unittest.TestCase):
    def test_method_registry_and_short_seed_plan(self):
        from pac.evaluation.formal_supervised import METHOD_REGISTRY, build_profile_plan
        from pac.experiment_config import load_supervised_config

        config = load_supervised_config(ROOT / "config" / "pac_supervised.yaml")
        self.assertEqual(
            tuple(METHOD_REGISTRY),
            ("real10kg_smc_steady", "real10kg_mpc_ltv_v3", "predictive_alpha"),
        )
        plan = build_profile_plan(config, "short")
        self.assertEqual(plan.model_seeds, (31000,))
        self.assertEqual(plan.environment_seeds, (41000,))
        self.assertEqual(plan.scenarios, (1,))
        self.assertEqual(plan.steps, 20)

    def test_short_smoke_writes_paired_schema(self):
        from pac.evaluation.formal_supervised import run_formal_supervised

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "dataset"
            checkpoint_dir = root / "checkpoints"
            output_root = root / "runs" / "formal_supervised"
            _write_dataset(dataset_dir)
            dataset_hash = json.loads(
                (dataset_dir / "manifest.json").read_text(encoding="utf-8")
            )["dataset_hash"]
            _write_checkpoint(checkpoint_dir, dataset_hash)

            run_dir = run_formal_supervised(
                config_path=ROOT / "config" / "pac_supervised.yaml",
                dataset_dir=dataset_dir,
                checkpoints_dir=checkpoint_dir,
                output_root=output_root,
                profile="short",
                run_id="short-smoke",
                allow_dirty=True,
            )

            self.assertEqual(run_dir, output_root / "short-smoke")
            required = {
                "manifest.json",
                "raw_metrics.csv",
                "window_metrics.csv",
                "overall_summary.csv",
                "by_scenario.csv",
                "paired_effects.csv",
                "formal_summary.json",
            }
            self.assertTrue(required.issubset({path.name for path in run_dir.iterdir()}))
            raw = pd.read_csv(run_dir / "raw_metrics.csv")
            self.assertEqual(set(raw["method"]), {
                "real10kg_smc_steady",
                "real10kg_mpc_ltv_v3",
                "predictive_alpha",
            })
            self.assertEqual(len(raw), 3)
            self.assertEqual(raw["episode_uid"].nunique(), 1)
            for column in (
                "rmse_3d",
                "heading_rmse_deg",
                "applied_control_cost",
                "action_saturation_step_fraction",
                "solver_fallback_step_fraction",
                "solver_deadline_miss_step_fraction",
                "actuator_rate_limit_episode_mean",
                "final_error",
                "max_error",
                "success_1m",
            ):
                self.assertIn(column, raw.columns)
            self.assertTrue(list((run_dir / "timeseries").glob("*.csv")))
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["profile"], "short")
            self.assertEqual(manifest["model_seeds"], [31000])
            self.assertEqual(manifest["environment_seeds"], [41000])
            self.assertIn("dataset_hash", manifest)
            self.assertIn("checkpoint_hashes", manifest)

    def test_dry_profile_does_not_write(self):
        from pac.evaluation.formal_supervised import run_formal_supervised

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "runs" / "formal_supervised"
            plan = run_formal_supervised(
                config_path=ROOT / "config" / "pac_supervised.yaml",
                dataset_dir=root / "missing-dataset",
                checkpoints_dir=root / "missing-checkpoints",
                output_root=output_root,
                profile="dry",
                run_id="dry",
                allow_dirty=True,
            )
            self.assertEqual(plan["profile"], "dry")
            self.assertFalse(output_root.exists())


if __name__ == "__main__":
    unittest.main()
