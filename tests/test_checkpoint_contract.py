from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class CheckpointContractTest(unittest.TestCase):
    def test_v3_checkpoint_round_trip_and_contract_validation(self):
        from pac.authority.model import (
            TemporalAlphaTransformer,
            load_v3_checkpoint,
            save_v3_checkpoint,
        )
        from pac.experiment_config import load_v3_config

        config = load_v3_config(ROOT / "config" / "pac_v3.yaml").authority_model
        model = TemporalAlphaTransformer(24, 16, 32, 4, 1, 0.1)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"
            payload = save_v3_checkpoint(
                path,
                model,
                config=config,
                model_seed=31000,
                dataset_hash="dataset-hash",
                git_commit="detached-or-dirty-commit",
                dependency_versions={"torch": "test"},
                training_metrics={
                    "best_val_mse": 0.25,
                    "best_epoch": 1,
                    "epochs_ran": 1,
                    "param_count": 14113,
                },
            )
            self.assertEqual(payload["protocol_version"], "formal_true_mpc_v3")
            loaded, metadata = load_v3_checkpoint(
                path,
                config=config,
                model_seed=31000,
                dataset_hash="dataset-hash",
            )
            self.assertEqual(metadata["param_count"], 14113)
            self.assertEqual(sum(parameter.numel() for parameter in loaded.parameters()), 14113)
            self.assertEqual(set(model.state_dict()), set(loaded.state_dict()))
            with self.assertRaises((FileExistsError, ValueError)):
                save_v3_checkpoint(
                    path,
                    model,
                    config=config,
                    model_seed=31000,
                    dataset_hash="dataset-hash",
                )

    def test_v3_checkpoint_requires_each_artifact_metadata_key(self):
        import torch
        from pac.authority.model import (
            TemporalAlphaTransformer,
            load_v3_checkpoint,
            save_v3_checkpoint,
        )
        from pac.experiment_config import load_v3_config

        config = load_v3_config(ROOT / "config" / "pac_v3.yaml").authority_model
        model = TemporalAlphaTransformer(24, 16, 32, 4, 1, 0.1)
        required = ("git_commit", "dependency_versions", "training_metrics")
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.pt"
            save_v3_checkpoint(
                source,
                model,
                config=config,
                model_seed=31000,
                dataset_hash="dataset-hash",
                git_commit="dirty-worktree",
                dependency_versions={"torch": "test"},
                training_metrics={
                    "best_val_mse": 0.25,
                    "best_epoch": 1,
                    "epochs_ran": 1,
                    "param_count": 14113,
                },
            )
            payload = torch.load(source, map_location="cpu", weights_only=True)
            for key in required:
                missing = dict(payload)
                missing.pop(key)
                path = Path(temp_dir) / f"missing-{key}.pt"
                torch.save(missing, path)
                with self.subTest(case=f"missing-{key}"):
                    with self.assertRaises(ValueError):
                        load_v3_checkpoint(path)
            wrong_types = {
                "git_commit": 123,
                "dependency_versions": [],
                "training_metrics": [],
            }
            for key, value in wrong_types.items():
                wrong = dict(payload)
                wrong[key] = value
                path = Path(temp_dir) / f"wrong-{key}.pt"
                torch.save(wrong, path)
                with self.subTest(case=f"wrong-{key}"):
                    with self.assertRaises(ValueError):
                        load_v3_checkpoint(path)

    def test_v3_checkpoint_validates_training_metric_types_and_parameter_count(self):
        import torch
        from pac.authority.model import (
            TemporalAlphaTransformer,
            load_v3_checkpoint,
            save_v3_checkpoint,
        )
        from pac.experiment_config import load_v3_config

        config = load_v3_config(ROOT / "config" / "pac_v3.yaml").authority_model
        model = TemporalAlphaTransformer(24, 16, 32, 4, 1, 0.1)
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.pt"
            save_v3_checkpoint(
                source,
                model,
                config=config,
                model_seed=31000,
                dataset_hash="dataset-hash",
                git_commit="HEAD-dirty",
                dependency_versions={"torch": "test"},
                training_metrics={
                    "best_val_mse": 0.25,
                    "best_epoch": 1,
                    "epochs_ran": 1,
                    "param_count": 14113,
                },
            )
            payload = torch.load(source, map_location="cpu", weights_only=True)
            cases = {
                "best_val_mse": "not-a-number",
                "best_epoch": 1.5,
                "epochs_ran": float("nan"),
                "param_count": 14112,
            }
            for key, value in cases.items():
                wrong = dict(payload)
                metrics = dict(payload["training_metrics"])
                metrics[key] = value
                wrong["training_metrics"] = metrics
                path = Path(temp_dir) / f"metric-{key}.pt"
                torch.save(wrong, path)
                with self.subTest(case=key):
                    with self.assertRaises(ValueError):
                        load_v3_checkpoint(path)

            wrong_top = dict(payload)
            wrong_top["param_count"] = 14112
            path = Path(temp_dir) / "top-param-count.pt"
            torch.save(wrong_top, path)
            with self.assertRaises(ValueError):
                load_v3_checkpoint(path)

    def test_v2_checkpoint_is_rejected_by_v3_loader(self):
        import torch
        from pac.authority.model import load_v3_checkpoint

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "old.pt"
            torch.save({"state_dict": {}}, path)
            with self.assertRaises(ValueError):
                load_v3_checkpoint(path)

    def test_checkpoint_write_is_atomic_and_cleans_temp_on_failure(self):
        from pac.authority.model import TemporalAlphaTransformer, save_v3_checkpoint
        from pac.experiment_config import load_v3_config

        config = load_v3_config(ROOT / "config" / "pac_v3.yaml").authority_model
        model = TemporalAlphaTransformer(24, 16, 32, 4, 1, 0.1)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "checkpoint.pt"
            with patch("pac.authority.model.os.replace", side_effect=RuntimeError("rename failed")):
                with self.assertRaisesRegex(RuntimeError, "rename failed"):
                    save_v3_checkpoint(
                        target,
                        model,
                        config=config,
                        model_seed=31000,
                        dataset_hash="dataset-hash",
                        git_commit="HEAD",
                        git_dirty=False,
                        git_diff_sha256="clean",
                        dependency_versions={"torch": "test"},
                        training_metrics={
                            "best_val_mse": 0.25,
                            "best_epoch": 1,
                            "epochs_ran": 1,
                            "param_count": 14113,
                        },
                    )
            self.assertFalse(target.exists())
            self.assertEqual(list(root.glob(".checkpoint.pt.tmp-*")), [])

    def test_corrupt_checkpoint_failures_include_path_and_do_not_swallow_keyboard_interrupt(self):
        import torch
        from pac.authority.model import load_v3_checkpoint

        with tempfile.TemporaryDirectory() as temp_dir:
            corrupt = Path(temp_dir) / "corrupt.pt"
            corrupt.write_bytes(b"not-a-torch-checkpoint")
            with self.assertRaises(ValueError) as corrupt_error:
                load_v3_checkpoint(corrupt)
            self.assertIn(str(corrupt), str(corrupt_error.exception))
            self.assertIn("contract", str(corrupt_error.exception))
            wrong_type = Path(temp_dir) / "wrong-type.pt"
            torch.save([], wrong_type)
            with self.assertRaises(ValueError) as wrong_type_error:
                load_v3_checkpoint(wrong_type)
            self.assertIn(str(wrong_type), str(wrong_type_error.exception))
            self.assertIn("contract", str(wrong_type_error.exception))
            overflow = Path(temp_dir) / "overflow.pt"
            torch.save({"protocol_version": "formal_true_mpc_v3", "input_dim": 10**100}, overflow)
            with self.assertRaises(ValueError) as overflow_error:
                load_v3_checkpoint(overflow)
            self.assertIn(str(overflow), str(overflow_error.exception))
            self.assertIn("contract", str(overflow_error.exception))
            with patch("pac.authority.model.torch.load", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    load_v3_checkpoint(corrupt)

    def test_checkpoint_aliases_and_expected_config_are_strict(self):
        import torch
        from dataclasses import replace
        from pac.authority.model import (
            TemporalAlphaTransformer,
            load_v3_checkpoint,
            save_v3_checkpoint,
        )
        from pac.experiment_config import load_v3_config

        config = load_v3_config(ROOT / "config" / "pac_v3.yaml").authority_model
        model = TemporalAlphaTransformer(24, 16, 32, 4, 1, 0.1)
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.pt"
            save_v3_checkpoint(
                source,
                model,
                config=config,
                model_seed=31000,
                dataset_hash="dataset-hash",
                git_commit="HEAD",
                git_dirty=False,
                git_diff_sha256="clean",
                dependency_versions={"torch": "test"},
                training_metrics={
                    "best_val_mse": 0.25,
                    "best_epoch": 1,
                    "epochs_ran": 1,
                    "param_count": 14113,
                },
            )
            payload = torch.load(source, map_location="cpu", weights_only=True)
            aliases = {
                "architecture": "mlp",
                "policy_architecture": "mlp",
                "input_dim": 23,
                "feature_mode": "other",
                "history_len": 15,
                "embed_dim": 16,
                "transformer_embed_dim": 16,
                "heads": 2,
                "transformer_heads": 2,
                "layers": 2,
                "transformer_layers": 2,
                "dropout": 0.2,
                "model_dropout": 0.2,
            }
            for key, value in aliases.items():
                wrong = dict(payload)
                wrong[key] = value
                path = Path(temp_dir) / f"alias-{key}.pt"
                torch.save(wrong, path)
                with self.subTest(alias=key):
                    with self.assertRaises(ValueError):
                        load_v3_checkpoint(path, config=config)
            wrong_config = replace(config, alpha_gain=0.5)
            with self.assertRaises(ValueError):
                load_v3_checkpoint(source, config=wrong_config)


if __name__ == "__main__":
    unittest.main()
