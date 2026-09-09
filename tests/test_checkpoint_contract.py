from __future__ import annotations

from pathlib import Path
import tempfile
import unittest


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
                training_metrics={"best_epoch": 1, "val_mse": 0.25},
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

    def test_v2_checkpoint_is_rejected_by_v3_loader(self):
        import torch
        from pac.authority.model import load_v3_checkpoint

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "old.pt"
            torch.save({"state_dict": {}}, path)
            with self.assertRaises(ValueError):
                load_v3_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
