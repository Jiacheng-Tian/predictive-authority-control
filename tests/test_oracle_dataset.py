from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def _dataset(seed: int = 1):
    from pac.authority.dataset import OracleDataset

    metadata = pd.DataFrame({
        "split": ["train", "val"],
        "scenario": [1, 1],
        "environment_seed": [seed, seed + 1],
        "episode_uid": [f"scn1_env{seed}", f"scn1_env{seed + 1}"],
        "step": [0, 0],
        "sample_time": [0.01, 0.01],
        "teacher_alpha": [0.0, 1.0],
        "oracle_best": [1.0, 1.0],
        "oracle_worst": [1.0, 2.0],
        "episode_fingerprint": [f"fp-{seed}", f"fp-{seed + 1}"],
    })
    return OracleDataset(np.zeros((2, 24), dtype=np.float32), np.zeros(2, dtype=np.float32), metadata)


class OracleDatasetTest(unittest.TestCase):
    def test_schema_and_save_load_hash_tamper_and_nonempty_refusal(self):
        from pac.authority.dataset import load_oracle_dataset, save_oracle_dataset

        dataset = _dataset()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "oracle"
            manifest = save_oracle_dataset(
                dataset,
                output,
                {"config_hash": "abc", "git_commit": "def", "seed_partitions": {"train": [1]}},
            )
            self.assertEqual(load_oracle_dataset(output).features.shape, (2, 24))
            self.assertIn("dataset_hash", manifest)
            with self.assertRaises(FileExistsError):
                save_oracle_dataset(dataset, output, {})
            values = np.load(output / "features.npy")
            values[0, 0] = 1.0
            np.save(output / "features.npy", values)
            with self.assertRaises(ValueError):
                load_oracle_dataset(output)

    def test_episode_fingerprint_is_stable_and_unique(self):
        from pac.authority.dataset import assert_unique_episode_fingerprints, episode_fingerprint
        from pac.evaluation.episode_spec import build_episode_spec

        first = build_episode_spec(1, 1, 2, 0.01)
        second = build_episode_spec(1, 2, 2, 0.01)
        self.assertEqual(episode_fingerprint(first), episode_fingerprint(first))
        self.assertNotEqual(episode_fingerprint(first), episode_fingerprint(second))
        assert_unique_episode_fingerprints(pd.DataFrame({"episode_fingerprint": ["a", "b"]}))
        with self.assertRaises(ValueError):
            assert_unique_episode_fingerprints(pd.DataFrame({"episode_fingerprint": ["a", "a"]}))


if __name__ == "__main__":
    unittest.main()
