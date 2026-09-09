from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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

    def test_save_hash_uses_float_roundtrip_metadata_and_loads_short_style_rows(self):
        from pac.authority.dataset import load_oracle_dataset, save_oracle_dataset

        dataset = _dataset()
        dataset.metadata.loc[0, "sample_time"] = 0.12345678901234567
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "oracle"
            save_oracle_dataset(dataset, output, {"config_hash": "abc"})
            loaded = load_oracle_dataset(output)
            self.assertEqual(loaded.features.dtype, np.dtype("float32"))
            self.assertEqual(loaded.labels.dtype, np.dtype("float32"))
            self.assertAlmostEqual(
                loaded.metadata.loc[0, "sample_time"],
                dataset.metadata.loc[0, "sample_time"],
                places=15,
            )

    def test_save_requires_both_splits_and_rejects_cross_episode_fingerprint(self):
        from pac.authority.dataset import save_oracle_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            only_train = _dataset()
            only_train.metadata["split"] = "train"
            with self.assertRaises(ValueError):
                save_oracle_dataset(only_train, Path(temp_dir) / "only-train", {})

            duplicate = _dataset()
            duplicate.metadata.loc[1, "episode_fingerprint"] = duplicate.metadata.loc[0, "episode_fingerprint"]
            with self.assertRaises(ValueError):
                save_oracle_dataset(duplicate, Path(temp_dir) / "duplicate", {})

            cross_split = _dataset()
            cross_split.metadata.loc[1, "episode_uid"] = cross_split.metadata.loc[0, "episode_uid"]
            with self.assertRaises(ValueError):
                save_oracle_dataset(cross_split, Path(temp_dir) / "cross-split", {})

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

    def test_episode_fingerprint_normalizes_endian_and_memory_layout(self):
        from pac.authority.dataset import episode_fingerprint

        little = np.arange(12, dtype="<f8").reshape(3, 4)
        big = np.asfortranarray(little.astype(">f8"))
        first = SimpleNamespace(values=little, nested=np.array([1.0, 2.0], dtype="<f8"))
        second = SimpleNamespace(values=big, nested=np.array([1.0, 2.0], dtype=">f8"))
        self.assertEqual(episode_fingerprint(first), episode_fingerprint(second))

    def test_content_hash_is_portable_and_manifest_distinguishes_actual_seeds(self):
        from pac.authority.dataset import canonical_dataset_hash, save_oracle_dataset

        first = _dataset()
        reordered = first.metadata.loc[:, list(reversed(first.metadata.columns))].copy()
        second = type(first)(
            np.ascontiguousarray(first.features.astype(">f4")),
            np.asfortranarray(first.labels.astype("<f4")),
            reordered,
        )
        semantic = {
            "protocol_version": "formal_true_mpc_v3",
            "config_hash": "config-semantic-sha",
            "actual_seed_partitions": {"train": [1], "val": [2]},
            "oracle_settings": {"horizon": 20, "alpha_grid": [0.0, 1.0]},
            "profile": "short",
            "config_source": "first-root/config/pac_v3.yaml",
            "git_commit": "first",
            "dependency_versions": {"numpy": "one"},
        }
        changed_paths = dict(semantic)
        changed_paths.update({
            "config_source": "second-root/config/pac_v3.yaml",
            "git_commit": "second",
            "dependency_versions": {"numpy": "two"},
        })
        self.assertEqual(canonical_dataset_hash(first, semantic), canonical_dataset_hash(second, changed_paths))
        swapped = type(first)(first.features, first.labels, reordered.iloc[::-1].reset_index(drop=True))
        self.assertNotEqual(canonical_dataset_hash(first, semantic), canonical_dataset_hash(swapped, semantic))
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = save_oracle_dataset(
                first,
                Path(temp_dir) / "oracle",
                {**semantic, "configured_seed_partitions": {"train": [1, 2], "val": [3]}},
            )
            self.assertEqual(manifest["seed_partitions"], {"train": [1], "val": [2]})
            self.assertEqual(manifest["configured_seed_partitions"], {"train": [1, 2], "val": [3]})

    def test_numeric_episode_uid_roundtrips_as_string(self):
        from pac.authority.dataset import load_oracle_dataset, save_oracle_dataset

        dataset = _dataset()
        dataset.metadata["episode_uid"] = ["001", "002"]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "oracle"
            save_oracle_dataset(dataset, output, {})
            loaded = load_oracle_dataset(output)
            self.assertEqual(loaded.metadata["episode_uid"].tolist(), ["001", "002"])
            self.assertEqual(str(loaded.metadata["episode_uid"].dtype), "string")


if __name__ == "__main__":
    unittest.main()
