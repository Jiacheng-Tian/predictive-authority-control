"""Tests for the v4 experiment configuration."""

from __future__ import annotations

from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "pac_v4.yaml"

_V3_SEEDS = set(range(11000, 11008)) | {12000, 12001} | set(range(31000, 31005)) | \
    set(range(41000, 41020)) | set(range(51000, 51005)) | set(range(61000, 61010)) | \
    set(range(71000, 71010))


class V4ConfigTests(unittest.TestCase):
    def test_default_config_loads(self):
        from pac.v4.config import load_v4_config, seed_partitions

        config = load_v4_config(CONFIG_PATH)
        self.assertEqual(config.protocol.version, "predictive_authority_v4")
        self.assertEqual(config.world_model.members, 5)
        self.assertEqual(config.world_model.horizons, (10, 20))
        self.assertEqual(config.world_model.history_len, 16)
        self.assertIn("ou_current", config.disturbances.families)
        partitions = seed_partitions(config)
        self.assertEqual(len(partitions), 7)
        self.assertIn("sspo_search", partitions)

    def test_environment_seeds_disjoint_from_v3(self):
        from pac.v4.config import load_v4_config

        config = load_v4_config(CONFIG_PATH)
        for role in ("wm_train", "wm_val", "wm_test", "eval_seen", "eval_unseen"):
            for seed in getattr(config.seeds, role):
                self.assertNotIn(seed, _V3_SEEDS, f"{role} seed {seed} collides with v3")

    def test_invalid_protocol_rejected(self):
        from pac.v4.config import load_v4_config

        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        data["protocol"]["version"] = "formal_true_mpc_v3"
        with self.assertRaises(ValueError):
            load_v4_config_from_mapping(data)

    def test_seed_collision_with_v3_rejected(self):
        from pac.v4.config import load_v4_config

        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        data["seeds"]["wm_train"] = [41000, 21001, 21002, 21003, 21004, 21005, 21006, 21007]
        with self.assertRaises(ValueError):
            load_v4_config_from_mapping(data)

    def test_internal_seed_collision_rejected(self):
        from pac.v4.config import load_v4_config

        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        data["seeds"]["wm_val"] = list(data["seeds"]["wm_train"])
        with self.assertRaises(ValueError):
            load_v4_config_from_mapping(data)

    def test_disturbance_test_ranges_do_not_overlap_train(self):
        from pac.v4.config import load_v4_config

        config = load_v4_config(CONFIG_PATH)
        for family in config.disturbances.families:
            family_config = config.disturbances.families_config[family]
            for parameter in family_config.ranges["train"]:
                train_range = family_config.ranges["train"][parameter]
                test_range = family_config.ranges["test"][parameter]
                disjoint = (
                    train_range.high <= test_range.low
                    or test_range.high <= train_range.low
                )
                self.assertTrue(
                    disjoint,
                    f"{family}.{parameter} test range overlaps train",
                )


def load_v4_config_from_mapping(data):
    import tempfile

    from pac.v4.config import load_v4_config

    with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False, encoding="utf-8") as stream:
        yaml.safe_dump(data, stream)
        path = stream.name
    try:
        return load_v4_config(path)
    finally:
        Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
