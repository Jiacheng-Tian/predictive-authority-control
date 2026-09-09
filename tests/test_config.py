from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class PACConfigTest(unittest.TestCase):
    def test_default_config_matches_formal_v2_protocol(self):
        from pac.config import load_config

        config = load_config(ROOT / "config" / "pac.yaml")

        self.assertEqual(config.environment.steps, 2100)
        self.assertEqual(config.environment.scenarios, (1, 2, 3))
        self.assertEqual(config.controllers.primary, "real10kg_smc_steady")
        self.assertEqual(config.controllers.authority, "real10kg_predictive_event")
        self.assertEqual(config.authority.history_len, 16)
        self.assertEqual(config.authority.embed_dim, 32)
        self.assertEqual(config.authority.heads, 4)
        self.assertEqual(config.training.data_seeds, (0, 1, 2))
        self.assertEqual(config.training.model_seeds, (20, 21, 22, 23, 24))
        self.assertEqual(config.evaluation.episode_seeds, tuple(range(10)))
        self.assertEqual(config.sspo.search_seeds, (20000, 20001, 20002))
        self.assertEqual(config.sspo.eval_seeds, tuple(range(20000, 20010)))
        self.assertEqual(config.sspo.bias_grid, (-0.05, 0.0, 0.1, 0.2, 0.3))
        self.assertEqual(config.sspo.window_regret_weight, 0.02)

    def test_invalid_transformer_dimensions_are_rejected(self):
        from pac.config import load_config

        source = yaml.safe_load((ROOT / "config" / "pac.yaml").read_text(encoding="utf-8"))
        source["authority"]["embed_dim"] = 30
        source["authority"]["heads"] = 4
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.yaml"
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "embed_dim must be divisible by heads"):
                load_config(path)

    def test_duplicate_seeds_are_rejected(self):
        from pac.config import load_config

        source = yaml.safe_load((ROOT / "config" / "pac.yaml").read_text(encoding="utf-8"))
        source["training"]["model_seeds"] = [20, 20]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.yaml"
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model_seeds must contain unique values"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
