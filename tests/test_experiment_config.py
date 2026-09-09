from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
V3_PATH = ROOT / "config" / "pac_v3.yaml"


class V3ExperimentConfigTest(unittest.TestCase):
    def test_default_values_are_loaded_into_frozen_dataclasses(self):
        from pac.experiment_config import V3ExperimentConfig, load_v3_config

        config = load_v3_config(V3_PATH)

        self.assertIsInstance(config, V3ExperimentConfig)
        self.assertEqual(config.protocol.version, "formal_true_mpc_v3")
        self.assertEqual(config.environment.dt, 0.01)
        self.assertEqual(config.environment.steps, 2100)
        self.assertEqual(config.environment.scenarios, (1, 2, 3))
        self.assertEqual(config.controller.primary, "real10kg_smc_steady")
        self.assertEqual(config.controller.authority, "real10kg_mpc_ltv_v3")
        self.assertEqual(config.actuator.command_min, -1.0)
        self.assertEqual(config.actuator.command_max, 1.0)
        self.assertEqual(config.actuator.max_delta_per_step, 0.10)
        self.assertEqual(config.actuator.max_force_n, 35.0)
        self.assertEqual(config.mpc.horizon, 20)
        self.assertEqual(config.mpc.q_diag, (80.0, 80.0, 120.0, 4.0, 2.0, 12.0, 8.0, 8.0, 12.0, 0.5, 0.2, 3.0))
        self.assertEqual(config.mpc.r_diag, (0.05,) * 6)
        self.assertEqual(config.mpc.s_diag, (0.5,) * 6)
        self.assertEqual(config.training.oracle_train_seeds, tuple(range(11000, 11008)))
        self.assertEqual(config.training.oracle_val_seeds, (12000, 12001))
        self.assertEqual(config.training.model_seeds, tuple(range(31000, 31005)))
        self.assertEqual(config.training.max_epochs, 180)
        self.assertEqual(config.training.patience, 20)
        self.assertEqual(config.evaluation.episode_seeds, tuple(range(41000, 41020)))
        self.assertEqual(config.sspo.search_seeds, tuple(range(51000, 51005)))
        self.assertEqual(config.sspo.eval_seeds, tuple(range(61000, 61010)))
        self.assertEqual(config.robustness.eval_seeds, tuple(range(71000, 71010)))
        self.assertEqual(config.outputs.run_root, "runs/formal_true_mpc_v3")
        self.assertEqual(config.outputs.evidence_root, "results/formal_true_mpc_v3")

        with self.assertRaises((AttributeError, TypeError)):
            config.mpc.horizon = 21

    def test_all_seed_partitions_are_pairwise_disjoint(self):
        from pac.experiment_config import load_v3_config, seed_partitions
        from pac.evaluation.seeds import validate_disjoint_seed_partitions

        config = load_v3_config(V3_PATH)
        partitions = seed_partitions(config)

        self.assertIsNone(validate_disjoint_seed_partitions(partitions))
        values = list(partitions.values())
        self.assertEqual(sum(map(len, values)), len({seed for values_ in values for seed in values_}))

    def test_search_eval_overlap_is_rejected_with_roles_and_seed(self):
        from pac.experiment_config import load_v3_config

        source = yaml.safe_load(V3_PATH.read_text(encoding="utf-8"))
        source["sspo"]["eval_seeds"] = [51000]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.yaml"
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"sspo_search.*sspo_eval.*51000|sspo_eval.*sspo_search.*51000"):
                load_v3_config(path)

    def test_duplicate_seed_is_rejected_with_role_and_seed(self):
        from pac.experiment_config import load_v3_config

        source = yaml.safe_load(V3_PATH.read_text(encoding="utf-8"))
        source["training"]["model_seeds"] = [31000, 31000]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.yaml"
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"model.*31000"):
                load_v3_config(path)

    def test_mpc_dimensions_are_rejected(self):
        from pac.experiment_config import load_v3_config

        source = yaml.safe_load(V3_PATH.read_text(encoding="utf-8"))
        source["mpc"]["q_diag"] = [1.0] * 11
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.yaml"
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"q_diag.*12"):
                load_v3_config(path)


if __name__ == "__main__":
    unittest.main()
