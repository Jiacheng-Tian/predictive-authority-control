from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
V3_PATH = ROOT / "config" / "pac_supervised.yaml"


class SupervisedExperimentConfigTest(unittest.TestCase):
    def test_default_values_are_loaded_into_frozen_dataclasses(self):
        from pac.experiment_config import SupervisedExperimentConfig, load_supervised_config

        config = load_supervised_config(V3_PATH)

        self.assertIsInstance(config, SupervisedExperimentConfig)
        self.assertEqual(config.protocol.version, "formal_supervised")
        self.assertEqual(config.environment.dt, 0.01)
        self.assertEqual(config.environment.steps, 2100)
        self.assertEqual(config.environment.scenarios, (1, 2, 3))
        self.assertEqual(config.environment.mass_scale_xy, 1.0)
        self.assertEqual(config.environment.damping_scale_xy, 1.0)
        self.assertEqual(config.environment.current_amplitude_scale, 2.5)
        self.assertEqual(config.environment.current_frequency_scale, 1.0)
        self.assertEqual(config.environment.vertical_current, 0.75)
        self.assertEqual(config.environment.eval_initial_position_std, 0.03)
        self.assertEqual(config.environment.eval_initial_velocity_std, 0.01)
        self.assertEqual(config.environment.vehicle_profile, "real_10kg_v1")
        self.assertEqual(config.environment.thruster_layout, "real_10kg_x")
        self.assertEqual(config.controller.primary, "real10kg_smc_steady")
        self.assertEqual(config.controller.authority, "real10kg_mpc_ltv_v3")
        self.assertEqual(config.actuator.command_min, -1.0)
        self.assertEqual(config.actuator.command_max, 1.0)
        self.assertEqual(config.actuator.max_delta_per_step, 0.10)
        self.assertEqual(config.actuator.max_force_n, 35.0)
        self.assertEqual(config.mpc.horizon, 20)
        self.assertEqual(config.mpc.q_diag, (80.0, 80.0, 120.0, 4.0, 2.0, 12.0, 8.0, 8.0, 12.0, 0.5, 0.2, 3.0))
        self.assertEqual(config.mpc.terminal_scale, 5.0)
        self.assertEqual(config.mpc.r_diag, (0.05,) * 6)
        self.assertEqual(config.mpc.s_diag, (0.5,) * 6)
        self.assertEqual(config.mpc.eps_abs, 1.0e-4)
        self.assertEqual(config.mpc.eps_rel, 1.0e-4)
        self.assertEqual(config.mpc.max_iter, 1000)
        self.assertEqual(config.mpc.time_limit_s, 0.0075)
        self.assertEqual(config.mpc.accept_inaccurate_residual, 1.0e-3)
        self.assertEqual(config.mpc.max_consecutive_plan_reuse, 3)
        self.assertEqual(config.oracle.mpc_solver_time_limit_s, 0.05)
        self.assertEqual(config.training.oracle_train_seeds, tuple(range(11000, 11008)))
        self.assertEqual(config.training.oracle_val_seeds, (12000, 12001))
        self.assertEqual(config.training.model_seeds, tuple(range(31000, 31005)))
        self.assertEqual(config.training.max_epochs, 180)
        self.assertEqual(config.training.patience, 20)
        self.assertEqual(config.evaluation.episode_seeds, tuple(range(41000, 41020)))
        self.assertEqual(config.sspo.search_seeds, tuple(range(51000, 51005)))
        self.assertEqual(config.sspo.eval_seeds, tuple(range(61000, 61010)))
        self.assertEqual(config.robustness.eval_seeds, tuple(range(71000, 71010)))
        self.assertEqual(config.outputs.run_root, "runs/formal_supervised")
        self.assertEqual(config.outputs.evidence_root, "results/formal_supervised")

        with self.assertRaises((AttributeError, TypeError)):
            config.mpc.horizon = 21

    def test_all_seed_partitions_are_pairwise_disjoint(self):
        from pac.experiment_config import load_supervised_config, seed_partitions
        from pac.evaluation.seeds import validate_disjoint_seed_partitions

        config = load_supervised_config(V3_PATH)
        partitions = seed_partitions(config)

        self.assertIsNone(validate_disjoint_seed_partitions(partitions))
        values = list(partitions.values())
        self.assertEqual(sum(map(len, values)), len({seed for values_ in values for seed in values_}))

    def test_search_eval_overlap_is_rejected_with_roles_and_seed(self):
        from pac.experiment_config import load_supervised_config

        source = yaml.safe_load(V3_PATH.read_text(encoding="utf-8"))
        source["sspo"]["eval_seeds"] = [51000]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.yaml"
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"sspo_search.*sspo_eval.*51000|sspo_eval.*sspo_search.*51000"):
                load_supervised_config(path)

    def test_duplicate_seed_is_rejected_with_role_and_seed(self):
        from pac.experiment_config import load_supervised_config

        source = yaml.safe_load(V3_PATH.read_text(encoding="utf-8"))
        source["training"]["model_seeds"] = [31000, 31000]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.yaml"
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"model.*31000"):
                load_supervised_config(path)

    def test_mpc_dimensions_are_rejected(self):
        from pac.experiment_config import load_supervised_config

        source = yaml.safe_load(V3_PATH.read_text(encoding="utf-8"))
        source["mpc"]["q_diag"] = [1.0] * 11
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.yaml"
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"q_diag.*12"):
                load_supervised_config(path)

    def test_oracle_solver_time_limit_must_cover_online_budget(self):
        from pac.experiment_config import load_supervised_config

        source = yaml.safe_load(V3_PATH.read_text(encoding="utf-8"))
        source["oracle"]["mpc_solver_time_limit_s"] = 0.007
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.yaml"
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mpc_solver_time_limit_s"):
                load_supervised_config(path)

    def test_every_seed_role_rejects_negative_seeds(self):
        from pac.experiment_config import load_supervised_config

        seed_fields = {
            ("training", "oracle_train_seeds"): "oracle_train",
            ("training", "oracle_val_seeds"): "oracle_val",
            ("training", "model_seeds"): "model",
            ("evaluation", "episode_seeds"): "evaluation",
            ("sspo", "search_seeds"): "sspo_search",
            ("sspo", "eval_seeds"): "sspo_eval",
            ("robustness", "eval_seeds"): "robustness_eval",
        }
        for (section, field), role in seed_fields.items():
            source = yaml.safe_load(V3_PATH.read_text(encoding="utf-8"))
            source[section][field] = [-1]
            with self.subTest(role=role):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "invalid.yaml"
                    path.write_text(yaml.safe_dump(source), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, rf"{role}.*-1"):
                        load_supervised_config(path)

    def test_every_seed_role_enforces_uint64_bounds(self):
        from pac.experiment_config import load_supervised_config

        seed_fields = {
            ("training", "oracle_train_seeds"): "oracle_train",
            ("training", "oracle_val_seeds"): "oracle_val",
            ("training", "model_seeds"): "model",
            ("evaluation", "episode_seeds"): "evaluation",
            ("sspo", "search_seeds"): "sspo_search",
            ("sspo", "eval_seeds"): "sspo_eval",
            ("robustness", "eval_seeds"): "robustness_eval",
        }
        max_seed = 2**64 - 1
        for (section, field), role in seed_fields.items():
            too_large = yaml.safe_load(V3_PATH.read_text(encoding="utf-8"))
            too_large[section][field] = [2**64]
            accepted = yaml.safe_load(V3_PATH.read_text(encoding="utf-8"))
            accepted[section][field] = [max_seed]
            with self.subTest(role=role, case="too_large"):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "invalid.yaml"
                    path.write_text(yaml.safe_dump(too_large), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, rf"{role}.*18446744073709551616"):
                        load_supervised_config(path)
            with self.subTest(role=role, case="max"):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "valid.yaml"
                    path.write_text(yaml.safe_dump(accepted), encoding="utf-8")
                    config = load_supervised_config(path)
                    section_config = {
                        "oracle_train_seeds": config.training.oracle_train_seeds,
                        "oracle_val_seeds": config.training.oracle_val_seeds,
                        "model_seeds": config.training.model_seeds,
                        "episode_seeds": config.evaluation.episode_seeds,
                        "search_seeds": config.sspo.search_seeds,
                        "eval_seeds": (
                            config.sspo.eval_seeds
                            if section == "sspo"
                            else config.robustness.eval_seeds
                        ),
                    }[field]
                    self.assertEqual(section_config, (max_seed,))

    def test_environment_scale_and_initial_std_bounds(self):
        from pac.experiment_config import load_supervised_config

        invalid_fields = {
            "mass_scale_xy": 0.0,
            "damping_scale_xy": -0.1,
            "current_frequency_scale": 0.0,
            "current_amplitude_scale": -0.1,
            "eval_initial_position_std": -0.1,
            "eval_initial_velocity_std": -0.1,
        }
        for field, invalid in invalid_fields.items():
            source = yaml.safe_load(V3_PATH.read_text(encoding="utf-8"))
            source["environment"][field] = invalid
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "invalid.yaml"
                    path.write_text(yaml.safe_dump(source), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "environment"):
                        load_supervised_config(path)

        source = yaml.safe_load(V3_PATH.read_text(encoding="utf-8"))
        source["environment"].update(
            {
                "current_amplitude_scale": 0.0,
                "eval_initial_position_std": 0.0,
                "eval_initial_velocity_std": 0.0,
                "vertical_current": -10.0,
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "valid.yaml"
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            config = load_supervised_config(path)
            self.assertEqual(config.environment.current_amplitude_scale, 0.0)
            self.assertEqual(config.environment.eval_initial_position_std, 0.0)
            self.assertEqual(config.environment.eval_initial_velocity_std, 0.0)
            self.assertEqual(config.environment.vertical_current, -10.0)

    def test_non_finite_float_values_are_rejected(self):
        from pac.experiment_config import _float_value, load_supervised_config

        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    _float_value(value, "test_value")

        source = yaml.safe_load(V3_PATH.read_text(encoding="utf-8"))
        source["environment"]["dt"] = float("nan")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.yaml"
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite"):
                load_supervised_config(path)

    def test_formal_scenarios_must_be_exactly_one_two_three(self):
        from pac.experiment_config import load_supervised_config

        for scenarios in ([1, 2], [1, 2, 4]):
            source = yaml.safe_load(V3_PATH.read_text(encoding="utf-8"))
            source["environment"]["scenarios"] = scenarios
            with self.subTest(scenarios=scenarios):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "invalid.yaml"
                    path.write_text(yaml.safe_dump(source), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, r"scenarios.*1.*2.*3"):
                        load_supervised_config(path)

    def test_required_strings_reject_null_list_and_blank_values(self):
        from pac.experiment_config import load_supervised_config

        fields = [
            (("protocol", "version"), None),
            (("controller", "primary"), []),
            (("controller", "authority"), "  "),
            (("environment", "vehicle_profile"), None),
            (("environment", "thruster_layout"), []),
            (("outputs", "run_root"), ""),
            (("outputs", "evidence_root"), None),
        ]
        for (section, field), invalid in fields:
            source = yaml.safe_load(V3_PATH.read_text(encoding="utf-8"))
            source[section][field] = invalid
            with self.subTest(field=f"{section}.{field}"):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "invalid.yaml"
                    path.write_text(yaml.safe_dump(source), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "non-empty string"):
                        load_supervised_config(path)

    def test_output_paths_must_be_relative_and_contained(self):
        from pac.experiment_config import load_supervised_config

        for field, invalid in (("run_root", "C" + ":/absolute"), ("evidence_root", "results/../escape")):
            source = yaml.safe_load(V3_PATH.read_text(encoding="utf-8"))
            source["outputs"][field] = invalid
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "invalid.yaml"
                    path.write_text(yaml.safe_dump(source), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, r"relative|parent"):
                        load_supervised_config(path)


if __name__ == "__main__":
    unittest.main()
