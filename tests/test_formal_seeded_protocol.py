import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd


class FormalSeededProtocolTest(unittest.TestCase):
    def test_formal_protocol_manifest_declares_training_and_eval_dimensions(self):
        from scripts import run_formal_seeded_protocol as protocol

        manifest = protocol.build_formal_protocol_manifest()

        self.assertEqual(manifest["pac_training_seeds"], [20, 21, 22, 23, 24])
        self.assertEqual(manifest["evaluation_episode_seeds"], list(range(10)))
        self.assertEqual(manifest["evaluation_episodes_per_train_seed"], 10)
        self.assertEqual(manifest["total_pac_rollouts_per_scenario"], 50)
        self.assertEqual(manifest["fixed_controller_rollouts_per_scenario"], 10)
        self.assertEqual(manifest["output_directory"], "results/formal_seeded_v2")
        self.assertEqual(manifest["figure_directory"], "results/figures/formal_seeded_v2")
        self.assertEqual(manifest["pac"]["epochs"], 180)
        self.assertEqual(manifest["pac"]["policy_architecture"], "transformer")
        self.assertNotIn("NN direct control", manifest["main_methods"])

    def test_merge_pac_seed_outputs_adds_explicit_episode_dimensions(self):
        from scripts import run_formal_seeded_protocol as protocol

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for train_seed in [20, 21]:
                run_dir = root / f"pac_train_seed_{train_seed}"
                run_dir.mkdir(parents=True)
                pd.DataFrame([
                    {
                        "method": "predictive_alpha",
                        "scenario": "constant",
                        "scenario_id": 1,
                        "seed": train_seed * 1000 + eval_seed,
                        "rmse": 0.1 + 0.001 * train_seed + 0.0001 * eval_seed,
                    }
                    for eval_seed in [0, 1, 2]
                ]).to_csv(run_dir / "raw_metrics.csv", index=False)
                pd.DataFrame([
                    {
                        "method": "predictive_alpha",
                        "scenario": "constant",
                        "scenario_id": 1,
                        "seed": train_seed * 1000 + eval_seed,
                        "window": "startup_0_3s",
                        "rmse_3d": 0.2,
                        "xy_rmse": 0.1,
                        "z_rmse": 0.01,
                        "window_action_jerk_mean": 0.001,
                        "window_action_saturation_step_fraction": 0.0,
                    }
                    for eval_seed in [0, 1, 2]
                ]).to_csv(run_dir / "window_metrics.csv", index=False)
                ts_dir = run_dir / "timeseries"
                ts_dir.mkdir()
                pd.DataFrame([
                    {
                        "method": "predictive_alpha",
                        "scenario": "constant",
                        "scenario_id": 1,
                        "seed": train_seed * 1000,
                        "time": 0.0,
                        "step": 0,
                        "error": 0.1,
                    }
                ]).to_csv(ts_dir / "timeseries_3d.csv", index=False)

            raw, window, timeseries = protocol.merge_pac_seed_outputs(
                root,
                train_seeds=[20, 21],
                eval_episode_seeds=[0, 1, 2],
            )

        self.assertEqual(len(raw), 6)
        self.assertEqual(set(raw["train_seed"]), {20, 21})
        self.assertEqual(set(raw["eval_episode"]), {0, 1, 2})
        self.assertEqual(raw["episode_uid"].nunique(), 6)
        self.assertIn("train_seed", window.columns)
        self.assertIn("eval_episode", window.columns)
        self.assertIn("episode_uid", window.columns)
        self.assertEqual(set(timeseries["train_seed"]), {20, 21})
        self.assertEqual(set(timeseries["eval_episode"]), {0})

    def test_formal_eval_episode_rows_are_cartesian_train_seed_and_eval_seed(self):
        from scripts import run_formal_seeded_protocol as protocol

        rows = protocol.formal_eval_episode_rows(
            train_seeds=[20, 21],
            eval_episode_seeds=[0, 1, 2],
        )

        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0], {"train_seed": 20, "eval_episode": 0, "sim_seed": 20000})
        self.assertEqual(rows[-1], {"train_seed": 21, "eval_episode": 2, "sim_seed": 21002})
        self.assertEqual(len({row["sim_seed"] for row in rows}), 6)

    def test_annotate_fixed_controller_outputs_recovers_formal_episode_labels(self):
        from scripts import run_formal_seeded_protocol as protocol

        frame = pd.DataFrame([
            {
                "controller": "real10kg_smc_steady",
                "scenario_id": 1,
                "seed": 20000,
                "train_seed": 999,
                "eval_episode": 999,
                "episode_uid": "stale",
            },
            {"controller": "real10kg_mpc_event", "scenario_id": 1, "seed": 21002},
        ])
        out = protocol.annotate_formal_eval_dimensions(
            frame,
            train_seeds=[20, 21],
            eval_episode_seeds=[0, 1, 2],
        )

        self.assertEqual(out.loc[0, "train_seed"], 20)
        self.assertEqual(out.loc[0, "eval_episode"], 0)
        self.assertEqual(out.loc[1, "train_seed"], 21)
        self.assertEqual(out.loc[1, "eval_episode"], 2)
        self.assertEqual(out["episode_uid"].tolist(), ["train20_eval0_scn1", "train21_eval2_scn1"])

    def test_annotate_fixed_controller_outputs_updates_timeseries_when_present(self):
        from scripts import run_formal_seeded_protocol as protocol

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame([
                {"controller": "real10kg_smc_steady", "scenario_id": 1, "seed": 20000},
            ]).to_csv(root / "raw_metrics.csv", index=False)
            pd.DataFrame([
                {"controller": "real10kg_smc_steady", "scenario_id": 1, "seed": 20000, "window": "startup_0_3s"},
            ]).to_csv(root / "window_metrics.csv", index=False)
            ts_dir = root / "timeseries"
            ts_dir.mkdir()
            pd.DataFrame([
                {"controller": "real10kg_smc_steady", "scenario_id": 1, "seed": 20000, "time": 0.0},
            ]).to_csv(ts_dir / "timeseries_3d.csv", index=False)

            protocol.annotate_fixed_controller_outputs(
                root,
                train_seeds=[20],
                eval_episode_seeds=[0],
            )
            ts = pd.read_csv(ts_dir / "timeseries_3d.csv")

        self.assertEqual(ts.loc[0, "train_seed"], 20)
        self.assertEqual(ts.loc[0, "eval_episode"], 0)
        self.assertEqual(ts.loc[0, "episode_uid"], "train20_eval0_scn1")

    def test_pac_command_uses_unique_simulator_seeds_for_formal_eval_episodes(self):
        from scripts import run_formal_seeded_protocol as protocol

        command = protocol.build_pac_command(
            out_dir="out",
            figure_dir="fig",
            train_seed=20,
            eval_episode_seeds=[0, 1],
            scenarios=[1],
            steps=60,
            epochs=1,
            no_timeseries=True,
        )

        eval_seed_index = command.index("--eval-seeds") + 1
        self.assertEqual(command[eval_seed_index], "20000,20001")

    def test_fixed_controller_command_uses_ten_unique_episodes_once(self):
        from scripts import run_formal_seeded_protocol as protocol

        command = protocol.build_fixed_controller_command(
            out_dir="out",
            figure_dir="fig",
            train_seeds=[20, 21, 22, 23, 24],
            eval_episode_seeds=list(range(10)),
            scenarios=[1, 2, 3],
            steps=60,
            no_figures=True,
        )

        seed_index = command.index("--seeds") + 1
        self.assertEqual(
            command[seed_index],
            "20000,20001,20002,20003,20004,20005,20006,20007,20008,20009",
        )

    def test_dry_run_does_not_create_or_rewrite_output_artifacts(self):
        from scripts import run_formal_seeded_protocol as protocol

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "formal"
            figure_dir = Path(tmp) / "figures"
            args = protocol.build_parser().parse_args([
                "--out-dir", str(out_dir),
                "--figure-dir", str(figure_dir),
                "--train-seeds", "20",
                "--eval-episode-seeds", "0",
                "--scenarios", "1",
                "--steps", "60",
                "--epochs", "1",
                "--dry-run",
            ])

            with redirect_stdout(io.StringIO()):
                protocol.run_formal_protocol(args)

            self.assertFalse(out_dir.exists())
            self.assertFalse(figure_dir.exists())


if __name__ == "__main__":
    unittest.main()
