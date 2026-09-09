from __future__ import annotations

import unittest

import numpy as np


class EpisodePairingTest(unittest.TestCase):
    def test_episode_spec_is_byte_stable_and_readonly(self):
        from pac.evaluation.episode_spec import build_episode_spec

        first = build_episode_spec(2, 41000, steps=4, dt=0.01)
        second = build_episode_spec(2, 41000, steps=4, dt=0.01)
        self.assertEqual(first.episode_uid, "scn2_env41000")
        self.assertEqual(first.initial_eta.tobytes(), second.initial_eta.tobytes())
        self.assertEqual(first.initial_nu.tobytes(), second.initial_nu.tobytes())
        self.assertEqual(
            first.current_estimation_noise.tobytes(),
            second.current_estimation_noise.tobytes(),
        )
        self.assertFalse(first.initial_eta.flags.writeable)
        with self.assertRaises(ValueError):
            first.initial_eta[0] = 1.0
        self.assertNotEqual(
            first.current_estimation_noise.tobytes(),
            build_episode_spec(2, 41001, steps=4, dt=0.01).current_estimation_noise.tobytes(),
        )

    def test_paired_grid_reuses_environment_seed_across_learned_models(self):
        from pac.evaluation.protocol import build_paired_evaluation_grid

        rows = build_paired_evaluation_grid(
            model_seeds=[31000, 31001],
            episode_seeds=[41000],
            scenarios=[1],
        )
        learned = [row for row in rows if row["row_type"] == "learned"]
        baseline = [row for row in rows if row["row_type"] == "baseline"]
        self.assertEqual({row["environment_seed"] for row in learned}, {41000})
        self.assertEqual({row["model_seed"] for row in learned}, {31000, 31001})
        self.assertTrue(all(row["model_seed"] is None for row in baseline))
        self.assertTrue(all(row["episode_uid"] == "scn1_env41000" for row in rows))

    def test_paired_grid_accepts_zero_seeds_and_positive_scenarios(self):
        from pac.evaluation.protocol import build_paired_evaluation_grid

        rows = build_paired_evaluation_grid(
            model_seeds=[0], episode_seeds=[1], scenarios=[4]
        )
        self.assertEqual({row["model_seed"] for row in rows if row["row_type"] == "learned"}, {0})
        self.assertEqual({row["episode_seed"] for row in rows}, {1})
        self.assertEqual({row["scenario_id"] for row in rows}, {4})
        self.assertEqual({row["episode_uid"] for row in rows}, {"scn4_env1"})
        with self.assertRaises(ValueError):
            build_paired_evaluation_grid(model_seeds=[-1], episode_seeds=[1], scenarios=[1])
        with self.assertRaises(ValueError):
            build_paired_evaluation_grid(model_seeds=[0], episode_seeds=[-1], scenarios=[1])

    def test_episode_spec_rejects_float_integer_fields(self):
        from pac.evaluation.episode_spec import build_episode_spec

        common = dict(scenario_id=1, episode_seed=0, steps=2, dt=0.01)
        with self.assertRaises(ValueError):
            build_episode_spec(**{**common, "episode_seed": 0.0})
        with self.assertRaises(ValueError):
            build_episode_spec(**{**common, "steps": np.float64(2.0)})
        with self.assertRaises(ValueError):
            build_episode_spec(**{**common, "scenario_id": True})

    def test_simulator_reset_accepts_spec_without_resampling(self):
        from pac.evaluation.episode_spec import build_episode_spec
        from pac.simulation.core import AUVSimulator

        spec = build_episode_spec(1, 41000, steps=3, dt=0.02)
        simulator = AUVSimulator(scenario=1, max_steps=3, dt=0.02)
        simulator.reset(episode_spec=spec)
        np.testing.assert_array_equal(simulator.dynamics.eta, spec.initial_eta)
        np.testing.assert_array_equal(simulator.dynamics.nu, spec.initial_nu)

    def test_gym_reset_options_forward_episode_spec(self):
        from pac.evaluation.episode_spec import build_episode_spec
        from pac.simulation.environment import AUVTrackingEnv

        spec = build_episode_spec(1, 41000, steps=3, dt=0.02)
        environment = AUVTrackingEnv(scenario=1, max_steps=3, dt=0.02)
        environment.reset(options={"episode_spec": spec})
        np.testing.assert_array_equal(environment.simulator.dynamics.eta, spec.initial_eta)


if __name__ == "__main__":
    unittest.main()
