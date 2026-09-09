from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch
from unittest.mock import patch


class MetricTimeAlignmentTest(unittest.TestCase):
    def test_engineering_metrics_use_applied_action_columns(self):
        from pac.evaluation.metrics import compute_timeseries_engineering_metrics

        frame = pd.DataFrame({
            "step": [0, 1],
            "time": [0.01, 0.02],
            "error": [0.1, 0.1],
            "action_0": [0.0, 0.0],
            "requested_action_0": [1.0, 1.0],
            "actuator_rate_limited_fraction": [0.5, 0.0],
        })
        metrics = compute_timeseries_engineering_metrics(frame)
        self.assertEqual(metrics["action_peak_abs"], 0.0)
        self.assertEqual(metrics["actuator_rate_limited_fraction_mean"], 0.25)

    def test_aligned_fixed_episode_records_post_state_and_applied_action(self):
        from pac.evaluation.episode_spec import build_episode_spec
        from pac.evaluation.episodes import run_fixed_controller_episode

        spec = build_episode_spec(1, 41000, steps=3, dt=0.01)
        metrics = run_fixed_controller_episode(
            scenario=1,
            seed=41000,
            steps=3,
            mass_scale_xy=1.0,
            damping_scale_xy=1.0,
            base_controller="real10kg_smc_steady",
            current_amplitude_scale=1.0,
            current_frequency_scale=1.0,
            vertical_current=0.0,
            episode_spec=spec,
            dt=0.01,
            actuator_max_delta_per_step=0.05,
            save_ts=True,
        )
        ts = metrics["ts"]
        self.assertTrue(np.allclose(ts["time"].to_numpy(), [0.01, 0.02, 0.03]))
        for index in range(6):
            self.assertTrue(np.allclose(ts[f"action_{index}"], ts[f"applied_action_{index}"]))
            self.assertIn(f"requested_action_{index}", ts)
        self.assertIn("estimated_current_0", ts)
        self.assertIn("true_current_0", ts)
        self.assertEqual(metrics["environment_seed"], 41000)
        self.assertEqual(metrics["episode_uid"], "scn1_env41000")

    def test_aligned_pac_uses_estimated_current_and_shared_spec_state(self):
        from pac.authority.evaluation import run_predictive_alpha_episode
        from pac.evaluation.episode_spec import build_episode_spec

        class Controller:
            def reset(self):
                pass

            def set_trajectory3d(self, enabled=True):
                del enabled

            def compute(self, *args, **kwargs):
                del args, kwargs
                return np.zeros(6)

        class Model:
            def eval(self):
                return self

            def __call__(self, values):
                return torch.zeros(values.shape[0], 1)

        spec = build_episode_spec(1, 41000, steps=2, dt=0.01)
        with patch(
            "pac.authority.evaluation.build_controller",
            side_effect=[("primary", Controller()), ("authority", Controller())],
        ):
            metrics = run_predictive_alpha_episode(
                model=Model(),
                scenario=1,
                seed=41000,
                steps=2,
                mass_scale_xy=1.0,
                damping_scale_xy=1.0,
                current_amplitude_scale=1.0,
                current_frequency_scale=1.0,
                vertical_current=0.0,
                primary_controller="primary",
                authority_controller="authority",
                feature_mode="state_phase",
                episode_spec=spec,
                dt=0.01,
                actuator_max_delta_per_step=0.05,
                save_ts=True,
            )
        self.assertEqual(metrics["ts"]["sample_time"].tolist(), [0.01, 0.02])
        self.assertIn("estimated_current_0", metrics["ts"])


if __name__ == "__main__":
    unittest.main()
