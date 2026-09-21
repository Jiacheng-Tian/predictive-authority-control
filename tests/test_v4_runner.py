"""Parity and behavior tests for the v4 evaluation runner.

The two parity tests are the stage-2 validity contract: on structured
episodes the v4 loop must reproduce the archived v3 evaluation numbers
exactly, so every method column in the formal v4 tables is comparable with
the frozen v3 baseline.
"""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from pac.authority.evaluation import run_predictive_alpha_episode, scalar_metrics
from pac.evaluation.episode_spec import build_episode_spec
from pac.evaluation.episodes import run_fixed_controller_episode
from pac.v4.config import load_v4_config
from pac.v4.disturbances import build_v4_episode_spec
from pac.v4.eval.runner import (
    ConstantAlphaPolicy,
    FixedControllerPolicy,
    WMAdvice,
    build_backbone_policy,
    run_v4_policy_episode,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "pac_v4.yaml"
PARITY_SEED = 41234
PARITY_STEPS = 400
FIXED_STEPS = 300
# Wall-clock-dependent solver metrics cannot be reproduced across runs;
# parity tests assert them loosely instead of exactly.
_TIMING_METRICS = {"solver_deadline_miss_step_fraction", "solver_fallback_step_fraction"}


def _float_metrics(metrics: dict) -> dict[str, float]:
    return {
        key: value
        for key, value in scalar_metrics(metrics).items()
        if isinstance(value, (float, int, bool)) and not isinstance(value, str)
    }


def _assert_paired_metrics(test: unittest.TestCase, v3_metrics, v4_metrics) -> None:
    v3_floats = _float_metrics(v3_metrics)
    v4_floats = _float_metrics(v4_metrics)
    missing = sorted(set(v3_floats) - set(v4_floats))
    test.assertEqual(missing, [], "v4 metrics must cover every numeric v3 metric")
    for key, expected in v3_floats.items():
        actual = v4_floats[key]
        if key in _TIMING_METRICS:
            test.assertLessEqual(
                abs(actual - expected),
                0.05,
                f"timing metric {key} diverged beyond tolerance",
            )
            continue
        if np.isnan(float(expected)) or np.isnan(float(actual)):
            # Short episodes leave some derived windows undefined on both
            # sides; NaN must agree with NaN.
            test.assertTrue(
                np.isnan(float(expected)) and np.isnan(float(actual)),
                f"metric {key} NaN mismatch",
            )
            continue
        test.assertAlmostEqual(
            actual,
            expected,
            delta=max(1.0e-9, abs(expected) * 1.0e-9),
            msg=f"metric {key} diverged",
        )


class V4RunnerParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_v4_config(CONFIG_PATH)

    def test_v3_transformer_parity_on_structured_episode(self):
        v3_spec = build_episode_spec(1, PARITY_SEED, PARITY_STEPS, 0.01)
        v4_spec = build_v4_episode_spec(
            "structured", 1, PARITY_SEED, PARITY_STEPS, 0.01
        )
        policy = build_backbone_policy(self.config, self.config.seeds.model[0])
        v3_metrics = run_predictive_alpha_episode(
            model=policy.model,
            scenario=1,
            seed=PARITY_SEED,
            steps=PARITY_STEPS,
            mass_scale_xy=self.config.environment.mass_scale_xy,
            damping_scale_xy=self.config.environment.damping_scale_xy,
            current_amplitude_scale=self.config.environment.current_amplitude_scale,
            current_frequency_scale=self.config.environment.current_frequency_scale,
            vertical_current=self.config.environment.vertical_current,
            primary_controller=self.config.controller.primary,
            authority_controller=self.config.controller.authority,
            feature_mode=self.config.authority_model.feature_mode,
            initial_position_std=self.config.environment.eval_initial_position_std,
            initial_velocity_std=self.config.environment.eval_initial_velocity_std,
            alpha_gain=self.config.authority_model.alpha_gain,
            alpha_threshold=0.0,
            vehicle_profile=self.config.environment.vehicle_profile,
            thruster_layout=self.config.environment.thruster_layout,
            alpha_smoothing=self.config.authority_model.alpha_smoothing,
            alpha_rate_limit=self.config.authority_model.alpha_rate_limit,
            alpha_deadband=self.config.authority_model.alpha_deadband,
            policy_architecture="transformer",
            history_len=self.config.authority_model.history_len,
            save_ts=True,
            episode_spec=v3_spec,
            dt=self.config.environment.dt,
            actuator_max_delta_per_step=self.config.actuator.max_delta_per_step,
            aligned_metrics=True,
        )
        v4_metrics = run_v4_policy_episode(policy, v4_spec, self.config)
        _assert_paired_metrics(self, v3_metrics, v4_metrics)
        self.assertAlmostEqual(
            v4_metrics["rmse_3d"], v3_metrics["rmse_3d"], delta=1.0e-12
        )
        self.assertAlmostEqual(
            v4_metrics["heading_rmse_deg"], v3_metrics["heading_rmse_deg"], delta=1.0e-9
        )

    def test_fixed_controller_parity_primary(self):
        v3_spec = build_episode_spec(2, PARITY_SEED, FIXED_STEPS, 0.01)
        v4_spec = build_v4_episode_spec(
            "structured", 2, PARITY_SEED, FIXED_STEPS, 0.01
        )
        v3_metrics = run_fixed_controller_episode(
            scenario=2,
            seed=PARITY_SEED,
            steps=FIXED_STEPS,
            mass_scale_xy=self.config.environment.mass_scale_xy,
            damping_scale_xy=self.config.environment.damping_scale_xy,
            base_controller="real10kg_smc_steady",
            current_amplitude_scale=self.config.environment.current_amplitude_scale,
            current_frequency_scale=self.config.environment.current_frequency_scale,
            initial_position_std=self.config.environment.eval_initial_position_std,
            initial_velocity_std=self.config.environment.eval_initial_velocity_std,
            vertical_current=self.config.environment.vertical_current,
            vehicle_profile=self.config.environment.vehicle_profile,
            thruster_layout=self.config.environment.thruster_layout,
            save_ts=True,
            episode_spec=v3_spec,
            dt=self.config.environment.dt,
            actuator_max_delta_per_step=self.config.actuator.max_delta_per_step,
            aligned_metrics=True,
        )
        v4_metrics = run_v4_policy_episode(
            FixedControllerPolicy("primary"), v4_spec, self.config
        )
        _assert_paired_metrics(self, v3_metrics, v4_metrics)

    def test_fixed_controller_parity_authority(self):
        v3_spec = build_episode_spec(1, PARITY_SEED, FIXED_STEPS, 0.01)
        v4_spec = build_v4_episode_spec(
            "structured", 1, PARITY_SEED, FIXED_STEPS, 0.01
        )
        v3_metrics = run_fixed_controller_episode(
            scenario=1,
            seed=PARITY_SEED,
            steps=FIXED_STEPS,
            mass_scale_xy=self.config.environment.mass_scale_xy,
            damping_scale_xy=self.config.environment.damping_scale_xy,
            base_controller="real10kg_mpc_ltv_v3",
            current_amplitude_scale=self.config.environment.current_amplitude_scale,
            current_frequency_scale=self.config.environment.current_frequency_scale,
            initial_position_std=self.config.environment.eval_initial_position_std,
            initial_velocity_std=self.config.environment.eval_initial_velocity_std,
            vertical_current=self.config.environment.vertical_current,
            vehicle_profile=self.config.environment.vehicle_profile,
            thruster_layout=self.config.environment.thruster_layout,
            save_ts=True,
            episode_spec=v3_spec,
            dt=self.config.environment.dt,
            actuator_max_delta_per_step=self.config.actuator.max_delta_per_step,
            aligned_metrics=True,
        )
        v4_metrics = run_v4_policy_episode(
            FixedControllerPolicy("authority"), v4_spec, self.config
        )
        self.assertAlmostEqual(
            v4_metrics["rmse_3d"], v3_metrics["rmse_3d"], delta=1.0e-12
        )
        self.assertAlmostEqual(
            v4_metrics["energy"], v3_metrics["energy"], delta=1.0e-9
        )


class V4RunnerBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_v4_config(CONFIG_PATH)

    def test_constant_alpha_policy_on_ou_episode(self):
        spec = build_v4_episode_spec(
            "ou_current",
            2,
            45123,
            150,
            0.01,
            disturbance_params={"theta": 0.5, "sigma": 0.1, "mean_scale": 0.3},
        )
        metrics = run_v4_policy_episode(
            ConstantAlphaPolicy(0.5), spec, self.config
        )
        self.assertEqual(metrics["family"], "ou_current")
        # The constant passes through the v3 alpha stack: gain 1.2 lifts the
        # 0.5 request to 0.6, and the 0.0125/step rate limit ramps there over
        # 48 steps, giving a mean slightly above 0.5 across 150 steps.
        self.assertAlmostEqual(metrics["authority_alpha_mean"], 0.504, delta=0.03)
        self.assertLessEqual(metrics["authority_alpha_mean"], 0.6)
        self.assertTrue(np.isfinite(metrics["rmse_3d"]))

    def test_wm_advice_injection_and_refresh(self):
        class RecordingComputer:
            calls = 0

            def advise(self, **kwargs):
                RecordingComputer.calls += 1
                return WMAdvice(
                    best_alpha=0.3,
                    best_cost=1.0,
                    cost_spread=0.5,
                    uncertainty=1.0e-6,
                    gate_open=True,
                    compute_seconds=0.001,
                )

        spec = build_v4_episode_spec(
            "structured", 1, 45124, 61, 0.01
        )
        computer = RecordingComputer()
        metrics = run_v4_policy_episode(
            ConstantAlphaPolicy(0.2), spec, self.config, wm_computer=computer
        )
        # refresh every 10 steps over 61 decision steps (0..60)
        expected_calls = len([s for s in range(61) if s % 10 == 0])
        self.assertEqual(computer.calls, expected_calls)
        self.assertGreater(metrics["wm_gate_open_fraction"], 0.0)
        gate_flags = metrics["ts"]["wm_gate_open"].to_numpy()
        flagged = np.flatnonzero(gate_flags > 0)
        self.assertTrue(all(int(step) % 10 == 0 for step in flagged))


if __name__ == "__main__":
    unittest.main()
