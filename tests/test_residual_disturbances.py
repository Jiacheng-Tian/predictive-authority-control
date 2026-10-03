"""Tests for disturbance families and the DisturbanceSimulator."""

from __future__ import annotations

import unittest

import numpy as np

from pac.evaluation.episode_spec import build_episode_spec
from pac.residual.disturbances import (
    build_paired_episode_spec,
    draw_disturbance_params,
    make_disturbance_simulator,
    paired_episode_uid,
)

STEPS = 120


def _simulator(family, params, seed=32100, scenario=1, **kwargs):
    spec = build_paired_episode_spec(
        family, scenario, seed, STEPS, 0.01, disturbance_params=params
    )
    simulator = make_disturbance_simulator(spec, **kwargs)
    simulator.reset(episode_spec=spec)
    return simulator, spec


class PairedDisturbanceTests(unittest.TestCase):
    def test_structured_spec_matches_v3_realization(self):
        seed, steps = 424242, 200
        v3_spec = build_episode_spec(1, seed, steps, 0.01)
        v4_spec = build_paired_episode_spec("structured", 1, seed, steps, 0.01)
        np.testing.assert_allclose(v4_spec.initial_eta, v3_spec.initial_eta, atol=0.0)
        np.testing.assert_allclose(v4_spec.initial_nu, v3_spec.initial_nu, atol=0.0)
        np.testing.assert_allclose(
            v4_spec.current_estimation_noise,
            v3_spec.current_estimation_noise,
            atol=0.0,
        )

    def test_episode_uid_contract(self):
        self.assertEqual(
            paired_episode_uid("ou_current", 2, 21000), "scn2_ou_current_env21000"
        )
        with self.assertRaises(ValueError):
            build_paired_episode_spec("unknown_family", 1, 1, 10, 0.01)

    def test_structured_simulator_matches_base_currents(self):
        from pac.simulation.core import AUVSimulator

        base = AUVSimulator(scenario=1, max_steps=STEPS)
        base.reset(seed=7)
        structured, _spec = _simulator("structured", {}, seed=7)
        for _ in range(50):
            base_current = base._current_for_dynamics(base._generate_current(0.0))
            v4_current = structured._current_for_dynamics(structured._generate_current(0.0))
            np.testing.assert_allclose(base_current, v4_current, atol=0.0)
            base.step(np.zeros(6))
            structured.step(np.zeros(6))

    def test_ou_current_deterministic_and_stateful(self):
        params = {"theta": 0.6, "sigma": 0.1, "mean_scale": 0.3}
        first, _ = _simulator("ou_current", params, seed=99)
        second, _ = _simulator("ou_current", params, seed=99)
        trajectories = []
        for simulator in (first, second):
            values = [
                simulator._current_for_dynamics(simulator._generate_current(0.0)).copy()
            ]
            for _ in range(30):
                simulator.step(np.zeros(6))
                values.append(
                    simulator._current_for_dynamics(simulator._generate_current(0.0)).copy()
                )
            trajectories.append(np.stack(values))
        np.testing.assert_allclose(trajectories[0], trajectories[1], atol=0.0)
        self.assertGreater(float(np.std(trajectories[0][:, 0])), 0.0)

    def test_colored_noise_deterministic(self):
        params = {"rho": 0.95, "sigma": 0.15}
        first, _ = _simulator("colored_noise", params, seed=5)
        second, _ = _simulator("colored_noise", params, seed=5)
        values = []
        for simulator in (first, second):
            rows = [
                simulator._current_for_dynamics(simulator._generate_current(0.0)).copy()
            ]
            for _ in range(20):
                simulator.step(np.zeros(6))
                rows.append(
                    simulator._current_for_dynamics(simulator._generate_current(0.0)).copy()
                )
            values.append(np.stack(rows))
        np.testing.assert_allclose(values[0], values[1], atol=0.0)

    def test_stochastic_current_advances_once_per_step(self):
        params = {"theta": 0.6, "sigma": 0.1, "mean_scale": 0.3}
        simulator, _ = _simulator("ou_current", params, seed=11)
        simulator.step(np.zeros(6))
        measured = [
            simulator._current_for_dynamics(simulator._generate_current(0.0)).copy()
            for _ in range(5)
        ]
        for row in measured[1:]:
            np.testing.assert_allclose(measured[0], row, atol=0.0)

    def test_actuator_delay_and_noise(self):
        params = {"delay_steps": 2, "action_noise_std": 0.0}
        simulator, _ = _simulator("actuator_delay_noise", params, seed=3, scenario=3)
        simulator.step(np.full(6, 0.5))
        simulator.step(np.full(6, 0.9))
        simulator.step(np.full(6, -0.8))
        applied = np.asarray(simulator.actuator_telemetry["applied_action"])
        np.testing.assert_allclose(applied[0], np.zeros(6), atol=0.0)
        np.testing.assert_allclose(applied[1], np.zeros(6), atol=0.0)
        np.testing.assert_allclose(applied[2], np.full(6, 0.5), atol=1e-12)

        noisy_params = {"delay_steps": 0, "action_noise_std": 0.05}
        noisy, _ = _simulator("actuator_delay_noise", noisy_params, seed=4)
        noisy.step(np.zeros(6))
        applied_noisy = np.asarray(noisy.actuator_telemetry["applied_action"])
        self.assertGreater(float(np.abs(applied_noisy).max()), 0.0)
        self.assertLessEqual(float(np.abs(applied_noisy).max()), 0.25)

    def test_random_freq_amp_uses_drawn_scales(self):
        params = {"amplitude_scale": 3.4, "frequency_scale": 1.5}
        simulator, _ = _simulator("random_freq_amp", params, seed=8, scenario=2)
        self.assertAlmostEqual(simulator.current_amplitude_scale, 3.4)
        self.assertAlmostEqual(simulator.current_frequency_scale, 1.5)

    def test_mass_damping_mismatch_propagates(self):
        params = {"mass_scale_xy": 1.2, "damping_scale_xy": 0.8}
        simulator, _ = _simulator("mass_damping_mismatch", params, seed=8, scenario=2)
        baseline = _simulator("structured", {}, seed=8, scenario=2)[0]
        # effective mass (rigid + added) for surge is 14.0 at scale 1.0
        self.assertAlmostEqual(
            float(simulator.dynamics.M[0, 0]),
            float(baseline.dynamics.M[0, 0]) * 1.2,
        )
        self.assertAlmostEqual(
            float(simulator.dynamics.D[0, 0]),
            float(baseline.dynamics.D[0, 0]) * 0.8,
        )

    def test_draw_disturbance_params_deterministic_and_bounded(self):
        ranges = {"theta": (0.2, 1.0), "sigma": (0.05, 0.2), "mean_scale": (0.0, 0.5)}
        first = draw_disturbance_params("ou_current", 21000, ranges)
        second = draw_disturbance_params("ou_current", 21000, ranges)
        self.assertEqual(first, second)
        self.assertTrue(0.2 <= first["theta"] <= 1.0)
        self.assertTrue(0.05 <= first["sigma"] <= 0.2)
        integer_ranges = {"delay_steps": (2, 4), "action_noise_std": (0.0, 0.02)}
        delayed = draw_disturbance_params("actuator_delay_noise", 21000, integer_ranges)
        self.assertIsInstance(delayed["delay_steps"], int)
        self.assertTrue(2 <= delayed["delay_steps"] <= 4)

    def test_fast_ou_is_ou_dynamics_with_extrapolated_params(self):
        from pac.residual.disturbances import build_paired_episode_spec, make_disturbance_simulator

        params = {"theta": 4.5, "sigma": 0.4, "mean_scale": 0.3}
        spec = build_paired_episode_spec(
            "fast_ou", 2, 515001, STEPS, 0.01, disturbance_params=params
        )
        simulator = make_disturbance_simulator(spec)
        simulator.reset(episode_spec=spec)
        currents = []
        for step_index in range(60):
            simulator.step(np.zeros(6))
            currents.append(simulator.current_at_step(step_index).copy())
        array = np.stack(currents)
        self.assertTrue(np.isfinite(array).all())
        # theta=4.5 (time constant ~0.22 s) decorrelates the process well
        # within a 0.6 s window.
        lag0 = float(np.corrcoef(array[:-10, 0], array[:-10, 0])[0, 1])
        lag10 = float(np.corrcoef(array[:-10, 0], array[10:, 0])[0, 1])
        self.assertLess(lag10, 0.9 * lag0 + 0.1)

    def test_estimation_delay_overrides_spec_estimator(self):
        from pac.residual.disturbances import build_paired_episode_spec, make_disturbance_simulator

        params = {"delay_steps": 15, "noise_std": 0.08}
        spec = build_paired_episode_spec(
            "estimation_delay", 2, 515002, STEPS, 0.01, disturbance_params=params
        )
        self.assertEqual(spec.current_delay_steps, 15)
        self.assertAlmostEqual(spec.current_noise_std, 0.08)
        self.assertEqual(spec.family, "estimation_delay")
        simulator = make_disturbance_simulator(spec)
        simulator.reset(episode_spec=spec)
        # Structured (sinusoidal) currents: family must not alter dynamics.
        base = _simulator("structured", {}, seed=515002, scenario=2)[0]
        for _ in range(30):
            simulator.step(np.zeros(6))
            base.step(np.zeros(6))
        np.testing.assert_allclose(
            simulator.realized_current_trajectory, base.realized_current_trajectory,
            atol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
