from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class _Dynamics:
    dt = 0.1

    def __init__(self):
        self.eta = np.zeros(6)
        self.nu = np.zeros(6)
        self.calls = []

    def predict_step(self, eta, nu, tau, v_current):
        self.calls.append((np.asarray(eta).copy(), np.asarray(nu).copy(), np.asarray(tau).copy()))
        return np.asarray(eta, dtype=float).copy(), np.asarray(nu, dtype=float).copy()


class _Simulator:
    def __init__(self):
        self.dynamics = _Dynamics()
        self.actuator = type("Actuator", (), {"limits": None, "_previous_applied": np.zeros(6)})()

    @staticmethod
    def _get_target(t):
        return np.zeros(6, dtype=float)

    @staticmethod
    def _action_to_wrench(action):
        return np.asarray(action, dtype=float), None


class _Primary:
    def __init__(self):
        self.calls = []

    def compute(self, target, eta, nu, *, t=None, current_prediction=None):
        self.calls.append(float(t))
        return np.zeros(6, dtype=float)


class _Authority:
    def __init__(self):
        self.calls = 0

    def compute(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("the oracle must reuse the supplied MPC plan")


class RolloutOracleTest(unittest.TestCase):
    def test_ties_choose_lower_alpha_and_reuse_plan_without_mpc_calls(self):
        from pac.authority.oracle import OracleSettings, choose_rollout_oracle_alpha
        from pac.simulation.actuators import ActuatorLimits

        simulator = _Simulator()
        primary = _Primary()
        authority = _Authority()
        decision = choose_rollout_oracle_alpha(
            current_eta=np.zeros(6),
            current_nu=np.zeros(6),
            t=0.0,
            previous_applied=np.zeros(6),
            primary_controller=primary,
            mpc_controller=authority,
            mpc_plan=np.ones((2, 6), dtype=float),
            simulator=simulator,
            current_provider=lambda stamp: np.zeros(3),
            actuator_limits=ActuatorLimits(-1.0, 1.0, 10.0),
            settings=OracleSettings(horizon=3, alpha_grid=(0.0, 0.5, 1.0)),
        )

        self.assertEqual(decision.alpha, 0.0)
        self.assertEqual(len(decision.costs), 3)
        self.assertEqual(authority.calls, 0)
        self.assertEqual(len(primary.calls), 9)
        self.assertEqual(len(simulator.dynamics.calls), 9)

    def test_oracle_does_not_mutate_real_simulator_state_or_actuator(self):
        from pac.authority.oracle import choose_rollout_oracle_alpha
        from pac.simulation.actuators import ActuatorLimits

        simulator = _Simulator()
        simulator.dynamics.eta[:] = 3.0
        simulator.dynamics.nu[:] = 4.0
        simulator.actuator.limits = ActuatorLimits(-1.0, 1.0, 0.1)
        simulator.actuator._previous_applied[:] = 0.25
        eta_before = simulator.dynamics.eta.copy()
        nu_before = simulator.dynamics.nu.copy()
        applied_before = simulator.actuator._previous_applied.copy()
        choose_rollout_oracle_alpha(
            current_eta=eta_before,
            current_nu=nu_before,
            t=0.0,
            previous_applied=applied_before,
            primary_controller=_Primary(),
            mpc_controller=_Authority(),
            mpc_plan=np.zeros((3, 6)),
            simulator=simulator,
            current_provider=lambda stamp: np.zeros(3),
            actuator_limits=simulator.actuator.limits,
        )
        np.testing.assert_array_equal(simulator.dynamics.eta, eta_before)
        np.testing.assert_array_equal(simulator.dynamics.nu, nu_before)
        np.testing.assert_array_equal(simulator.actuator._previous_applied, applied_before)

    def test_v3_collection_rejects_missing_or_reused_mpc_plan(self):
        from pac.authority.training import collect_teacher_dataset_v3
        from pac.experiment_config import load_v3_config

        config = load_v3_config(Path(__file__).resolve().parents[1] / "config" / "pac_v3.yaml")

        class FakeController:
            def __init__(self, authority=False, mode="none"):
                self.authority = authority
                self.mode = mode
                self.last_plan = np.zeros((20, 6)) if authority else None
                self.last_telemetry = {
                    "accepted": mode == "none",
                    "fallback_mode": mode,
                }

            def reset(self):
                pass

            def set_trajectory3d(self, enabled=True):
                pass

            def compute(self, target, eta, nu, *, t=None, current_prediction=None):
                return np.zeros(6)

        for mode in ("solver_failure", "reuse_plan"):
            with self.subTest(mode=mode):
                def build(name, current_mode=mode):
                    return name, FakeController(authority="mpc" in name, mode=current_mode if "mpc" in name else "none")

                with patch("pac.authority.training.build_controller", side_effect=build):
                    with self.assertRaisesRegex(RuntimeError, r"episode=.*step=0"):
                        collect_teacher_dataset_v3(config, "short")


if __name__ == "__main__":
    unittest.main()
