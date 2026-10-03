from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest


class ControllerPairTest(unittest.TestCase):
    def test_pair_reflects_configured_mpc_and_plant_parameters(self):
        from pac.authority.training import build_controller_pair
        from pac.experiment_config import load_supervised_config

        root = Path(__file__).resolve().parents[1]
        config = load_supervised_config(root / "config" / "pac_supervised.yaml")
        mpc = replace(
            config.mpc,
            horizon=7,
            q_diag=tuple(value + 1.0 for value in config.mpc.q_diag),
        )
        actuator = replace(config.actuator, max_force_n=42.0, max_delta_per_step=0.07)
        environment = replace(config.environment, dt=0.02, vehicle_profile="real_10kg_v1")
        pair_config = replace(config, mpc=mpc, actuator=actuator, environment=environment)
        primary, authority = build_controller_pair(pair_config)
        self.assertEqual(authority.settings.N, 7)
        self.assertEqual(authority.settings.q_diag, mpc.q_diag)
        self.assertEqual(authority.settings.u_min, actuator.command_min)
        self.assertEqual(authority.settings.slew_limit, actuator.max_delta_per_step)
        self.assertEqual(authority.dynamics.dt, environment.dt)
        self.assertEqual(authority.thruster_layout.max_force, actuator.max_force_n)
        self.assertEqual(primary.thruster_layout.max_force, actuator.max_force_n)

    def test_online_and_oracle_solver_budgets_are_explicit(self):
        from pac.authority.training import build_controller_pair
        from pac.experiment_config import load_supervised_config

        config = load_supervised_config(Path(__file__).resolve().parents[1] / "config" / "pac_supervised.yaml")
        _, online = build_controller_pair(config)
        _, oracle = build_controller_pair(
            config, solver_time_limit_s=config.oracle.mpc_solver_time_limit_s
        )
        self.assertEqual(online.settings.time_limit_s, 0.0075)
        self.assertEqual(oracle.settings.time_limit_s, 0.05)
        self.assertEqual(config.mpc.time_limit_s, 0.0075)
