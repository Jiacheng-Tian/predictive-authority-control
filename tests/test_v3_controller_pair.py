from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest


class V3ControllerPairTest(unittest.TestCase):
    def test_pair_reflects_configured_mpc_and_plant_parameters(self):
        from pac.authority.training import build_v3_controller_pair
        from pac.experiment_config import load_v3_config

        root = Path(__file__).resolve().parents[1]
        config = load_v3_config(root / "config" / "pac_v3.yaml")
        mpc = replace(
            config.mpc,
            horizon=7,
            q_diag=tuple(value + 1.0 for value in config.mpc.q_diag),
        )
        actuator = replace(config.actuator, max_force_n=42.0, max_delta_per_step=0.07)
        environment = replace(config.environment, dt=0.02, vehicle_profile="real_10kg_v1")
        pair_config = replace(config, mpc=mpc, actuator=actuator, environment=environment)
        primary, authority = build_v3_controller_pair(pair_config)
        self.assertEqual(authority.settings.N, 7)
        self.assertEqual(authority.settings.q_diag, mpc.q_diag)
        self.assertEqual(authority.settings.u_min, actuator.command_min)
        self.assertEqual(authority.settings.slew_limit, actuator.max_delta_per_step)
        self.assertEqual(authority.dynamics.dt, environment.dt)
        self.assertEqual(authority.thruster_layout.max_force, actuator.max_force_n)
        self.assertEqual(primary.thruster_layout.max_force, actuator.max_force_n)
