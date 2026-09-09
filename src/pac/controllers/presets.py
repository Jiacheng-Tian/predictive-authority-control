"""Formal-v2-compatible controller presets."""

from __future__ import annotations

from pac.controllers.predictive import OneStepPredictiveController
from pac.controllers.smc import SMCController
from pac.simulation.thrusters import build_real_10kg_x_layout
from pac.simulation.vehicle_profiles import get_vehicle_profile


def build_real10kg_predictive_event() -> OneStepPredictiveController:
    """Build the one-step predictive authority expert."""
    profile = get_vehicle_profile("real_10kg_v1")
    return OneStepPredictiveController(
        M=profile.effective_mass,
        D=profile.linear_damping,
        D_quad=profile.quadratic_damping,
        q_pos=16.0,
        q_att=2.0,
        q_nu=400.0,
        z_pos_gain=0.22,
        z_current_feedforward=1.0,
        r=1e-5,
        thruster_layout=build_real_10kg_x_layout(),
    )


def build_real10kg_smc_steady() -> SMCController:
    """Build the steady-tracking SMC primary expert."""
    profile = get_vehicle_profile("real_10kg_v1")
    return SMCController(
        lambda_gain=[1.5, 1.5, 1.4, 1.0, 1.0, 1.4],
        reaching_gain=[2.0, 2.0, 4.0, 0.0, 0.0, 4.0],
        robust_gain=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        equivalent_mass=profile.effective_mass,
        equivalent_damping=profile.linear_damping,
        equivalent_quadratic_damping=profile.quadratic_damping,
        current_feedforward=1.0,
        thruster_layout=build_real_10kg_x_layout(),
        control_yaw=True,
    )
