"""Reusable fixed baseline presets for stress-test evaluations."""
from __future__ import annotations

from controllers.mpc import MPCController
from controllers.smc import SMCController
from env.thrusters import build_real_10kg_x_layout
from env.vehicle_profiles import get_vehicle_profile


def build_real10kg_mpc_3d() -> MPCController:
    """Scenario-balanced MPC controller for the real 10 kg thruster-layout model."""
    profile = get_vehicle_profile("real_10kg_v1")
    layout = build_real_10kg_x_layout()
    return MPCController(
        horizon=10,
        M=profile.effective_mass,
        D=profile.linear_damping,
        D_quad=profile.quadratic_damping,
        q_pos=2.5,
        q_att=2.0,
        q_nu=220.0,
        z_pos_gain=0.18,
        z_current_feedforward=1.0,
        r=1e-4,
        smooth_alpha=0.8,
        action_mode="thruster",
        thruster_layout=layout,
    )


def build_real10kg_mpc_event() -> MPCController:
    """Aggressive MPC expert biased toward startup and disturbance events."""
    profile = get_vehicle_profile("real_10kg_v1")
    layout = build_real_10kg_x_layout()
    return MPCController(
        horizon=10,
        M=profile.effective_mass,
        D=profile.linear_damping,
        D_quad=profile.quadratic_damping,
        q_pos=16.0,
        q_att=2.0,
        q_nu=400.0,
        z_pos_gain=0.22,
        z_current_feedforward=1.0,
        r=1e-5,
        smooth_alpha=0.0,
        action_mode="thruster",
        thruster_layout=layout,
    )


def build_real10kg_smc() -> SMCController:
    """Initial SMC controller that allocates wrench commands to six thrusters."""
    profile = get_vehicle_profile("real_10kg_v1")
    layout = build_real_10kg_x_layout()
    return SMCController(
        lambda_gain=[0.8, 0.8, 1.0, 1.0, 1.0, 1.4],
        reaching_gain=[2.0, 2.0, 2.0, 0.0, 0.0, 4.0],
        equivalent_mass=profile.effective_mass,
        equivalent_damping=profile.linear_damping,
        equivalent_quadratic_damping=profile.quadratic_damping,
        current_feedforward=1.0,
        action_mode="thruster",
        thruster_layout=layout,
        control_yaw=True,
    )


def build_real10kg_smc_steady() -> SMCController:
    """Low-reaching SMC expert biased toward steady low-jerk tracking."""
    profile = get_vehicle_profile("real_10kg_v1")
    layout = build_real_10kg_x_layout()
    return SMCController(
        lambda_gain=[1.5, 1.5, 1.4, 1.0, 1.0, 1.4],
        reaching_gain=[2.0, 2.0, 4.0, 0.0, 0.0, 4.0],
        robust_gain=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        equivalent_mass=profile.effective_mass,
        equivalent_damping=profile.linear_damping,
        equivalent_quadratic_damping=profile.quadratic_damping,
        current_feedforward=1.0,
        action_mode="thruster",
        thruster_layout=layout,
        control_yaw=True,
    )


def build_mpc_tuned(config: dict) -> MPCController:
    """Build the high-authority MPC sensitivity baseline from config."""
    return MPCController(**config["mpc"])


def build_smc_tuned(config: dict) -> SMCController:
    """Build the high-authority equivalent SMC sensitivity baseline from config."""
    return SMCController(**config["smc"])
