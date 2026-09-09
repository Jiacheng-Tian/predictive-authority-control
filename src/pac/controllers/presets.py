"""Controller presets for legacy-v2 and true-MPC-v3 experiments."""

from __future__ import annotations

from pac.controllers.legacy_predictive import LegacyOneStepPredictiveController
from pac.controllers.mpc import MPCController
from pac.controllers.mpc_qp import MPCQPSettings
from pac.controllers.smc import SMCController
from pac.simulation.dynamics import AUVDynamics
from pac.simulation.thrusters import build_real_10kg_x_layout
from pac.simulation.vehicle_profiles import get_vehicle_profile


def build_legacy_one_step_predictive_v2() -> LegacyOneStepPredictiveController:
    """Build the archived formal-v2 one-step predictive authority expert."""
    profile = get_vehicle_profile("real_10kg_v1")
    return LegacyOneStepPredictiveController(
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


def build_real10kg_mpc_event() -> LegacyOneStepPredictiveController:
    """Compatibility alias for the archived formal-v2 authority expert."""
    return build_legacy_one_step_predictive_v2()


def build_real10kg_mpc_ltv_v3() -> MPCController:
    """Build the finite-horizon true-MPC authority expert."""
    layout = build_real_10kg_x_layout()
    fallback = build_real10kg_smc_steady()
    fallback.set_trajectory3d(True)
    return MPCController(
        dynamics=AUVDynamics(vehicle_profile="real_10kg_v1", dt=0.01),
        thruster_layout=layout,
        settings=MPCQPSettings(N=20),
        fallback_controller=fallback,
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
