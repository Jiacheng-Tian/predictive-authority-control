"""Controller presets for legacy-v2 and true-MPC-v3 experiments."""

from __future__ import annotations

import numpy as np

from pac.controllers.legacy_predictive import LegacyOneStepPredictiveController
from pac.controllers.mpc import MPCController
from pac.controllers.mpc_qp import MPCQPSettings
from pac.controllers.smc import SMCController
from pac.simulation.dynamics import AUVDynamics
from pac.simulation.thrusters import build_real_10kg_x_layout, build_thruster_layout
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


def build_real10kg_smc_steady(
        *,
        vehicle_profile: str = "real_10kg_v1",
        thruster_layout=None) -> SMCController:
    """Build the steady-tracking SMC primary expert."""
    profile = get_vehicle_profile(vehicle_profile)
    layout = build_real_10kg_x_layout() if thruster_layout is None else thruster_layout
    return SMCController(
        lambda_gain=[1.5, 1.5, 1.4, 1.0, 1.0, 1.4],
        reaching_gain=[2.0, 2.0, 4.0, 0.0, 0.0, 4.0],
        robust_gain=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        equivalent_mass=profile.effective_mass,
        equivalent_damping=profile.linear_damping,
        equivalent_quadratic_damping=profile.quadratic_damping,
        current_feedforward=1.0,
        thruster_layout=layout,
        control_yaw=True,
    )


def build_v3_controller_pair(config, *, solver_time_limit_s: float | None = None):
    """Build the v3 SMC/MPC pair directly from one experiment config."""
    if config.controller.primary != "real10kg_smc_steady":
        raise ValueError(f"unsupported v3 primary controller: {config.controller.primary}")
    if config.controller.authority != "real10kg_mpc_ltv_v3":
        raise ValueError(f"unsupported v3 authority controller: {config.controller.authority}")
    layout = build_thruster_layout(
        config.environment.thruster_layout,
        max_force=config.actuator.max_force_n,
    )
    primary = build_real10kg_smc_steady(
        vehicle_profile=config.environment.vehicle_profile,
        thruster_layout=layout,
    )
    fallback = build_real10kg_smc_steady(
        vehicle_profile=config.environment.vehicle_profile,
        thruster_layout=layout,
    )
    dynamics = AUVDynamics(
        mass_scale_xy=config.environment.mass_scale_xy,
        damping_scale_xy=config.environment.damping_scale_xy,
        vehicle_profile=config.environment.vehicle_profile,
        dt=config.environment.dt,
    )
    time_limit_s = (
        config.mpc.time_limit_s
        if solver_time_limit_s is None else float(solver_time_limit_s)
    )
    if not np.isfinite(time_limit_s) or time_limit_s < config.mpc.time_limit_s:
        raise ValueError(
            "solver_time_limit_s must be finite and >= config.mpc.time_limit_s"
        )
    settings = MPCQPSettings(
        N=config.mpc.horizon,
        q_diag=config.mpc.q_diag,
        terminal_scale=config.mpc.terminal_scale,
        r_diag=config.mpc.r_diag,
        s_diag=config.mpc.s_diag,
        u_min=config.actuator.command_min,
        u_max=config.actuator.command_max,
        slew_limit=config.actuator.max_delta_per_step,
        eps_abs=config.mpc.eps_abs,
        eps_rel=config.mpc.eps_rel,
        max_iter=config.mpc.max_iter,
        time_limit_s=time_limit_s,
        accept_inaccurate_residual=config.mpc.accept_inaccurate_residual,
        max_consecutive_plan_reuse=config.mpc.max_consecutive_plan_reuse,
    )
    authority = MPCController(
        dynamics=dynamics,
        thruster_layout=layout,
        settings=settings,
        fallback_controller=fallback,
    )
    primary.set_trajectory3d(True)
    authority.set_trajectory3d(True)
    return primary, authority
