"""Reward v1 for residual RL (frozen weights from the residual config).

Single-step reward following the frozen normalized weighted form:

    r_t = -w_e e_pos/s_pos - w_h e_head/s_head - w_u ||u||^2
          - w_du ||du||^2 - w_da |dalpha| - w_sat rho_sat
          - w_dl rho_dl - w_c rho_con

All weights and normalization scales are read from :class:`RewardConfig`
and frozen before training; they are recorded in every checkpoint
manifest.  The deadline term is telemetry-only: RL never changes MPC
scheduling.
"""

from __future__ import annotations

import numpy as np


def compute_step_reward(
        reward_config,
        *,
        position_error: float,
        heading_error: float,
        applied_action,
        previous_applied,
        alpha: float,
        previous_alpha: float,
        saturation_fraction: float,
        deadline_missed: bool,
        constraint_violated: bool,
        include_delta_alpha: bool = True) -> float:
    """Evaluate the frozen single-step reward for one transition.

    ``include_delta_alpha=False`` serves the direct-RL control arm, which
    commands thrusters directly and has no authority coefficient, so its
    reward drops the ``w_delta_alpha`` term.
    """
    applied = np.asarray(applied_action, dtype=float).reshape(-1)
    previous = np.asarray(previous_applied, dtype=float).reshape(-1)
    if applied.shape != previous.shape:
        raise ValueError("applied and previous actions must have equal shape")
    e_pos = float(position_error)
    e_head = abs(float(heading_error))
    control = float(np.mean(applied ** 2))
    delta_control = float(np.mean((applied - previous) ** 2))
    reward = (
        -reward_config.w_position * (e_pos / reward_config.position_scale_m)
        - reward_config.w_heading * (e_head / reward_config.heading_scale_rad)
        -reward_config.w_control * control
        -reward_config.w_delta_control * delta_control
        -reward_config.w_saturation * float(np.clip(saturation_fraction, 0.0, 1.0))
        -reward_config.w_deadline * float(bool(deadline_missed))
        -reward_config.w_constraint * float(bool(constraint_violated))
    )
    if include_delta_alpha:
        delta_alpha = abs(float(alpha) - float(previous_alpha))
        reward -= reward_config.w_delta_alpha * delta_alpha
    w_center = float(getattr(reward_config, "w_center", 0.0))
    center_gate = float(getattr(reward_config, "center_gate_error_m", 0.0))
    if w_center > 0.0 and e_pos > center_gate:
        alpha_center = float(getattr(reward_config, "alpha_center", 0.5))
        reward -= w_center * (float(alpha) - alpha_center) ** 2
    return float(reward)


__all__ = ["compute_step_reward"]
