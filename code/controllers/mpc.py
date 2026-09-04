"""Lightweight MPC-style controller for 2D/3D AUV tracking.`r`n`r`nThe controller keeps the public interface used by the current reproducibility`r`npipeline. Internally it uses a damped one-step quadratic tracking law with`r`ninput limits and warm-start smoothing.`r`n"""
from __future__ import annotations

import numpy as np

from env.thrusters import build_thruster_layout


class MPCController:
    """Nominal constrained tracking controller with MPC-compatible hooks."""

    def __init__(
            self,
            horizon: int = 10,
            M=None,
            D=None,
            D_quad=None,
            max_thrust=None,
            q_pos: float = 100.0,
            q_att: float = 5.0,
            q_nu: float = 2.0,
            z_pos_gain: float = 0.03,
            z_current_feedforward: float = 1.0,
            r: float = 1e-4,
            smooth_alpha: float = 0.8,
            action_mode: str = "wrench",
            thruster_layout=None):
        self.horizon = int(horizon)
        self.M = np.asarray(
            M if M is not None else [50.0, 50.0, 30.0, 5.0, 5.0, 10.0],
            dtype=float,
        )
        if self.M.ndim == 2:
            self.M = np.diag(self.M)
        self.D = np.asarray(
            D if D is not None else [20.0, 20.0, 10.0, 2.0, 2.0, 3.0],
            dtype=float,
        )
        if self.D.ndim == 2:
            self.D = np.diag(self.D)
        self.D_quad = np.asarray(
            D_quad if D_quad is not None else np.zeros(6, dtype=float),
            dtype=float,
        )
        if self.D_quad.ndim == 2:
            self.D_quad = np.diag(self.D_quad)
        self.max_thrust = np.asarray(
            max_thrust if max_thrust is not None
            else [60.0, 60.0, 30.0, 10.0, 10.0, 20.0],
            dtype=float,
        )
        self.q_pos = float(q_pos)
        self.q_att = float(q_att)
        self.q_nu = float(q_nu)
        self.z_pos_gain = float(z_pos_gain)
        self.z_current_feedforward = float(z_current_feedforward)
        self.r = float(r)
        self.smooth_alpha = float(smooth_alpha)
        self.action_mode = str(action_mode or "wrench")
        self.thruster_layout = (
            thruster_layout
            if thruster_layout is not None
            else (
                build_thruster_layout()
                if self.action_mode == "thruster"
                else None
            )
        )
        self._trajectory3d = False
        self._prev_action = np.zeros(6, dtype=float)
        self._last_raw_solution = np.zeros(6, dtype=float)

    def reset(self):
        self._prev_action = np.zeros(6, dtype=float)
        self._last_raw_solution = np.zeros(6, dtype=float)

    def set_trajectory3d(self, enabled=True):
        self._trajectory3d = bool(enabled)

    @staticmethod
    def _wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _target_velocity(self, t):
        vx = 0.9 * np.cos(0.3 * t)
        vy = 0.9 * np.cos(0.6 * t)
        if self._trajectory3d:
            return np.array([
                vx,
                vy,
                0.16 * np.cos(0.2 * t),
                0.0,
                0.0,
                0.0,
            ])
        return np.array([
            vx,
            vy,
            0.0,
            0.0,
            0.0,
            0.0,
        ])

    def _world_xy_to_body(self, xy, psi):
        cp, sp = np.cos(psi), np.sin(psi)
        return np.array([cp * xy[0] + sp * xy[1], -sp * xy[0] + cp * xy[1]])

    def _current_to_body(self, current_prediction, psi):
        current = np.asarray(current_prediction, dtype=float).reshape(-1)
        out = np.zeros(6, dtype=float)
        if current.size >= 2:
            out[:2] = self._world_xy_to_body(current[:2], psi)
        if current.size >= 3:
            out[2] = self.z_current_feedforward * current[2]
        return out

    def compute(
            self,
            target_eta,
            current_eta,
            current_nu,
            t=None,
            dt=0.01,
            reference_offset=None,
            parameter_scales=None,
            current_prediction=None,
            smooth_alpha=None,
            update_state=True):
        """Return normalized action in [-1, 1]."""
        del dt  # The current law is one-step; dt is kept for API compatibility.
        target = np.asarray(target_eta, dtype=float).copy()
        eta = np.asarray(current_eta, dtype=float)
        nu = np.asarray(current_nu, dtype=float)
        if reference_offset is not None:
            ref = np.asarray(reference_offset, dtype=float)
            target[:min(6, ref.size)] += ref[:min(6, ref.size)]

        err = target - eta
        err[3:6] = [self._wrap_angle(a) for a in err[3:6]]
        pos_err_body = self._world_xy_to_body(err[:2], eta[5])
        target_vel_world = self._target_velocity(float(t or 0.0))
        target_vel_body = np.zeros(6, dtype=float)
        target_vel_body[:2] = self._world_xy_to_body(target_vel_world[:2], eta[5])
        if self._trajectory3d:
            target_vel_body[2] = target_vel_world[2]

        scales = parameter_scales or {}
        q_pos = self.q_pos * float(scales.get("q_pos", 1.0))
        q_nu = self.q_nu * float(scales.get("q_nu", 1.0))
        q_att = self.q_att * float(scales.get("q_att", 1.0))
        r = max(self.r * float(scales.get("r", 1.0)), 1e-8)

        desired_nu = target_vel_body.copy()
        desired_nu[0] += 0.055 * q_pos * pos_err_body[0]
        desired_nu[1] += 0.055 * q_pos * pos_err_body[1]
        desired_nu[2] += self.z_pos_gain * q_pos * err[2]
        desired_nu[5] += 0.08 * q_att * err[5]

        vel_err = desired_nu - nu
        damping_nu = desired_nu.copy()
        if current_prediction is not None:
            damping_nu -= self._current_to_body(current_prediction, eta[5])
        tau = self.M * (0.08 * q_nu * vel_err) + self.D * damping_nu
        tau += self.D_quad * np.abs(damping_nu) * damping_nu
        tau = tau / (1.0 + r * 1000.0)
        if self.action_mode == "thruster":
            raw = self.thruster_layout.allocate_wrench(tau)
        else:
            raw = np.clip(tau / self.max_thrust, -1.0, 1.0)
        alpha = self.smooth_alpha if smooth_alpha is None else float(smooth_alpha)
        alpha = float(np.clip(alpha, 0.0, 0.99))
        action = alpha * self._prev_action + (1.0 - alpha) * raw
        action = np.clip(action, -1.0, 1.0)
        self._last_raw_solution = action.copy()
        if update_state:
            self._prev_action = action.copy()
        return action

    def commit_last_raw_solution(self):
        self._prev_action = np.asarray(self._last_raw_solution, dtype=float).copy()

    def commit_normalized_action(self, action):
        self._prev_action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)


