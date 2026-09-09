"""MPC controller used as the PAC authority expert."""

from __future__ import annotations

import numpy as np

from pac.simulation.thrusters import ThrusterLayout


class MPCController:
    """MPC tracking controller with bounded thruster allocation."""

    def __init__(
            self,
            *,
            M,
            D,
            D_quad,
            q_pos: float,
            q_att: float,
            q_nu: float,
            z_pos_gain: float,
            z_current_feedforward: float,
            r: float,
            thruster_layout: ThrusterLayout):
        self.M = np.asarray(M, dtype=float)
        self.D = np.asarray(D, dtype=float)
        self.D_quad = np.asarray(D_quad, dtype=float)
        self.q_pos = float(q_pos)
        self.q_att = float(q_att)
        self.q_nu = float(q_nu)
        self.z_pos_gain = float(z_pos_gain)
        self.z_current_feedforward = float(z_current_feedforward)
        self.r = float(r)
        self.thruster_layout = thruster_layout
        self._trajectory3d = False

    def reset(self) -> None:
        """The formal MPC expert is stateless."""

    def set_trajectory3d(self, enabled: bool = True) -> None:
        self._trajectory3d = bool(enabled)

    @staticmethod
    def _wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _world_xy_to_body(xy, psi):
        cp, sp = np.cos(psi), np.sin(psi)
        return np.array([cp * xy[0] + sp * xy[1], -sp * xy[0] + cp * xy[1]])

    def _target_velocity(self, t):
        vx = 0.9 * np.cos(0.3 * t)
        vy = 0.9 * np.cos(0.6 * t)
        return np.array([
            vx,
            vy,
            0.16 * np.cos(0.2 * t) if self._trajectory3d else 0.0,
            0.0,
            0.0,
            0.0,
        ])

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
            *,
            t=None,
            current_prediction=None):
        """Return the normalized six-thruster action in ``[-1, 1]``."""
        target = np.asarray(target_eta, dtype=float).copy()
        eta = np.asarray(current_eta, dtype=float)
        nu = np.asarray(current_nu, dtype=float)
        err = target - eta
        err[3:6] = [self._wrap_angle(value) for value in err[3:6]]
        pos_err_body = self._world_xy_to_body(err[:2], eta[5])
        target_vel_world = self._target_velocity(float(t or 0.0))
        target_vel_body = np.zeros(6, dtype=float)
        target_vel_body[:2] = self._world_xy_to_body(target_vel_world[:2], eta[5])
        if self._trajectory3d:
            target_vel_body[2] = target_vel_world[2]

        desired_nu = target_vel_body.copy()
        desired_nu[0] += 0.055 * self.q_pos * pos_err_body[0]
        desired_nu[1] += 0.055 * self.q_pos * pos_err_body[1]
        desired_nu[2] += self.z_pos_gain * self.q_pos * err[2]
        desired_nu[5] += 0.08 * self.q_att * err[5]

        vel_err = desired_nu - nu
        damping_nu = desired_nu.copy()
        if current_prediction is not None:
            damping_nu -= self._current_to_body(current_prediction, eta[5])
        tau = self.M * (0.08 * self.q_nu * vel_err) + self.D * damping_nu
        tau += self.D_quad * np.abs(damping_nu) * damping_nu
        tau = tau / (1.0 + max(self.r, 1e-8) * 1000.0)
        action = self.thruster_layout.allocate_wrench(tau)
        return np.clip(action, -1.0, 1.0)
