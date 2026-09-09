"""Formal sliding-surface controller used as the PAC primary expert."""

from __future__ import annotations

import numpy as np

from pac.simulation.thrusters import ThrusterLayout


class SMCController:
    """Equivalent-control SMC with the formal-v2 reaching law."""

    def __init__(
            self,
            *,
            lambda_gain,
            reaching_gain,
            robust_gain,
            equivalent_mass,
            equivalent_damping,
            equivalent_quadratic_damping,
            current_feedforward: float,
            control_yaw: bool,
            thruster_layout: ThrusterLayout,
            boundary_layer: float = 0.1):
        self.lam = np.array(lambda_gain, dtype=float)
        self.delta = float(boundary_layer)
        self.reaching_gain = np.array(reaching_gain, dtype=float)
        self.robust_gain = np.array(robust_gain, dtype=float)
        self.equivalent_mass = np.array(equivalent_mass, dtype=float)
        self.equivalent_damping = np.array(equivalent_damping, dtype=float)
        self.equivalent_quadratic_damping = np.array(
            equivalent_quadratic_damping,
            dtype=float,
        )
        self.current_feedforward = float(current_feedforward)
        self.control_yaw = bool(control_yaw)
        self.thruster_layout = thruster_layout
        self._trajectory3d = False

    def reset(self) -> None:
        """The formal SMC expert is stateless."""

    def set_trajectory3d(self, enabled: bool = True) -> None:
        self._trajectory3d = bool(enabled)

    @staticmethod
    def _wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _world_xy_to_body(xy, psi):
        cp, sp = np.cos(psi), np.sin(psi)
        return np.array([cp * xy[0] + sp * xy[1], -sp * xy[0] + cp * xy[1]])

    def _current_to_body(self, current_prediction, psi):
        current = np.asarray(current_prediction, dtype=float).reshape(-1)
        out = np.zeros(6, dtype=float)
        if current.size >= 2:
            out[:2] = self._world_xy_to_body(current[:2], psi)
        if current.size >= 3:
            out[2] = current[2]
        return self.current_feedforward * out

    @staticmethod
    def _target_kinematics(t):
        vx = 0.9 * np.cos(0.3 * t)
        vy = 0.9 * np.cos(0.6 * t)
        ax = -0.27 * np.sin(0.3 * t)
        ay = -0.54 * np.sin(0.6 * t)
        psi = np.arctan2(vy, vx)
        denom = max(vx * vx + vy * vy, 1e-9)
        yaw_rate = (vx * ay - vy * ax) / denom

        eps = 1e-4
        vx2 = 0.9 * np.cos(0.3 * (t + eps))
        vy2 = 0.9 * np.cos(0.6 * (t + eps))
        ax2 = -0.27 * np.sin(0.3 * (t + eps))
        ay2 = -0.54 * np.sin(0.6 * (t + eps))
        denom2 = max(vx2 * vx2 + vy2 * vy2, 1e-9)
        yaw_rate2 = (vx2 * ay2 - vy2 * ax2) / denom2
        yaw_acc = (yaw_rate2 - yaw_rate) / eps
        return (
            np.array([vx, vy], dtype=float),
            np.array([ax, ay], dtype=float),
            float(psi),
            float(yaw_rate),
            float(yaw_acc),
        )

    @staticmethod
    def _target_z_kinematics(t):
        return float(0.16 * np.cos(0.2 * t)), float(-0.032 * np.sin(0.2 * t))

    def compute(
            self,
            target_eta,
            current_eta,
            current_nu,
            *,
            t=None,
            current_prediction=None):
        t = float(t or 0.0)
        eta = np.asarray(current_eta, dtype=float)
        nu = np.asarray(current_nu, dtype=float)
        error = np.asarray(target_eta, dtype=float) - eta
        for index in (3, 4, 5):
            error[index] = self._wrap_angle(error[index])

        psi = float(eta[5])
        error_body_xy = self._world_xy_to_body(error[:2], psi)
        target_vel, target_acc, desired_yaw, yaw_rate, yaw_acc = self._target_kinematics(t)
        target_vel_body = self._world_xy_to_body(target_vel, psi)
        target_acc_body = self._world_xy_to_body(target_acc, psi)
        current_body = (
            self._current_to_body(current_prediction, psi)
            if current_prediction is not None
            else np.zeros(6, dtype=float)
        )

        vel_error_xy = target_vel_body - nu[:2]
        s_xy = vel_error_xy + self.lam[:2] * error_body_xy
        nu_ref_xy = target_vel_body + self.lam[:2] * error_body_xy
        nu_rel_ref_xy = nu_ref_xy - current_body[:2]
        reaching_xy = (
            self.reaching_gain[:2] * s_xy
            + self.robust_gain[:2] * np.clip(s_xy / self.delta, -1.0, 1.0)
        )

        tau = np.zeros(6, dtype=float)
        tau[:2] = (
            self.equivalent_mass[:2] * (target_acc_body + reaching_xy)
            + self.equivalent_damping[:2] * nu_rel_ref_xy
            + self.equivalent_quadratic_damping[:2]
            * np.abs(nu_rel_ref_xy)
            * nu_rel_ref_xy
        )

        if self._trajectory3d:
            target_z_vel, target_z_acc = self._target_z_kinematics(t)
            z_vel_error = target_z_vel - nu[2]
            s_z = z_vel_error + self.lam[2] * error[2]
            nu_ref_z = target_z_vel + self.lam[2] * error[2]
            nu_rel_ref_z = nu_ref_z - current_body[2]
            z_reaching = (
                self.reaching_gain[2] * s_z
                + self.robust_gain[2] * np.clip(s_z / self.delta, -1.0, 1.0)
            )
            tau[2] = (
                self.equivalent_mass[2] * (target_z_acc + z_reaching)
                + self.equivalent_damping[2] * nu_rel_ref_z
                + self.equivalent_quadratic_damping[2]
                * abs(nu_rel_ref_z)
                * nu_rel_ref_z
            )

        if self.control_yaw:
            yaw_error = self._wrap_angle(desired_yaw - psi)
            yaw_vel_error = yaw_rate - nu[5]
            s_yaw = yaw_vel_error + self.lam[5] * yaw_error
            yaw_reaching = (
                self.reaching_gain[5] * s_yaw
                + self.robust_gain[5] * np.clip(s_yaw / self.delta, -1.0, 1.0)
            )
            tau[5] = (
                self.equivalent_mass[5] * (yaw_acc + yaw_reaching)
                + self.equivalent_damping[5] * yaw_rate
            )

        return self.thruster_layout.allocate_wrench(tau)
