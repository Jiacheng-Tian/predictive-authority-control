"""Batched SMC and actuator primitives for fast candidate rollouts.

``SMCController.compute`` is a pure closed-form function of
``(target, eta, nu, t, current_prediction)`` with no internal state, so a
whole batch of candidate states can be evaluated with one vectorized call.
The batched implementation replicates the scalar path operation for
operation (including the yaw-acceleration finite difference and the
pinv-based thruster allocation) and is pinned by an exact parity test.
"""

from __future__ import annotations

import numpy as np


def _wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class BatchedSMC:
    """Vectorized mirror of ``SMCController.compute`` over B states."""

    def __init__(self, controller):
        self.lam = np.asarray(controller.lam, dtype=float).copy()
        self.delta = float(controller.delta)
        self.reaching = np.asarray(controller.reaching_gain, dtype=float).copy()
        self.robust = np.asarray(controller.robust_gain, dtype=float).copy()
        self.mass = np.asarray(controller.equivalent_mass, dtype=float).copy()
        self.damping = np.asarray(controller.equivalent_damping, dtype=float).copy()
        self.quadratic = np.asarray(
            controller.equivalent_quadratic_damping, dtype=float
        ).copy()
        self.current_feedforward = float(controller.current_feedforward)
        self.control_yaw = bool(controller.control_yaw)
        self.trajectory3d = bool(controller._trajectory3d)
        allocation = np.asarray(
            controller.thruster_layout.allocation_matrix, dtype=float
        )
        self._allocation_pinv = np.linalg.pinv(allocation)
        self._max_force = float(controller.thruster_layout.max_force)

    @staticmethod
    def _target_kinematics(t: float) -> dict[str, float]:
        """Exact copy of ``SMCController._target_kinematics`` for scalar t."""
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
        return {
            "vx": float(vx), "vy": float(vy),
            "ax": float(ax), "ay": float(ay),
            "psi": float(psi),
            "yaw_rate": float(yaw_rate),
            "yaw_acc": float(yaw_acc),
        }

    def compute_batch(
            self,
            target,
            eta,
            nu,
            t: float,
            current=None) -> np.ndarray:
        """Normalized thruster commands ``(B, 6)`` for a batch of states."""
        eta = np.atleast_2d(np.asarray(eta, dtype=float))
        nu = np.atleast_2d(np.asarray(nu, dtype=float))
        target = np.asarray(target, dtype=float).reshape(6)
        if eta.shape != nu.shape or eta.shape[1] != 6:
            raise ValueError("eta and nu must have shape (B, 6)")
        stamp = float(t or 0.0)
        batch = eta.shape[0]

        error = target[None, :] - eta
        error[:, 3:6] = _wrap(error[:, 3:6])

        psi = eta[:, 5]
        cp, sp = np.cos(psi), np.sin(psi)

        def world_to_body(xy: np.ndarray) -> np.ndarray:
            return np.stack(
                [cp * xy[:, 0] + sp * xy[:, 1], -sp * xy[:, 0] + cp * xy[:, 1]],
                axis=1,
            )

        error_body_xy = world_to_body(error[:, :2])
        kinematics = self._target_kinematics(stamp)
        target_vel = world_to_body(
            np.broadcast_to(
                np.array([kinematics["vx"], kinematics["vy"]]), (batch, 2)
            )
        )
        target_acc = world_to_body(
            np.broadcast_to(
                np.array([kinematics["ax"], kinematics["ay"]]), (batch, 2)
            )
        )
        if current is not None:
            current_array = np.asarray(current, dtype=float).reshape(-1, 3)
            if current_array.shape[0] != batch:
                raise ValueError("current must have shape (B, 3)")
            current_body = np.zeros((batch, 3))
            current_body[:, 0] = self.current_feedforward * (
                cp * current_array[:, 0] + sp * current_array[:, 1]
            )
            current_body[:, 1] = self.current_feedforward * (
                -sp * current_array[:, 0] + cp * current_array[:, 1]
            )
            current_body[:, 2] = self.current_feedforward * current_array[:, 2]
        else:
            current_body = np.zeros((batch, 3))

        vel_error_xy = target_vel - nu[:, :2]
        s_xy = vel_error_xy + self.lam[:2] * error_body_xy
        nu_ref_xy = target_vel + self.lam[:2] * error_body_xy
        nu_rel_xy = nu_ref_xy - current_body[:, :2]
        reaching_xy = self.reaching[:2] * s_xy + self.robust[:2] * np.clip(
            s_xy / self.delta, -1.0, 1.0
        )
        tau = np.zeros((batch, 6), dtype=float)
        tau[:, :2] = (
            self.mass[:2] * (target_acc + reaching_xy)
            + self.damping[:2] * nu_rel_xy
            + self.quadratic[:2] * np.abs(nu_rel_xy) * nu_rel_xy
        )

        if self.trajectory3d:
            target_z_vel = 0.16 * np.cos(0.2 * stamp)
            target_z_acc = -0.032 * np.sin(0.2 * stamp)
            z_vel_error = target_z_vel - nu[:, 2]
            s_z = z_vel_error + self.lam[2] * error[:, 2]
            nu_ref_z = target_z_vel + self.lam[2] * error[:, 2]
            nu_rel_z = nu_ref_z - current_body[:, 2]
            z_reaching = self.reaching[2] * s_z + self.robust[2] * np.clip(
                s_z / self.delta, -1.0, 1.0
            )
            tau[:, 2] = (
                self.mass[2] * (target_z_acc + z_reaching)
                + self.damping[2] * nu_rel_z
                + self.quadratic[2] * np.abs(nu_rel_z) * nu_rel_z
            )

        if self.control_yaw:
            yaw_error = _wrap(kinematics["psi"] - psi)
            yaw_vel_error = kinematics["yaw_rate"] - nu[:, 5]
            s_yaw = yaw_vel_error + self.lam[5] * yaw_error
            yaw_reaching = self.reaching[5] * s_yaw + self.robust[5] * np.clip(
                s_yaw / self.delta, -1.0, 1.0
            )
            tau[:, 5] = (
                self.mass[5] * (kinematics["yaw_acc"] + yaw_reaching)
                + self.damping[5] * kinematics["yaw_rate"]
            )

        forces = np.clip(
            tau @ self._allocation_pinv.T,
            -self._max_force,
            self._max_force,
        )
        return forces / self._max_force


def batched_actuator_apply(
        requested: np.ndarray,
        previous_applied: np.ndarray,
        *,
        command_min: float,
        command_max: float,
        max_delta_per_step: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized mirror of ``SharedActuator.apply`` amplitude/slew limits.

    Returns ``(applied, amplitude_clipped)``, each ``(B, 6)``.
    """
    requested = np.atleast_2d(np.asarray(requested, dtype=float))
    previous = np.atleast_2d(np.asarray(previous_applied, dtype=float))
    if requested.shape != previous.shape or requested.shape[1] != 6:
        raise ValueError("requested and previous_applied must have shape (B, 6)")
    amplitude_clipped = np.clip(requested, command_min, command_max)
    if max_delta_per_step is None:
        applied = amplitude_clipped.copy()
    else:
        delta = amplitude_clipped - previous
        applied = previous + np.clip(
            delta, -float(max_delta_per_step), float(max_delta_per_step)
        )
    return applied, amplitude_clipped


def batched_blend(
        primary: np.ndarray,
        authority: np.ndarray,
        alphas: np.ndarray) -> np.ndarray:
    """Blend actions for per-candidate alpha values, then clip to [-1, 1].

    ``primary``/``authority``: ``(B, 6)``; ``alphas``: ``(B,)``.
    """
    primary = np.atleast_2d(np.asarray(primary, dtype=float))
    authority = np.atleast_2d(np.asarray(authority, dtype=float))
    alphas = np.asarray(alphas, dtype=float).reshape(-1)
    if alphas.shape[0] != primary.shape[0]:
        raise ValueError("alphas must have shape (B,)")
    blended = (1.0 - alphas)[:, None] * primary + alphas[:, None] * authority
    return np.clip(blended, -1.0, 1.0)


__all__ = [
    "BatchedSMC",
    "batched_actuator_apply",
    "batched_blend",
]
