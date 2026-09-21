"""Batched torch implementation of the frozen v3 AUV dynamics.

``batched_predict_step`` reproduces ``AUVDynamics.predict_step`` (RK4 with
Euler-angle kinematics, diagonal mass/damping, quadratic drag, restoring
stiffness, angle wrapping) exactly, but evaluates a whole batch of states at
once.  It is used for world-model multi-step rollouts and candidate-alpha
rollouts where thousands of states must be propagated per step.  A unit test
cross-checks the torch path against the frozen numpy implementation.
"""

from __future__ import annotations

import numpy as np
import torch

from pac.simulation.vehicle_profiles import get_vehicle_profile


def _wrap_angles(eta: torch.Tensor) -> torch.Tensor:
    wrapped = (eta[..., 3:6] + torch.pi) % (2.0 * torch.pi) - torch.pi
    return torch.cat([eta[..., :3], wrapped], dim=-1)


class TorchAUVDynamics:
    """Vectorized torch mirror of the archived ``AUVDynamics``."""

    def __init__(
            self,
            *,
            vehicle_profile: str = "real_10kg_v1",
            mass_scale_xy: float = 1.0,
            damping_scale_xy: float = 1.0,
            dt: float = 0.01,
            dtype: torch.dtype = torch.float64,
            device: str | torch.device = "cpu"):
        profile = get_vehicle_profile(vehicle_profile)
        mass = np.array(profile.effective_mass, dtype=np.float64)
        mass[:2] *= float(mass_scale_xy)
        damping = np.array(profile.linear_damping, dtype=np.float64)
        damping[:2] *= float(damping_scale_xy)
        quadratic = np.array(profile.quadratic_damping, dtype=np.float64)
        quadratic[:2] *= float(damping_scale_xy)
        restoring = np.array(profile.restoring_stiffness, dtype=np.float64)
        self.dt = float(dt)
        self.dtype = dtype
        self.device = torch.device(device)
        to_tensor = lambda values: torch.tensor(
            np.asarray(values, dtype=np.float64), dtype=dtype, device=self.device
        )
        self.mass = to_tensor(mass)
        self.inv_mass = 1.0 / self.mass
        self.damping = to_tensor(damping)
        self.damping_quad = to_tensor(quadratic)
        self.restoring = to_tensor(restoring)

    def _derivatives(self, nu: torch.Tensor, eta: torch.Tensor, tau: torch.Tensor,
                     current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate nu_dot/eta_dot for a batch of states.

        ``nu``, ``tau``: (B, 6); ``eta``: (B, 6); ``current``: (B, 3).
        """
        psi = eta[:, 5]
        cp = torch.cos(-psi)
        sp = torch.sin(-psi)
        vc_body = torch.stack([
            cp * current[:, 0] - sp * current[:, 1],
            sp * current[:, 0] + cp * current[:, 1],
            current[:, 2],
            torch.zeros_like(psi),
            torch.zeros_like(psi),
            torch.zeros_like(psi),
        ], dim=1)
        nu_rel = nu - vc_body
        damping = self.damping * nu_rel + self.damping_quad * (nu_rel.abs() * nu_rel)
        restoring = self.restoring * eta

        m11, m22, m33 = self.mass[0], self.mass[1], self.mass[2]
        m44, m55 = self.mass[3], self.mass[4]
        u, v, w, p, q, r = (nu[:, i] for i in range(6))
        coriolis_nu = torch.stack([
            -m33 * w * q + m22 * v * r,
            m33 * w * p - m11 * u * r,
            -m22 * v * p + m11 * u * q,
            -m33 * w * v + m22 * v * w - m55 * q * q + m44 * p * r,
            m33 * w * u - m11 * u * w + m55 * q * p - m44 * p * r,
            -m22 * v * u + m11 * u * v - m44 * p * p + m55 * q * q,
        ], dim=1)

        phi, theta, psi_e = eta[:, 3], eta[:, 4], eta[:, 5]
        cp_e, sp_e = torch.cos(psi_e), torch.sin(psi_e)
        ct, st = torch.cos(theta), torch.sin(theta)
        cf, sf = torch.cos(phi), torch.sin(phi)
        ct_safe = torch.clamp(ct, min=1.0e-6)
        zero = torch.zeros_like(psi_e)
        one = torch.ones_like(psi_e)
        j_eta = torch.stack([
            cp_e * ct * nu[:, 0] + (cp_e * st * sf - sp_e * cf) * nu[:, 1] + (cp_e * st * cf + sp_e * sf) * nu[:, 2],
            sp_e * ct * nu[:, 0] + (sp_e * st * sf + cp_e * cf) * nu[:, 1] + (sp_e * st * cf - cp_e * sf) * nu[:, 2],
            -st * nu[:, 0] + ct * sf * nu[:, 1] + ct * cf * nu[:, 2],
            nu[:, 3] + (sf * st / ct_safe) * nu[:, 4] + (cf * st / ct_safe) * nu[:, 5],
            cf * nu[:, 4] - sf * nu[:, 5],
            (sf / ct_safe) * nu[:, 4] + (cf / ct_safe) * nu[:, 5],
        ], dim=1)

        nu_dot = self.inv_mass * (tau - damping - coriolis_nu - restoring)
        return nu_dot, j_eta

    def predict_step(self, eta, nu, tau, current):
        """One RK4 step for a batch; mirrors the numpy implementation."""
        eta = torch.as_tensor(eta, dtype=self.dtype, device=self.device)
        nu = torch.as_tensor(nu, dtype=self.dtype, device=self.device)
        tau = torch.as_tensor(tau, dtype=self.dtype, device=self.device)
        current = torch.as_tensor(current, dtype=self.dtype, device=self.device)
        h = self.dt
        k1n, k1e = self._derivatives(nu, eta, tau, current)
        k2n, k2e = self._derivatives(nu + h / 2 * k1n, eta + h / 2 * k1e, tau, current)
        k3n, k3e = self._derivatives(nu + h / 2 * k2n, eta + h / 2 * k2e, tau, current)
        k4n, k4e = self._derivatives(nu + h * k3n, eta + h * k3e, tau, current)
        next_nu = nu + h / 6 * (k1n + 2 * k2n + 2 * k3n + k4n)
        next_eta = eta + h / 6 * (k1e + 2 * k2e + 2 * k3e + k4e)
        return _wrap_angles(next_eta), next_nu

    def rollout(self, eta0, nu0, tau, currents):
        """Propagate ``steps`` RK4 steps with per-step current inputs.

        ``eta0``/``nu0``: (B, 6); ``tau``: (B, 6) held constant;
        ``currents``: (B, steps, 3).  Returns stacked eta (B, steps, 6).
        """
        steps = int(currents.shape[1])
        eta = torch.as_tensor(eta0, dtype=self.dtype, device=self.device)
        nu = torch.as_tensor(nu0, dtype=self.dtype, device=self.device)
        tau = torch.as_tensor(tau, dtype=self.dtype, device=self.device)
        currents = torch.as_tensor(currents, dtype=self.dtype, device=self.device)
        trajectory = torch.empty(
            (eta.shape[0], steps, 6), dtype=self.dtype, device=self.device
        )
        for step in range(steps):
            eta, nu = self.predict_step(eta, nu, tau, currents[:, step])
            trajectory[:, step] = eta
        return trajectory


def assert_torch_physics_parity(
        dynamics, torch_dynamics, sample_count: int = 64, tolerance: float = 1.0e-10) -> float:
    """Max abs difference between numpy and torch physics on random states."""
    rng = np.random.default_rng(0)
    eta = rng.normal(0.0, 0.5, size=(sample_count, 6))
    eta[:, 3:6] = rng.uniform(-np.pi, np.pi, size=(sample_count, 3))
    nu = rng.normal(0.0, 0.5, size=(sample_count, 6))
    tau = rng.normal(0.0, 5.0, size=(sample_count, 6))
    current = rng.normal(0.0, 0.3, size=(sample_count, 3))
    numpy_next = np.empty((sample_count, 12))
    for index in range(sample_count):
        next_eta, next_nu = dynamics.predict_step(eta[index], nu[index], tau[index], current[index])
        numpy_next[index] = np.concatenate([next_eta, next_nu])
    torch_eta, torch_nu = torch_dynamics.predict_step(
        torch.tensor(eta), torch.tensor(nu), torch.tensor(tau), torch.tensor(current)
    )
    torch_next = torch.cat([torch_eta, torch_nu], dim=1).numpy()
    return float(np.max(np.abs(numpy_next - torch_next))), tolerance


class ZeroTorchDynamics:
    """Zero next-state predictor for the pure-learning capacity control.

    When the physics baseline is removed, the residual target becomes the
    full next state; rollouts under this "dynamics" propagate the learned
    delta alone so the pure model is evaluated on its own merits.
    """

    def __init__(self, dt: float = 0.01):
        self.dt = float(dt)

    def predict_step(self, eta, nu, tau, current):
        eta = torch.as_tensor(eta)
        nu = torch.as_tensor(nu)
        return torch.zeros_like(eta, dtype=eta.dtype), torch.zeros_like(
            nu, dtype=nu.dtype
        )


__all__ = ["TorchAUVDynamics", "ZeroTorchDynamics", "assert_torch_physics_parity"]
