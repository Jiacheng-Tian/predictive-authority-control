import numpy as np

from .vehicle_profiles import get_vehicle_profile


class AUVDynamics:
    """
    6-DOF AUV dynamics for the current 10 kg experimental vehicle.

    State: eta=[x,y,z,phi,theta,psi], nu=[u,v,w,p,q,r]
    """

    def __init__(
            self,
            mass_scale_xy=1.0,
            damping_scale_xy=1.0,
            vehicle_profile="real_10kg_v1"):
        self.dt = 0.01
        self.profile = get_vehicle_profile(vehicle_profile)

        mass = np.array(self.profile.effective_mass, dtype=float)
        mass[:2] *= float(mass_scale_xy)
        self.M = np.diag(mass)
        self.invM = np.linalg.inv(self.M)

        damping = np.array(self.profile.linear_damping, dtype=float)
        damping[:2] *= float(damping_scale_xy)
        self.D = np.diag(damping)

        quadratic_damping = np.array(self.profile.quadratic_damping, dtype=float)
        quadratic_damping[:2] *= float(damping_scale_xy)
        self.D_quad = np.diag(quadratic_damping)

        self.K_hyd = np.array(self.profile.restoring_stiffness, dtype=float)

    def reset(self):
        self.eta = np.zeros(6)
        self.nu = np.zeros(6)

    def _J(self, eta):
        """6-DOF Jacobian using Euler-angle kinematics."""
        phi, theta, psi = eta[3], eta[4], eta[5]
        cp, sp = np.cos(psi), np.sin(psi)
        ct, st = np.cos(theta), np.sin(theta)
        cf, sf = np.cos(phi), np.sin(phi)

        R = np.array([
            [cp * ct, cp * st * sf - sp * cf, cp * st * cf + sp * sf],
            [sp * ct, sp * st * sf + cp * cf, sp * st * cf - cp * sf],
            [-st, ct * sf, ct * cf],
        ])
        ct_safe = np.clip(ct, 1e-6, None)
        T = np.array([
            [1, sf * st / ct_safe, cf * st / ct_safe],
            [0, cf, -sf],
            [0, sf / ct_safe, cf / ct_safe],
        ])
        J = np.zeros((6, 6))
        J[:3, :3] = R
        J[3:, 3:] = T
        return J

    def _C(self, nu):
        """Simplified Coriolis matrix for the diagonal effective mass model."""
        # Formal-v2 compatibility: this archived coupling is intentionally kept
        # in the numerical path. It is not fully skew-symmetric; correcting it
        # requires new checkpoints, metrics, and figures.
        m11, m22, m33 = self.M[0, 0], self.M[1, 1], self.M[2, 2]
        m44, m55, m66 = self.M[3, 3], self.M[4, 4], self.M[5, 5]
        u, v, w, p, q, r = nu
        del m66, r
        C = np.zeros((6, 6))
        C[0, 4] = -m33 * w
        C[0, 5] = m22 * v
        C[1, 3] = m33 * w
        C[1, 5] = -m11 * u
        C[2, 3] = -m22 * v
        C[2, 4] = m11 * u
        C[3, 1] = -m33 * w
        C[3, 2] = m22 * v
        C[3, 4] = -m55 * q
        C[3, 5] = m44 * p
        C[4, 0] = m33 * w
        C[4, 2] = -m11 * u
        C[4, 3] = m55 * q
        C[4, 5] = -m44 * p
        C[5, 0] = -m22 * v
        C[5, 1] = m11 * u
        C[5, 3] = -m44 * p
        C[5, 4] = m55 * q
        return C

    def _derivatives(self, nu, eta, tau, v_current):
        psi = eta[5]
        current = np.asarray(v_current, dtype=float).reshape(-1)
        if current.size < 2:
            raise ValueError("v_current must contain at least surge/sway components")

        cp, sp = np.cos(-psi), np.sin(-psi)
        vc_body = np.zeros(6)
        vc_body[0] = cp * current[0] - sp * current[1]
        vc_body[1] = sp * current[0] + cp * current[1]
        if current.size >= 3:
            vc_body[2] = current[2]

        nu_rel = nu - vc_body
        damping = self.D @ nu_rel + self.D_quad @ (np.abs(nu_rel) * nu_rel)
        restoring = self.K_hyd * eta
        nu_dot = self.invM @ (tau - damping - self._C(nu) @ nu - restoring)
        eta_dot = self._J(eta) @ nu
        return nu_dot, eta_dot

    def step(self, tau, v_current=None):
        if v_current is None:
            v_current = np.zeros(3)
        h = self.dt
        nu, eta = self.nu.copy(), self.eta.copy()

        k1n, k1e = self._derivatives(nu, eta, tau, v_current)
        k2n, k2e = self._derivatives(nu + h / 2 * k1n, eta + h / 2 * k1e, tau, v_current)
        k3n, k3e = self._derivatives(nu + h / 2 * k2n, eta + h / 2 * k2e, tau, v_current)
        k4n, k4e = self._derivatives(nu + h * k3n, eta + h * k3e, tau, v_current)

        self.nu = nu + h / 6 * (k1n + 2 * k2n + 2 * k3n + k4n)
        self.eta = eta + h / 6 * (k1e + 2 * k2e + 2 * k3e + k4e)

        for i in [3, 4, 5]:
            self.eta[i] = (self.eta[i] + np.pi) % (2 * np.pi) - np.pi

        return self.eta.copy(), self.nu.copy()
