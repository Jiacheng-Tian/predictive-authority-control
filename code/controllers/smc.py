import numpy as np

from env.thrusters import build_thruster_layout


class SMCController:
    """Sliding mode controller for AUV trajectory tracking.

    The current simulation version uses a tuned equivalent-control SMC law:

        s = (nu_d - nu) + lambda * e
        tau = M * (a_d + k_s * s + k_r * sat(s / phi)) + D * nu_ref

    where e, nu_d, a_d, and nu_ref are expressed in body-frame xy
    coordinates.
    """

    def __init__(
            self,
            lambda_gain=(1.5, 1.5, 2.0, 2.0, 2.0, 2.0),
            k_gain=(35.0, 35.0, 35.0, 35.0, 35.0, 35.0),
            boundary_layer=0.1,
            max_thrust=(60.0, 60.0, 30.0, 10.0, 10.0, 20.0),
            reaching_gain=(24.0, 24.0, 0.0, 0.0, 0.0, 0.0),
            robust_gain=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            equivalent_mass=(50.0, 50.0, 30.0, 5.0, 5.0, 10.0),
            equivalent_damping=(20.0, 20.0, 10.0, 2.0, 2.0, 3.0),
            equivalent_quadratic_damping=None,
            current_feedforward=1.0,
            control_yaw=False,
            action_mode="wrench",
            thruster_layout=None):
        self.lam = np.array(lambda_gain, dtype=float)
        self.k = np.array(k_gain, dtype=float)
        self.delta = float(boundary_layer)
        self.max_thrust = np.array(max_thrust, dtype=float)
        self.reaching_gain = np.array(reaching_gain, dtype=float)
        self.robust_gain = np.array(robust_gain, dtype=float)
        self.equivalent_mass = np.array(equivalent_mass, dtype=float)
        self.equivalent_damping = np.array(equivalent_damping, dtype=float)
        self.equivalent_quadratic_damping = np.array(
            (
                equivalent_quadratic_damping
                if equivalent_quadratic_damping is not None
                else np.zeros(6, dtype=float)
            ),
            dtype=float,
        )
        self.current_feedforward = float(current_feedforward)
        self.control_yaw = bool(control_yaw)
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

    def reset(self):
        pass

    def set_trajectory3d(self, enabled=True):
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
        z_vel = 0.16 * np.cos(0.2 * t)
        z_acc = -0.032 * np.sin(0.2 * t)
        return float(z_vel), float(z_acc)

    def compute(
            self,
            target_eta,
            current_eta,
            current_nu,
            t=None,
            dt=0.01,
            current_prediction=None,
            update_state=True):
        del dt
        del update_state
        return self._compute_equivalent(
            target_eta,
            current_eta,
            current_nu,
            t=t,
            current_prediction=current_prediction,
        )

    def _wrench_to_action(self, tau):
        if self.action_mode == "thruster":
            return self.thruster_layout.allocate_wrench(tau)
        return np.clip(tau / self.max_thrust, -1.0, 1.0)

    def _compute_equivalent(
            self,
            target_eta,
            current_eta,
            current_nu,
            t=None,
            current_prediction=None):
        t = float(t or 0.0)
        eta = np.asarray(current_eta, dtype=float)
        nu = np.asarray(current_nu, dtype=float)
        error = np.asarray(target_eta, dtype=float) - eta
        for i in [3, 4, 5]:
            error[i] = self._wrap_angle(error[i])

        psi = float(eta[5])
        error_body_xy = self._world_xy_to_body(error[:2], psi)
        target_vel, target_acc, desired_yaw, yaw_rate, yaw_acc = (
            self._target_kinematics(t)
        )
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

        return self._wrench_to_action(tau)
