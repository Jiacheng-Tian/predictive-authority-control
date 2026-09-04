import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .dynamics import AUVDynamics
from .thrusters import build_thruster_layout


DEFAULT_REWARD_CONFIG = {
    "pos_weight": 1.0,
    "vel_weight": 0.5,
    "smooth_weight": 0.02,
    "energy_weight": 0.005,
    "att_weight": 0.3,
    "heading_weight": 0.0,
    "bias": 0.0,
    "z_progress_weight": 0.0,
}


def compute_tracking_reward(
        pos_err,
        vel_err_norm,
        smoothness,
        energy,
        att_err,
        xy_err=None,
        z_err=None,
        xy_vel_err=None,
        z_vel_err=None,
        heading_err=None,
        reward_config=None):
    """Tracking-dominant reward with weak effort regularization."""
    cfg = dict(DEFAULT_REWARD_CONFIG)
    cfg.update(reward_config or {})
    xy_weight = cfg.get("xy_weight")
    z_weight = cfg.get("z_weight")
    use_axis_position = (
        xy_weight is not None
        and z_weight is not None
        and xy_err is not None
        and z_err is not None
    )
    if use_axis_position:
        position_penalty = (
            float(xy_weight) * float(abs(xy_err))
            + float(z_weight) * float(abs(z_err))
        )
    else:
        position_penalty = float(pos_err)
    xy_vel_weight = cfg.get("xy_vel_weight")
    z_vel_weight = cfg.get("z_vel_weight")
    use_axis_velocity = (
        xy_vel_weight is not None
        and z_vel_weight is not None
        and xy_vel_err is not None
        and z_vel_err is not None
    )
    if use_axis_velocity:
        velocity_penalty = (
            float(xy_vel_weight) * float(abs(xy_vel_err))
            + float(z_vel_weight) * float(abs(z_vel_err))
        )
    else:
        velocity_penalty = float(vel_err_norm)
    return float(
        -float(cfg.get("pos_weight", 1.0)) * position_penalty
        -float(cfg.get("vel_weight", 0.5)) * velocity_penalty
        -float(cfg.get("smooth_weight", 0.02)) * float(smoothness)
        -float(cfg.get("energy_weight", 0.005)) * float(energy)
        -float(cfg.get("att_weight", 0.3)) * float(att_err)
        -float(cfg.get("heading_weight", 0.0)) * float(abs(heading_err or 0.0))
        +float(cfg.get("bias", 0.0))
    )


class AUVTrackingEnv(gym.Env):
    """
    6-DOF AUV trajectory tracking environment.

    The current simulation setting uses a 3D Lissajous trajectory, a real_10kg_v1
    vehicle profile, and a six-thruster X layout.

    Observation:
        [ex, ey, ez, ephi, etheta, epsi,
         u, v, w, p, q, r,
         dtx, dty, dtz, sin(t*0.1), cos(t*0.1),
         v_current_norm, vx_current, vy_current]
    Action (6-dim): normalized wrench or normalized thruster command in [-1, 1].
    """

    metadata = {"render.modes": []}

    SCENARIO_MAP = {
        "constant": 1,
        "sinusoidal": 2,
        "step_change": 3,
        "turbulent": 4,
        "spatially_varying": 5,
    }

    def __init__(self, scenario=0, max_steps=2090, trajectory3d=False,
                 start_time=0.0, reward_config=None,
                 current_amplitude_scale=1.0,
                 current_frequency_scale=1.0,
                 observation_current_mode="clean",
                 current_observation_noise=0.0,
                 current_observation_delay_steps=0,
                 include_prev_action_in_obs=False,
                 mass_scale_xy=1.0,
                 damping_scale_xy=1.0,
                 initial_position_std=0.0,
                 initial_velocity_std=0.0,
                 vertical_current=0.0,
                 vehicle_profile="real_10kg_v1",
                 action_mode="wrench",
                 thruster_layout="real_10kg_x"):
        super().__init__()
        self.mass_scale_xy = float(mass_scale_xy)
        self.damping_scale_xy = float(damping_scale_xy)
        self.vehicle_profile = str(vehicle_profile or "real_10kg_v1")
        self.action_mode = str(action_mode or "wrench")
        if self.action_mode not in {"wrench", "thruster"}:
            raise ValueError(f"Unknown action_mode: {self.action_mode}")
        self.thruster_layout = (
            build_thruster_layout(thruster_layout)
            if self.action_mode == "thruster"
            else None
        )
        self.dynamics = AUVDynamics(
            mass_scale_xy=self.mass_scale_xy,
            damping_scale_xy=self.damping_scale_xy,
            vehicle_profile=self.vehicle_profile,
        )
        self.scenario = scenario
        self.trajectory3d = bool(trajectory3d)
        self.start_time = float(start_time)
        self.current_amplitude_scale = float(current_amplitude_scale)
        self.current_frequency_scale = float(current_frequency_scale)
        self.observation_current_mode = str(observation_current_mode or "clean")
        self.current_observation_noise = float(current_observation_noise)
        self.current_observation_delay_steps = max(
            0, int(current_observation_delay_steps))
        self.include_prev_action_in_obs = bool(include_prev_action_in_obs)
        self.initial_position_std = max(0.0, float(initial_position_std))
        self.initial_velocity_std = max(0.0, float(initial_velocity_std))
        self.vertical_current = float(vertical_current)
        self.reward_config = dict(DEFAULT_REWARD_CONFIG)
        self.reward_config.update(reward_config or {})
        self.max_steps = max_steps  # 20.9s @ dt=0.01 鈫?exactly 1 loop of 8-shape
        # Resolve scenario: support both int codes and string names
        if isinstance(scenario, str):
            self._active_scenario = self.SCENARIO_MAP.get(scenario, 1)
        else:
            self._active_scenario = max(1, scenario)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(
                (29 if self.trajectory3d else 23)
                + (6 if self.include_prev_action_in_obs else 0),
            ),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self.max_thrust = np.array([60.0, 60.0, 30.0, 10.0, 10.0, 20.0])
        if self.action_mode == "thruster":
            self.max_thrust = np.full(6, self.thruster_layout.max_force)
        self.current_step = 0
        self.prev_action = np.zeros(6)
        self._current_velocity = np.zeros(2)
        self._prev_current_velocity = np.zeros(2)
        self._true_current_velocity = np.zeros(3)
        self._observed_current_velocity = np.zeros(2)
        self._prev_observed_current_velocity = np.zeros(2)
        self._current_observation_history = []
        self.prev_z_abs_error = None

    def _action_to_wrench(self, action):
        if self.action_mode == "thruster":
            forces = self.thruster_layout.normalized_action_to_forces(action)
            wrench = self.thruster_layout.forces_to_wrench(forces)
            return wrench, forces
        wrench = np.asarray(action, dtype=float).reshape(6) * self.max_thrust
        return wrench, None

    def _get_target(self, t):
        """Lissajous 8-shaped trajectory:
        xd = 3*sin(0.3t), yd = 1.5*sin(0.6t) 鈥?period = 2蟺/0.3 鈮?20.9s
        Peak velocity: surge=0.9 m/s, sway=0.9 m/s (well within AUV K/D=1.5 m/s)
        """
        xd = 3.0 * np.sin(0.3 * t)
        yd = 1.5 * np.sin(0.6 * t)
        dxd = 0.9 * np.cos(0.3 * t)
        dyd = 0.9 * np.cos(0.6 * t)
        zd = 0.8 * np.sin(0.2 * t) if self.trajectory3d else 0.0
        psi = np.arctan2(dyd, dxd)
        return np.array([xd, yd, zd, 0.0, 0.0, psi])

    def _get_target_velocity(self, t):
        """Analytical derivative of 8-shaped Lissajous trajectory."""
        dxd = 0.9 * np.cos(0.3 * t)
        dyd = 0.9 * np.cos(0.6 * t)
        dzd = 0.16 * np.cos(0.2 * t) if self.trajectory3d else 0.0
        return np.array([dxd, dyd, dzd, 0.0, 0.0, 0.0])

    def _base_current(self, t):
        s = self._active_scenario
        if s == 1:  # constant
            return np.array([0.3, 0.0])
        elif s == 2:  # sinusoidal
            return np.array([0.3*np.sin(0.2*t) + 0.1*np.cos(0.1*t),
                             0.2*np.cos(0.1*t) + 0.1*np.sin(0.2*t)])
        elif s == 3:  # step change inside the 20.9s evaluation episode
            return np.array([0.3, 0.0]) if t < 10.0 else np.array([0.0, 0.3])
        elif s == 4:  # turbulent deterministic current
            return np.array([0.25*np.sin(0.7*t) + 0.12*np.sin(2.1*t),
                             0.22*np.cos(0.5*t) + 0.10*np.sin(1.7*t)])
        elif s == 5:  # spatially varying proxy along the 8-shape phase
            phase = np.sin(0.3 * t)
            return np.array([0.28*np.sin(0.4*t + phase),
                             0.22*np.cos(0.3*t - 0.5*phase)])
        return np.zeros(2)

    def _generate_current(self, t):
        scaled_t = self.start_time + (
            float(t) - self.start_time) * self.current_frequency_scale
        return self.current_amplitude_scale * self._base_current(scaled_t)

    def _current_for_dynamics(self, horizontal_current):
        horizontal = np.asarray(horizontal_current, dtype=float).reshape(-1)[:2]
        if self.trajectory3d:
            return np.array(
                [horizontal[0], horizontal[1], self.vertical_current],
                dtype=float,
            )
        return horizontal.copy()

    def _observe_current(self, true_current):
        mode = self.observation_current_mode
        true_current = np.asarray(true_current, dtype=float)[:2]
        if mode in {"clean", "O0_clean"}:
            observed = true_current
        elif mode in {"partial_current", "zero_current", "O3_partial_current"}:
            observed = np.zeros(2, dtype=float)
        elif mode in {"noisy_current", "O1_noisy_current"}:
            noise = self.np_random.normal(
                0.0, self.current_observation_noise, size=2)
            observed = true_current + noise
        elif mode in {"delayed_current", "O2_delayed_current"}:
            if not self._current_observation_history:
                observed = true_current
            else:
                index = max(
                    0,
                    len(self._current_observation_history)
                    - self.current_observation_delay_steps
                    - 1,
                )
                observed = self._current_observation_history[index]
        else:
            raise ValueError(f"Unknown observation_current_mode: {mode}")
        self._current_observation_history.append(true_current.copy())
        return np.asarray(observed, dtype=float)

    def _get_obs(self):
        t = self.start_time + self.current_step * self.dynamics.dt
        target = self._get_target(t)
        eta, nu = self.dynamics.eta, self.dynamics.nu

        error = target - eta
        for i in [3, 4, 5]:
            error[i] = (error[i] + np.pi) % (2 * np.pi) - np.pi

        # Rotate xy error to body frame
        psi = eta[5]
        cp, sp = np.cos(psi), np.sin(psi)
        ex_w, ey_w = error[0], error[1]
        error[0] = cp * ex_w + sp * ey_w
        error[1] = -sp * ex_w + cp * ey_w

        dt_target = self._get_target_velocity(t)
        # Rotate target velocity to body frame
        dtx, dty = dt_target[0], dt_target[1]
        dt_target[0] =  cp * dtx + sp * dty
        dt_target[1] = -sp * dtx + cp * dty

        v_norm = np.linalg.norm(self._observed_current_velocity)
        current_delta = (
            self._observed_current_velocity
            - self._prev_observed_current_velocity
        )
        event_score = float(np.linalg.norm(current_delta))

        # Append ocean current velocity for actor to learn current compensation
        # obs_dim = 18 + 2 = 20
        obs_values = [
            error[0], error[1], error[2],
            error[3], error[4], error[5],
            nu[0], nu[1], nu[2],
            nu[3], nu[4], nu[5],
            dt_target[0], dt_target[1], dt_target[2],
            np.sin(t * 0.1), np.cos(t * 0.1),
            v_norm,
            self._observed_current_velocity[0],
            self._observed_current_velocity[1],
        ]
        if self.trajectory3d:
            obs_values.extend([
                current_delta[0], current_delta[1], 0.0, event_score,
                np.sin(t * 0.2), np.cos(t * 0.2),
                float(self._active_scenario) / 5.0, 0.0, 0.0,
            ])
        else:
            obs_values.extend([current_delta[0], current_delta[1], event_score])
        if self.include_prev_action_in_obs:
            obs_values.extend(np.asarray(self.prev_action, dtype=float).tolist())
        obs = np.array(obs_values, dtype=np.float32)
        return np.clip(obs, -10.0, 10.0)

    def _compute_z_progress_reward(self, z_error):
        """Reward reductions in absolute Z tracking error for 3D shaping."""
        current = float(abs(z_error))
        previous = self.prev_z_abs_error
        self.prev_z_abs_error = current
        if previous is None:
            return 0.0
        return float(self.reward_config.get("z_progress_weight", 0.0)) * (
            float(previous) - current
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.dynamics.reset()
        if self.initial_position_std > 0.0:
            self.dynamics.eta[:3] = self.np_random.normal(
                0.0,
                self.initial_position_std,
                size=3,
            )
        if self.initial_velocity_std > 0.0:
            self.dynamics.nu[:3] = self.np_random.normal(
                0.0,
                self.initial_velocity_std,
                size=3,
            )
        self.current_step = 0
        self.prev_action = np.zeros(6)
        self._current_observation_history = []
        self.prev_z_abs_error = None
        if self.scenario == 0:
            self._active_scenario = int(self.np_random.integers(1, 4))
        elif isinstance(self.scenario, str):
            self._active_scenario = self.SCENARIO_MAP.get(self.scenario, 1)
        else:
            self._active_scenario = self.scenario
        true_current_xy = self._generate_current(self.start_time)
        self._true_current_velocity = self._current_for_dynamics(true_current_xy)
        observed_current = self._observe_current(true_current_xy)
        self._current_velocity = observed_current.copy()
        self._observed_current_velocity = observed_current.copy()
        self._prev_current_velocity = observed_current.copy()
        self._prev_observed_current_velocity = observed_current.copy()
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        t = self.start_time + self.current_step * self.dynamics.dt

        current_xy = self._generate_current(t)
        v_current = self._current_for_dynamics(current_xy)
        self._true_current_velocity = v_current.copy()
        self._prev_current_velocity = self._current_velocity.copy()
        self._prev_observed_current_velocity = (
            self._observed_current_velocity.copy())
        observed_current = self._observe_current(current_xy)
        self._current_velocity = observed_current.copy()
        self._observed_current_velocity = observed_current.copy()
        applied_wrench, thruster_forces = self._action_to_wrench(action)
        eta, nu = self.dynamics.step(applied_wrench, v_current)
        self.current_step += 1

        target = self._get_target(t)
        error = target - eta
        for i in [3, 4, 5]:
            error[i] = (error[i] + np.pi) % (2 * np.pi) - np.pi

        pos_slice = slice(0, 3) if self.trajectory3d else slice(0, 2)
        pos_err = np.linalg.norm(error[pos_slice])

        # Velocity error: target velocity - actual body velocity
        dt_target = self._get_target_velocity(t)
        # Rotate target velocity to body frame
        psi = eta[5]
        cp_psi, sp_psi = np.cos(psi), np.sin(psi)
        dtx, dty = dt_target[0], dt_target[1]
        dtx_body =  cp_psi * dtx + sp_psi * dty
        dty_body = -sp_psi * dtx + cp_psi * dty
        vel_target_body = np.array(
            [dtx_body, dty_body, dt_target[2], 0.0, 0.0, 0.0])
        vel_err = vel_target_body - nu  # 6-dim velocity error
        vel_err_norm = np.linalg.norm(vel_err[:3] if self.trajectory3d else vel_err[:2])

        att_err = np.linalg.norm(error[3:5])

        smoothness = float(np.sum((action - self.prev_action) ** 2))
        energy = float(np.sum(action ** 2))
        reward = compute_tracking_reward(
            pos_err=pos_err,
            vel_err_norm=vel_err_norm,
            smoothness=smoothness,
            energy=energy,
            att_err=att_err,
            xy_err=float(np.linalg.norm(error[:2])),
            z_err=float(error[2]),
            xy_vel_err=float(np.linalg.norm(vel_err[:2])),
            z_vel_err=float(vel_err[2]),
            heading_err=float(error[5]),
            reward_config=self.reward_config,
        )
        reward += self._compute_z_progress_reward(error[2])
        self.prev_action = action.copy()

        info = {
            "dist_error": float(pos_err),
            "xy_dist_error": float(np.linalg.norm(error[:2])),
            "z_error": float(error[2]),
            "heading_error": float(abs(error[5])),
            "energy": energy,
            "reward_smoothness": smoothness,
            "privileged_state": np.array([v_current[0], v_current[1], v_current[2] if v_current.size >= 3 else 0.0,
                                          0.0, 0.0, 0.0]),
            "applied_wrench": applied_wrench.copy(),
        }
        if thruster_forces is not None:
            info["thruster_forces"] = thruster_forces.copy()
        done = self.current_step >= self.max_steps or float(pos_err) > 20.0
        return self._get_obs(), float(reward), done, False, info

    @property
    def privileged_state(self):
        vc = self._true_current_velocity
        z_current = vc[2] if np.asarray(vc).size >= 3 else 0.0
        return np.array([vc[0], vc[1], z_current, 0.0, 0.0, 0.0], dtype=np.float32)


