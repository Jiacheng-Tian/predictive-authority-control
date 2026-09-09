"""Thin Gymnasium wrapper around the formal PAC simulator."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pac.simulation.core import AUVSimulator


class AUVTrackingEnv(gym.Env):
    """Expose the formal 3-D simulator through the Gymnasium API."""

    metadata = {"render_modes": []}

    def __init__(self, scenario: int = 1, max_steps: int = 2100, **kwargs):
        super().__init__()
        allowed = {
            "mass_scale_xy",
            "damping_scale_xy",
            "current_amplitude_scale",
            "current_frequency_scale",
            "initial_position_std",
            "initial_velocity_std",
            "vertical_current",
            "vehicle_profile",
            "thruster_layout",
        }
        unsupported = sorted(set(kwargs) - allowed)
        if unsupported:
            raise TypeError(f"unsupported simulator options: {unsupported}")
        simulator_kwargs = {key: value for key, value in kwargs.items() if key in allowed}
        self.simulator = AUVSimulator(
            scenario=scenario,
            max_steps=max_steps,
            **simulator_kwargs,
        )
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(29,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

    @property
    def dynamics(self):
        return self.simulator.dynamics

    @property
    def current_step(self):
        return self.simulator.current_step

    @property
    def privileged_state(self):
        return self.simulator.privileged_state

    def _get_target(self, t):
        return self.simulator._get_target(t)

    def _action_to_wrench(self, action):
        return self.simulator._action_to_wrench(action)

    def _observation(self) -> np.ndarray:
        dynamics = self.simulator.dynamics
        t = self.simulator.current_step * dynamics.dt
        target = self.simulator._get_target(t)
        error = target - dynamics.eta
        for index in (3, 4, 5):
            error[index] = (error[index] + np.pi) % (2 * np.pi) - np.pi
        psi = dynamics.eta[5]
        cosine, sine = np.cos(psi), np.sin(psi)
        error_x, error_y = error[0], error[1]
        error[0] = cosine * error_x + sine * error_y
        error[1] = -sine * error_x + cosine * error_y
        target_velocity = self.simulator._get_target_velocity(t)
        velocity_x, velocity_y = target_velocity[0], target_velocity[1]
        target_velocity[0] = cosine * velocity_x + sine * velocity_y
        target_velocity[1] = -sine * velocity_x + cosine * velocity_y
        current = self.simulator.privileged_state[:2]
        delta = self.simulator.current_delta
        values = [
            *error,
            *dynamics.nu,
            *target_velocity[:3],
            np.sin(t * 0.1),
            np.cos(t * 0.1),
            np.linalg.norm(current),
            *current,
            *delta,
            0.0,
            np.linalg.norm(delta),
            np.sin(t * 0.2),
            np.cos(t * 0.2),
            self.simulator.scenario / 5.0,
            0.0,
            0.0,
        ]
        return np.clip(np.asarray(values, dtype=np.float32), -10.0, 10.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        del options
        self.simulator.reset(seed=seed)
        return self._observation(), {}

    def step(self, action):
        done, info = self.simulator.step(action)
        terminated = bool(info["dist_error"] > 20.0)
        truncated = bool(done and not terminated)
        reward = -float(info["dist_error"])
        return self._observation(), reward, terminated, truncated, info
