"""Pure formal-v2-compatible AUV simulation core."""

from __future__ import annotations

import numpy as np

from pac.simulation.dynamics import AUVDynamics
from pac.simulation.thrusters import build_thruster_layout


class AUVSimulator:
    """Formal 3-D, thruster-actuated simulator for current scenarios 1--3."""

    def __init__(
            self,
            *,
            scenario: int,
            max_steps: int = 2100,
            mass_scale_xy: float = 1.0,
            damping_scale_xy: float = 1.0,
            current_amplitude_scale: float = 2.5,
            current_frequency_scale: float = 1.0,
            initial_position_std: float = 0.0,
            initial_velocity_std: float = 0.0,
            vertical_current: float = 0.75,
            vehicle_profile: str = "real_10kg_v1",
            thruster_layout: str = "real_10kg_x"):
        if int(scenario) not in {1, 2, 3}:
            raise ValueError("scenario must be one of 1, 2, or 3")
        self.scenario = int(scenario)
        self.max_steps = int(max_steps)
        self.mass_scale_xy = float(mass_scale_xy)
        self.damping_scale_xy = float(damping_scale_xy)
        self.current_amplitude_scale = float(current_amplitude_scale)
        self.current_frequency_scale = float(current_frequency_scale)
        self.initial_position_std = max(0.0, float(initial_position_std))
        self.initial_velocity_std = max(0.0, float(initial_velocity_std))
        self.vertical_current = float(vertical_current)
        self.vehicle_profile = str(vehicle_profile)
        self.thruster_layout_name = str(thruster_layout)
        self.thruster_layout = build_thruster_layout(thruster_layout)
        self.dynamics = AUVDynamics(
            mass_scale_xy=self.mass_scale_xy,
            damping_scale_xy=self.damping_scale_xy,
            vehicle_profile=self.vehicle_profile,
        )
        self.current_step = 0
        self.prev_action = np.zeros(6)
        self._true_current_velocity = np.zeros(3)
        self._previous_current_velocity = np.zeros(2)

    @staticmethod
    def _get_target(t):
        x = 3.0 * np.sin(0.3 * t)
        y = 1.5 * np.sin(0.6 * t)
        z = 0.8 * np.sin(0.2 * t)
        dx = 0.9 * np.cos(0.3 * t)
        dy = 0.9 * np.cos(0.6 * t)
        return np.array([x, y, z, 0.0, 0.0, np.arctan2(dy, dx)])

    @staticmethod
    def _get_target_velocity(t):
        return np.array([
            0.9 * np.cos(0.3 * t),
            0.9 * np.cos(0.6 * t),
            0.16 * np.cos(0.2 * t),
            0.0,
            0.0,
            0.0,
        ])

    def _base_current(self, t):
        if self.scenario == 1:
            return np.array([0.3, 0.0])
        if self.scenario == 2:
            return np.array([
                0.3 * np.sin(0.2 * t) + 0.1 * np.cos(0.1 * t),
                0.2 * np.cos(0.1 * t) + 0.1 * np.sin(0.2 * t),
            ])
        return np.array([0.3, 0.0]) if t < 10.0 else np.array([0.0, 0.3])

    def _generate_current(self, t):
        return self.current_amplitude_scale * self._base_current(
            float(t) * self.current_frequency_scale
        )

    def _current_for_dynamics(self, horizontal_current):
        horizontal = np.asarray(horizontal_current, dtype=float).reshape(-1)[:2]
        return np.array([horizontal[0], horizontal[1], self.vertical_current], dtype=float)

    def _action_to_wrench(self, action):
        forces = self.thruster_layout.normalized_action_to_forces(action)
        return self.thruster_layout.forces_to_wrench(forces), forces

    def reset(self, seed: int | None = None) -> None:
        self.dynamics.reset()
        random = np.random.default_rng(seed)
        if self.initial_position_std > 0.0:
            self.dynamics.eta[:3] = random.normal(0.0, self.initial_position_std, size=3)
        if self.initial_velocity_std > 0.0:
            self.dynamics.nu[:3] = random.normal(0.0, self.initial_velocity_std, size=3)
        self.current_step = 0
        self.prev_action = np.zeros(6)
        horizontal = self._generate_current(0.0)
        self._true_current_velocity = self._current_for_dynamics(horizontal)
        self._previous_current_velocity = horizontal.copy()

    def step(self, action) -> tuple[bool, dict]:
        action = np.clip(action, -1.0, 1.0)
        t = self.current_step * self.dynamics.dt
        horizontal_current = self._generate_current(t)
        current = self._current_for_dynamics(horizontal_current)
        self._previous_current_velocity = self._true_current_velocity[:2].copy()
        self._true_current_velocity = current.copy()
        wrench, thruster_forces = self._action_to_wrench(action)
        eta, nu = self.dynamics.step(wrench, current)
        self.current_step += 1

        # Formal v2 compares the post-step state with the pre-step reference.
        target = self._get_target(t)
        error = target - eta
        for index in (3, 4, 5):
            error[index] = (error[index] + np.pi) % (2 * np.pi) - np.pi
        position_error = float(np.linalg.norm(error[:3]))
        smoothness = float(np.sum((action - self.prev_action) ** 2))
        energy = float(np.sum(action ** 2))
        self.prev_action = action.copy()
        done = self.current_step >= self.max_steps or position_error > 20.0
        return done, {
            "dist_error": position_error,
            "xy_dist_error": float(np.linalg.norm(error[:2])),
            "z_error": float(error[2]),
            "heading_error": float(abs(error[5])),
            "energy": energy,
            "reward_smoothness": smoothness,
            "privileged_state": np.array([current[0], current[1], current[2], 0.0, 0.0, 0.0]),
            "applied_wrench": wrench.copy(),
            "thruster_forces": thruster_forces.copy(),
        }

    @property
    def privileged_state(self):
        current = self._true_current_velocity
        return np.array([current[0], current[1], current[2], 0.0, 0.0, 0.0], dtype=np.float32)

    @property
    def current_delta(self) -> np.ndarray:
        return self._true_current_velocity[:2] - self._previous_current_velocity
