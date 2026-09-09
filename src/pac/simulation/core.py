"""Pure formal-v2-compatible AUV simulation core."""

from __future__ import annotations

import numpy as np

from pac.simulation.actuators import ActuatorLimits, SharedActuator
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
            thruster_layout: str = "real_10kg_x",
            dt: float = 0.01,
            actuator_command_min: float = -1.0,
            actuator_command_max: float = 1.0,
            actuator_max_delta_per_step: float | None = None):
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
        self.actuator = SharedActuator(ActuatorLimits(
            command_min=actuator_command_min,
            command_max=actuator_command_max,
            max_delta_per_step=actuator_max_delta_per_step,
        ))
        self.dynamics = AUVDynamics(
            mass_scale_xy=self.mass_scale_xy,
            damping_scale_xy=self.damping_scale_xy,
            vehicle_profile=self.vehicle_profile,
            dt=dt,
        )
        self.current_step = 0
        self.prev_action = np.zeros(6)
        self._true_current_velocity = np.zeros(3)
        self._previous_current_velocity = np.zeros(2)
        self._episode_spec = None
        self._actuator_history: list[dict[str, object]] = []

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

    def reset(self, seed: int | None = None, *, episode_spec=None) -> None:
        validated_eta = None
        validated_nu = None
        if episode_spec is not None:
            try:
                scenario_value = episode_spec.scenario_id
                steps_value = episode_spec.steps
                if isinstance(scenario_value, (bool, np.bool_)) or not isinstance(
                        scenario_value, (int, np.integer)):
                    raise ValueError("episode_spec scenario_id must be an integer")
                if isinstance(steps_value, (bool, np.bool_)) or not isinstance(
                        steps_value, (int, np.integer)):
                    raise ValueError("episode_spec steps must be an integer")
                scenario_id = int(scenario_value)
                steps = int(steps_value)
                spec_dt = float(episode_spec.dt)
                initial_eta = np.asarray(episode_spec.initial_eta, dtype=float)
                initial_nu = np.asarray(episode_spec.initial_nu, dtype=float)
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "episode_spec must provide scenario_id, steps, dt, initial_eta, and initial_nu"
                ) from exc
            if (
                    scenario_id != episode_spec.scenario_id
                    or steps != episode_spec.steps
                    or scenario_id != self.scenario
                    or steps != self.max_steps
            ):
                raise ValueError("episode_spec scenario_id and steps must match simulator")
            if not np.isfinite(spec_dt) or not np.isclose(spec_dt, self.dynamics.dt):
                raise ValueError("episode_spec dt must match simulator")
            if initial_eta.shape != (6,) or initial_nu.shape != (6,):
                raise ValueError("episode_spec initial states must have shape (6,)")
            if not np.all(np.isfinite(initial_eta)) or not np.all(np.isfinite(initial_nu)):
                raise ValueError("episode_spec initial states must be finite")
            current_noise = getattr(episode_spec, "current_estimation_noise", None)
            if current_noise is not None:
                try:
                    current_noise = np.asarray(current_noise, dtype=float)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        "episode_spec current_estimation_noise must be finite"
                    ) from exc
                if current_noise.shape != (steps, 3) or not np.all(np.isfinite(current_noise)):
                    raise ValueError(
                        "episode_spec current_estimation_noise must have shape (steps, 3) and be finite"
                    )
            validated_eta = np.array(initial_eta, dtype=float, copy=True)
            validated_nu = np.array(initial_nu, dtype=float, copy=True)

        self.dynamics.reset()
        self.actuator.reset()
        if episode_spec is None:
            random = np.random.default_rng(seed)
            if self.initial_position_std > 0.0:
                self.dynamics.eta[:3] = random.normal(0.0, self.initial_position_std, size=3)
            if self.initial_velocity_std > 0.0:
                self.dynamics.nu[:3] = random.normal(0.0, self.initial_velocity_std, size=3)
        else:
            self.dynamics.eta = validated_eta
            self.dynamics.nu = validated_nu
        self._episode_spec = episode_spec
        self.current_step = 0
        self.prev_action = np.zeros(6)
        self._actuator_history = []
        horizontal = self._generate_current(0.0)
        self._true_current_velocity = self._current_for_dynamics(horizontal)
        self._previous_current_velocity = horizontal.copy()

    def step(self, action) -> tuple[bool, dict]:
        actuator_step = self.actuator.apply(action)
        applied_action = actuator_step.applied
        t = self.current_step * self.dynamics.dt
        horizontal_current = self._generate_current(t)
        current = self._current_for_dynamics(horizontal_current)
        self._previous_current_velocity = self._true_current_velocity[:2].copy()
        self._true_current_velocity = current.copy()
        wrench, thruster_forces = self._action_to_wrench(applied_action)
        eta, nu = self.dynamics.step(wrench, current)
        self.current_step += 1

        # Preserve formal-v2 metric timing for numerical compatibility.
        target = self._get_target(t)
        error = target - eta
        for index in (3, 4, 5):
            error[index] = (error[index] + np.pi) % (2 * np.pi) - np.pi
        position_error = float(np.linalg.norm(error[:3]))
        smoothness = float(np.sum((applied_action - self.prev_action) ** 2))
        energy = float(np.sum(applied_action ** 2))
        self.prev_action = applied_action.copy()
        done = self.current_step >= self.max_steps or position_error > 20.0
        info = {
            "dist_error": position_error,
            "xy_dist_error": float(np.linalg.norm(error[:2])),
            "z_error": float(error[2]),
            "heading_error": float(abs(error[5])),
            "energy": energy,
            "reward_smoothness": smoothness,
            "requested_action": actuator_step.requested.copy(),
            "amplitude_clipped_action": actuator_step.amplitude_clipped.copy(),
            "applied_action": applied_action.copy(),
            "actuator_amplitude_clipped_fraction": actuator_step.amplitude_clipped_fraction,
            "actuator_rate_limited_fraction": actuator_step.rate_limited_fraction,
            "privileged_state": np.array([current[0], current[1], current[2], 0.0, 0.0, 0.0]),
            "applied_wrench": wrench.copy(),
            "thruster_forces": thruster_forces.copy(),
            "true_current": current.copy(),
            "sample_time": float(self.current_step * self.dynamics.dt),
        }
        self._actuator_history.append({
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in info.items()
            if key in {
                "requested_action",
                "amplitude_clipped_action",
                "applied_action",
                "actuator_amplitude_clipped_fraction",
                "actuator_rate_limited_fraction",
                "true_current",
                "sample_time",
            }
        })
        return done, info

    @property
    def privileged_state(self):
        current = self._true_current_velocity
        return np.array([current[0], current[1], current[2], 0.0, 0.0, 0.0], dtype=np.float32)

    @property
    def current_delta(self) -> np.ndarray:
        return self._true_current_velocity[:2] - self._previous_current_velocity

    @property
    def episode_spec(self):
        return self._episode_spec

    @property
    def actuator_telemetry(self) -> dict[str, np.ndarray]:
        """Return immutable per-step actuator and true-current telemetry."""
        if not self._actuator_history:
            empty = {
                "requested_action": np.empty((0, 6), dtype=float),
                "amplitude_clipped_action": np.empty((0, 6), dtype=float),
                "applied_action": np.empty((0, 6), dtype=float),
                "actuator_amplitude_clipped_fraction": np.empty(0, dtype=float),
                "actuator_rate_limited_fraction": np.empty(0, dtype=float),
                "true_current": np.empty((0, 3), dtype=float),
                "sample_time": np.empty(0, dtype=float),
            }
            for values in empty.values():
                values.setflags(write=False)
            return empty
        keys = (
            "requested_action",
            "amplitude_clipped_action",
            "applied_action",
            "actuator_amplitude_clipped_fraction",
            "actuator_rate_limited_fraction",
            "true_current",
            "sample_time",
        )
        telemetry = {}
        for key in keys:
            values = np.asarray([row[key] for row in self._actuator_history], dtype=float)
            values = np.array(values, dtype=float, copy=True)
            values.setflags(write=False)
            telemetry[key] = values
        return telemetry
