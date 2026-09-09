"""Shared actuator command limits and per-step slew dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActuatorLimits:
    """Scalar command bounds and an optional per-channel slew limit."""

    command_min: float
    command_max: float
    max_delta_per_step: float | None = None

    def __post_init__(self) -> None:
        try:
            command_min = float(self.command_min)
            command_max = float(self.command_max)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("actuator command bounds must be finite numbers") from exc
        if not np.isfinite(command_min) or not np.isfinite(command_max):
            raise ValueError("actuator command bounds must be finite")
        if command_min >= command_max:
            raise ValueError("command_min must be less than command_max")
        if command_min > 0.0 or command_max < 0.0:
            raise ValueError("actuator command bounds must include zero")
        if command_min < -1.0 or command_max > 1.0:
            raise ValueError("actuator command bounds must lie in [-1, 1]")

        max_delta = self.max_delta_per_step
        if max_delta is not None:
            try:
                max_delta = float(max_delta)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("max_delta_per_step must be finite and non-negative") from exc
            if not np.isfinite(max_delta) or max_delta < 0.0:
                raise ValueError("max_delta_per_step must be finite and non-negative")

        object.__setattr__(self, "command_min", command_min)
        object.__setattr__(self, "command_max", command_max)
        object.__setattr__(self, "max_delta_per_step", max_delta)


@dataclass(frozen=True)
class ActuatorStep:
    """Immutable result of applying one shared actuator command."""

    requested: np.ndarray
    amplitude_clipped: np.ndarray
    applied: np.ndarray
    amplitude_clipped_fraction: float
    rate_limited_fraction: float


class SharedActuator:
    """Apply shared amplitude and per-channel slew limits to six commands."""

    def __init__(self, limits: ActuatorLimits):
        if not isinstance(limits, ActuatorLimits):
            raise TypeError("limits must be an ActuatorLimits instance")
        self.limits = limits
        self._previous_applied = np.zeros(6, dtype=float)

    def reset(self) -> None:
        self._previous_applied = np.zeros(6, dtype=float)

    @staticmethod
    def _validate_request(requested) -> np.ndarray:
        try:
            values = np.asarray(requested, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("requested actuator command must be a finite shape-(6,) array") from exc
        if values.shape != (6,):
            raise ValueError("requested actuator command must have shape (6,)")
        if not np.all(np.isfinite(values)):
            raise ValueError("requested actuator command must be finite")
        return values.copy()

    def apply(self, requested) -> ActuatorStep:
        requested_array = self._validate_request(requested)
        limits = self.limits
        amplitude_clipped = np.clip(
            requested_array,
            limits.command_min,
            limits.command_max,
        )
        amplitude_mask = requested_array != amplitude_clipped

        if limits.max_delta_per_step is None:
            applied = amplitude_clipped.copy()
            rate_mask = np.zeros(6, dtype=bool)
        else:
            delta = amplitude_clipped - self._previous_applied
            applied_delta = np.clip(
                delta,
                -limits.max_delta_per_step,
                limits.max_delta_per_step,
            )
            applied = self._previous_applied + applied_delta
            rate_mask = np.abs(delta) > limits.max_delta_per_step

        self._previous_applied = applied.copy()
        return ActuatorStep(
            requested=self._readonly_copy(requested_array),
            amplitude_clipped=self._readonly_copy(amplitude_clipped),
            applied=self._readonly_copy(applied),
            amplitude_clipped_fraction=float(np.count_nonzero(amplitude_mask) / 6.0),
            rate_limited_fraction=float(np.count_nonzero(rate_mask) / 6.0),
        )

    @staticmethod
    def _readonly_copy(values) -> np.ndarray:
        result = np.array(values, dtype=float, copy=True)
        result.setflags(write=False)
        return result
