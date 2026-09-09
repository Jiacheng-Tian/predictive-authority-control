"""Causal current observations used by paired evaluation."""

from __future__ import annotations

from collections import deque

import numpy as np


def _readonly(values, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if shape is not None and array.shape != shape:
        raise ValueError(f"value must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("value must be finite")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


class CausalCurrentEstimator:
    """Return delayed true current plus pre-generated, indexed noise."""

    def __init__(self, delay_steps: int, noise) -> None:
        if isinstance(delay_steps, (bool, np.bool_)):
            raise ValueError("delay_steps must be a non-negative integer")
        try:
            delay = int(delay_steps)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("delay_steps must be a non-negative integer") from exc
        if delay != delay_steps or delay < 0:
            raise ValueError("delay_steps must be a non-negative integer")
        noise_array = np.asarray(noise, dtype=float)
        if noise_array.ndim != 2 or noise_array.shape[1] != 3:
            raise ValueError("noise must have shape (steps, 3)")
        if not np.all(np.isfinite(noise_array)):
            raise ValueError("noise must be finite")
        self.delay_steps = delay
        self.noise = _readonly(noise_array)
        self._buffer: deque[np.ndarray] | None = None
        self._true: list[np.ndarray] = []
        self._delayed: list[np.ndarray] = []
        self._noise: list[np.ndarray] = []
        self._estimate: list[np.ndarray] = []

    def reset(self, initial_current) -> None:
        initial = _readonly(initial_current, (3,))
        self._buffer = deque(
            (initial.copy() for _ in range(self.delay_steps)),
            maxlen=max(self.delay_steps, 1),
        )
        self._true.clear()
        self._delayed.clear()
        self._noise.clear()
        self._estimate.clear()

    def estimate(self, true_current, step: int) -> np.ndarray:
        if self._buffer is None:
            raise RuntimeError("estimator must be reset before estimate")
        if isinstance(step, (bool, np.bool_)):
            raise ValueError("step must be a non-negative integer")
        try:
            index = int(step)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("step must be a non-negative integer") from exc
        if index != step or index < 0 or index >= len(self.noise):
            raise ValueError("step must index the pre-generated noise")
        current = _readonly(true_current, (3,))
        if self.delay_steps:
            delayed = self._buffer.popleft()
            self._buffer.append(current.copy())
        else:
            delayed = current.copy()
        noise = self.noise[index]
        estimate = _readonly(delayed + noise, (3,))
        self._true.append(current)
        self._delayed.append(_readonly(delayed, (3,)))
        self._noise.append(_readonly(noise, (3,)))
        self._estimate.append(estimate)
        return estimate.copy()

    @property
    def telemetry(self) -> dict[str, np.ndarray]:
        def stack(values: list[np.ndarray]) -> np.ndarray:
            if values:
                return _readonly(np.stack(values, axis=0), (len(values), 3))
            return _readonly(np.empty((0, 3), dtype=float), (0, 3))

        values = {
            "true_current": stack(self._true),
            "delayed_current": stack(self._delayed),
            "noise": stack(self._noise),
            "estimate": stack(self._estimate),
        }
        values["estimated_current"] = values["estimate"]
        return values

    @property
    def true_current(self) -> np.ndarray:
        return self.telemetry["true_current"]

    @property
    def delayed_current(self) -> np.ndarray:
        return self.telemetry["delayed_current"]

    @property
    def estimated_current(self) -> np.ndarray:
        return self.telemetry["estimate"]
