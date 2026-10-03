"""Stochastic-disturbance simulators and episode specs for v4.

The base v3 simulator is treated as frozen: :class:`DisturbanceSimulator` subclasses
``AUVSimulator`` and only layers v4-only disturbance families on top of the
archived dynamics.  Structured v4 episodes (family ``structured``) are
bit-identical to supervised episodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
from typing import Any, Mapping

import numpy as np

from pac.evaluation.episode_spec import EpisodeSpec
from pac.simulation.core import AUVSimulator

STRUCTURED_FAMILY = "structured"
DISTURBANCE_FAMILIES = (
    "ou_current",
    "colored_noise",
    "random_freq_amp",
    "mass_damping_mismatch",
    "actuator_delay_noise",
    "fast_ou",
)
# Evaluation-only families: fast_ou shares the OU simulator dynamics with
# extrapolated parameters; estimation_delay keeps structured (sinusoidal)
# currents and only degrades the causal current estimator through the
# episode spec's delay/noise fields.
EVAL_ONLY_FAMILIES = ("fast_ou", "estimation_delay")
_KNOWN_FAMILIES = (STRUCTURED_FAMILY, *DISTURBANCE_FAMILIES, *EVAL_ONLY_FAMILIES)
_FLOAT_PARAMS = {
    "theta",
    "sigma",
    "mean_scale",
    "rho",
    "amplitude_scale",
    "frequency_scale",
    "mass_scale_xy",
    "damping_scale_xy",
    "action_noise_std",
    "noise_std",
}
_INT_PARAMS = {"delay_steps"}
_MAX_SEED = 2**64 - 1


def family_rng_offset(family: str) -> int:
    """Deterministic per-family RNG stream offset derived from the name."""
    digest = hashlib.sha256(f"pac-v4-family\0{family}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def draw_disturbance_params(
        family: str,
        episode_seed: int,
        ranges: Mapping[str, Any]) -> dict[str, float | int]:
    """Draw one deterministic disturbance parameter set from profile ranges.

    ``ranges`` maps parameter names to ``(low, high)`` pairs.  Integer-valued
    parameters (``delay_steps``) are drawn uniformly over the inclusive
    integer interval; float parameters are drawn uniformly over the interval.
    """
    if family not in _KNOWN_FAMILIES:
        raise ValueError(f"unknown disturbance family: {family}")
    if isinstance(episode_seed, (bool, np.bool_)) or not isinstance(episode_seed, (int, np.integer)):
        raise ValueError("episode_seed must be an integer")
    seed = int(episode_seed)
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError("episode_seed must be a non-negative uint64 value")
    rng = np.random.default_rng(seed + family_rng_offset(family))
    params: dict[str, float | int] = {}
    for name, bounds in ranges.items():
        low, high = float(bounds[0]), float(bounds[1])
        if not np.isfinite(low) or not np.isfinite(high) or not low < high:
            raise ValueError(f"invalid range for {family}.{name}: ({low}, {high})")
        if name in _FLOAT_PARAMS:
            params[name] = float(rng.uniform(low, high))
        elif name in _INT_PARAMS:
            low_i, high_i = int(np.ceil(low)), int(np.floor(high))
            if low_i > high_i:
                raise ValueError(f"invalid integer range for {family}.{name}")
            params[name] = int(rng.integers(low_i, high_i + 1))
        else:
            raise ValueError(f"unknown disturbance parameter: {family}.{name}")
    missing = {"theta", "sigma", "mean_scale"} - set(params) if family in (
        "ou_current", "fast_ou") else set()
    if family == "colored_noise" and not {"rho", "sigma"} <= set(params):
        missing = {"rho", "sigma"}
    if missing:
        raise ValueError(f"missing disturbance parameters for {family}: {sorted(missing)}")
    return params


@dataclass(frozen=True)
class PairedEpisodeSpec:
    """Immutable v4 episode realization including disturbance parameters.

    Structured episodes (``family="structured"``) carry no disturbance
    parameters and are pairwise-comparable with supervised EpisodeSpec realizations
    generated from the same seed: the initial-state and estimation-noise
    draws use the identical construction.
    """

    family: str
    scenario_id: int
    episode_seed: int
    steps: int
    dt: float
    initial_eta: np.ndarray
    initial_nu: np.ndarray
    current_estimation_noise: np.ndarray
    current_delay_steps: int = 2
    current_noise_std: float = 0.01
    disturbance_params: Mapping[str, float | int] = field(default_factory=dict)
    episode_uid: str | None = None

    def __post_init__(self) -> None:
        if self.family not in _KNOWN_FAMILIES:
            raise ValueError(f"unknown disturbance family: {self.family}")
        if isinstance(self.scenario_id, (bool, np.bool_)) or not isinstance(
                self.scenario_id, (int, np.integer)):
            raise ValueError("scenario_id must be an integer")
        scenario_id = int(self.scenario_id)
        if scenario_id not in (1, 2, 3):
            raise ValueError("scenario_id must be one of 1, 2, or 3")
        if isinstance(self.episode_seed, (bool, np.bool_)) or not isinstance(
                self.episode_seed, (int, np.integer)):
            raise ValueError("episode_seed must be an integer")
        seed = int(self.episode_seed)
        if not 0 <= seed <= _MAX_SEED:
            raise ValueError("episode_seed must be a non-negative uint64 value")
        steps = int(self.steps)
        if steps <= 0:
            raise ValueError("steps must be positive")
        dt = float(self.dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        initial_eta = np.asarray(self.initial_eta, dtype=float)
        initial_nu = np.asarray(self.initial_nu, dtype=float)
        noise = np.asarray(self.current_estimation_noise, dtype=float)
        if initial_eta.shape != (6,) or initial_nu.shape != (6,):
            raise ValueError("initial states must have shape (6,)")
        if noise.shape != (steps, 3):
            raise ValueError("current_estimation_noise must have shape (steps, 3)")
        if not (np.isfinite(initial_eta).all() and np.isfinite(initial_nu).all()
                and np.isfinite(noise).all()):
            raise ValueError("episode spec arrays must be finite")
        params = dict(self.disturbance_params)
        if self.family == STRUCTURED_FAMILY:
            if params:
                raise ValueError("structured episodes must not carry disturbance params")
        else:
            if not params:
                raise ValueError("disturbance episodes require disturbance params")
            for key, value in params.items():
                if not isinstance(value, (int, float, np.integer, np.floating)):
                    raise ValueError(f"disturbance param {key} must be numeric")
                if not np.isfinite(float(value)):
                    raise ValueError(f"disturbance param {key} must be finite")
        canonical_uid = paired_episode_uid(self.family, scenario_id, seed)
        if self.episode_uid is not None and str(self.episode_uid) != canonical_uid:
            raise ValueError("episode_uid must depend only on family, scenario_id, and seed")
        object.__setattr__(self, "scenario_id", scenario_id)
        object.__setattr__(self, "episode_seed", seed)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "initial_eta", np.array(initial_eta, dtype=float, copy=True))
        object.__setattr__(self, "initial_nu", np.array(initial_nu, dtype=float, copy=True))
        object.__setattr__(self, "current_estimation_noise",
                           np.array(noise, dtype=float, copy=True))
        object.__setattr__(self, "disturbance_params", params)
        object.__setattr__(self, "episode_uid", canonical_uid)

    @property
    def environment_seed(self) -> int:
        return self.episode_seed

    def to_supervised_style_dict(self) -> dict[str, Any]:
        payload = {f.name: getattr(self, f.name) for f in fields(self)}
        payload["disturbance_params"] = dict(self.disturbance_params)
        return payload


def paired_episode_uid(family: str, scenario_id: int, episode_seed: int) -> str:
    if not isinstance(family, str) or not family:
        raise ValueError("family must be a non-empty string")
    return f"scn{int(scenario_id)}_{family}_env{int(episode_seed)}"


def build_paired_episode_spec(
        family: str,
        scenario_id: int,
        episode_seed: int,
        steps: int,
        dt: float,
        *,
        disturbance_params: Mapping[str, float | int] | None = None,
        current_delay_steps: int = 2,
        current_noise_std: float = 0.01) -> PairedEpisodeSpec:
    """Generate one deterministic v4 episode realization from an environment seed.

    The initial-state and current-estimation-noise construction matches the
    frozen supervised ``build_episode_spec`` so that structured v4 episodes and v3
    episodes from the same seed share initial conditions.
    """
    if family not in _KNOWN_FAMILIES:
        raise ValueError(f"unknown disturbance family: {family}")
    seed = int(episode_seed)
    if isinstance(episode_seed, (bool, np.bool_)) or not isinstance(episode_seed, (int, np.integer)):
        raise ValueError("episode_seed must be an integer")
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError("episode_seed must be a non-negative uint64 value")
    step_count = int(steps)
    if step_count <= 0:
        raise ValueError("steps must be positive")
    rng = np.random.default_rng(seed)
    initial_eta = np.zeros(6, dtype=float)
    initial_eta[:3] = rng.normal(0.0, 0.03, size=3)
    initial_nu = np.zeros(6, dtype=float)
    initial_nu[:3] = rng.normal(0.0, 0.01, size=3)
    noise = rng.normal(0.0, float(current_noise_std), size=(step_count, 3))
    params = dict(disturbance_params) if disturbance_params is not None else {}
    if family == "estimation_delay":
        # The estimation_delay family keeps structured currents and only
        # degrades the causal estimator: its drawn parameters override the
        # spec-level delay and noise fields.
        current_delay_steps = int(params.get("delay_steps", current_delay_steps))
        current_noise_std = float(params.get("noise_std", current_noise_std))
    return PairedEpisodeSpec(
        family=family,
        scenario_id=int(scenario_id),
        episode_seed=seed,
        steps=step_count,
        dt=float(dt),
        initial_eta=initial_eta,
        initial_nu=initial_nu,
        current_estimation_noise=noise,
        current_delay_steps=int(current_delay_steps),
        current_noise_std=float(current_noise_std),
        disturbance_params=params,
    )


class DisturbanceSimulator(AUVSimulator):
    """AUVSimulator with pluggable disturbance families.

    ``family="structured"`` reproduces the base simulator exactly.  The
    stochastic current families (``ou_current``, ``colored_noise``) advance a
    seeded process once per simulation step; ``_generate_current`` therefore
    ignores ``t`` for those families and callers must not poll it for future
    stamps.  Offline rollouts that need the realized current trajectory use
    :meth:`current_at_step` over the recorded sequence instead.
    """

    def __init__(
            self,
            *,
            scenario: int,
            max_steps: int = 2100,
            family: str = STRUCTURED_FAMILY,
            disturbance_params: Mapping[str, float | int] | None = None,
            disturbance_seed: int | None = None,
            **kwargs: Any):
        if family not in _KNOWN_FAMILIES:
            raise ValueError(f"unknown disturbance family: {family}")
        self.family = family
        params = dict(disturbance_params or {})
        if family == STRUCTURED_FAMILY and params:
            raise ValueError("structured simulator must not carry disturbance params")
        if family != STRUCTURED_FAMILY and not params:
            raise ValueError(f"disturbance family {family} requires params")
        self.disturbance_params = params
        if family == "mass_damping_mismatch":
            kwargs["mass_scale_xy"] = float(params["mass_scale_xy"])
            kwargs["damping_scale_xy"] = float(params["damping_scale_xy"])
        if family == "random_freq_amp":
            kwargs["current_amplitude_scale"] = float(params["amplitude_scale"])
            kwargs["current_frequency_scale"] = float(params["frequency_scale"])
        super().__init__(scenario=scenario, max_steps=max_steps, **kwargs)
        self._disturbance_seed = None if disturbance_seed is None else int(disturbance_seed)
        self._disturbance_rng = np.random.default_rng(
            0 if self._disturbance_seed is None else self._disturbance_seed
        )
        self._action_noise_std = (
            float(params.get("action_noise_std", 0.0))
            if family == "actuator_delay_noise" else 0.0
        )
        self._actuator_delay_steps = (
            int(params.get("delay_steps", 0))
            if family == "actuator_delay_noise" else 0
        )
        if self._action_noise_std < 0.0:
            raise ValueError("action_noise_std must be non-negative")
        if self._actuator_delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")
        self._command_queue: list[np.ndarray] = []
        self._ou_state = np.zeros(2, dtype=float)
        self._ar_state = np.zeros(2, dtype=float)
        self._current_trajectory: list[np.ndarray] = []
        self._process_cache = np.zeros(2, dtype=float)
        self._process_cache_id = -1

    # -- internal disturbance process helpers ------------------------------
    def _advance_ou_current(self) -> np.ndarray:
        params = self.disturbance_params
        theta = float(params["theta"])
        sigma = float(params["sigma"])
        mean_scale = float(params["mean_scale"])
        mean = np.array([0.3 * mean_scale, 0.0])
        dt = self.dynamics.dt
        self._ou_state = (
            self._ou_state
            + theta * (mean - self._ou_state) * dt
            + sigma * np.sqrt(dt) * self._disturbance_rng.standard_normal(2)
        )
        return self._ou_state.copy()

    def _advance_colored_current(self) -> np.ndarray:
        params = self.disturbance_params
        rho = float(params["rho"])
        sigma = float(params["sigma"])
        if not 0.0 <= rho < 1.0:
            raise ValueError("colored noise rho must be in [0, 1)")
        innovation_scale = sigma * np.sqrt(1.0 - rho ** 2)
        self._ar_state = (
            rho * self._ar_state
            + innovation_scale * self._disturbance_rng.standard_normal(2)
        )
        return self._ar_state.copy()

    def _generate_current(self, t):
        if self.family not in ("ou_current", "fast_ou", "colored_noise"):
            return super()._generate_current(t)
        # Stochastic currents advance exactly once per simulation step so that
        # pre-step measurements and the dynamics see the same realization.
        needed = int(self.current_step) + 1
        if self._process_cache_id != needed:
            if self.family in ("ou_current", "fast_ou"):
                value = self._advance_ou_current()
            else:
                value = self._advance_colored_current()
            self._process_cache = value
            self._process_cache_id = needed
        return np.array(self._process_cache, dtype=float, copy=True)

    def current_at_step(self, step: int) -> np.ndarray:
        """Return the realized true current recorded for ``step`` (read-only)."""
        index = int(step)
        if not 0 <= index < len(self._current_trajectory):
            raise ValueError(f"no realized current recorded for step {index}")
        return self._current_trajectory[index].copy()

    # -- lifecycle ---------------------------------------------------------
    def reset(self, seed: int | None = None, *, episode_spec=None) -> None:
        self._disturbance_rng = np.random.default_rng(
            0 if self._disturbance_seed is None else self._disturbance_seed
        )
        self._ou_state = np.zeros(2, dtype=float)
        self._ar_state = np.zeros(2, dtype=float)
        self._command_queue = [np.zeros(6) for _ in range(self._actuator_delay_steps)]
        self._current_trajectory = []
        self._process_cache = np.zeros(2, dtype=float)
        self._process_cache_id = -1
        super().reset(seed, episode_spec=episode_spec)

    def step(self, action) -> tuple[bool, dict]:
        requested = np.asarray(action, dtype=float).reshape(6)
        if not np.isfinite(requested).all():
            raise ValueError("action must be finite")
        if self._action_noise_std > 0.0:
            submitted = np.clip(
                requested
                + self._disturbance_rng.normal(0.0, self._action_noise_std, size=6),
                -1.0,
                1.0,
            )
        else:
            submitted = requested
        if self._command_queue:
            self._command_queue.append(submitted)
            command = self._command_queue.pop(0)
        else:
            command = submitted
        done, info = super().step(command)
        self._current_trajectory.append(np.asarray(info["true_current"], dtype=float).copy())
        return done, info

    @property
    def realized_current_trajectory(self) -> np.ndarray:
        return np.stack(self._current_trajectory, axis=0) if self._current_trajectory else (
            np.empty((0, 3), dtype=float)
        )


def make_disturbance_simulator(
        spec: PairedEpisodeSpec,
        *,
        mass_scale_xy: float = 1.0,
        damping_scale_xy: float = 1.0,
        current_amplitude_scale: float = 2.5,
        current_frequency_scale: float = 1.0,
        vertical_current: float = 0.75,
        vehicle_profile: str = "real_10kg_v1",
        thruster_layout: str = "real_10kg_x",
        dt: float = 0.01,
        actuator_command_min: float = -1.0,
        actuator_command_max: float = 1.0,
        actuator_max_delta_per_step: float | None = None,
        max_force: float = 35.0) -> DisturbanceSimulator:
    """Build a DisturbanceSimulator configured from one episode spec."""
    if spec.family == "mass_damping_mismatch":
        mass_scale_xy = float(spec.disturbance_params["mass_scale_xy"])
        damping_scale_xy = float(spec.disturbance_params["damping_scale_xy"])
    if spec.family == "random_freq_amp":
        current_amplitude_scale = float(spec.disturbance_params["amplitude_scale"])
        current_frequency_scale = float(spec.disturbance_params["frequency_scale"])
    return DisturbanceSimulator(
        scenario=spec.scenario_id,
        max_steps=spec.steps,
        family=spec.family,
        disturbance_params=spec.disturbance_params,
        disturbance_seed=spec.episode_seed,
        mass_scale_xy=mass_scale_xy,
        damping_scale_xy=damping_scale_xy,
        current_amplitude_scale=current_amplitude_scale,
        current_frequency_scale=current_frequency_scale,
        vertical_current=vertical_current,
        vehicle_profile=vehicle_profile,
        thruster_layout=thruster_layout,
        dt=dt,
        actuator_command_min=actuator_command_min,
        actuator_command_max=actuator_command_max,
        actuator_max_delta_per_step=actuator_max_delta_per_step,
        max_force=max_force,
    )


__all__ = [
    "STRUCTURED_FAMILY",
    "DISTURBANCE_FAMILIES",
    "PairedEpisodeSpec",
    "DisturbanceSimulator",
    "build_paired_episode_spec",
    "draw_disturbance_params",
    "family_rng_offset",
    "make_disturbance_simulator",
    "paired_episode_uid",
]
