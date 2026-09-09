"""Deterministic, paired episode inputs for evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pac.evaluation.seeds import episode_uid as make_episode_uid


_MAX_SEED = 2**64 - 1


def _integer(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if integer != value:
        raise ValueError(f"{name} must be an integer")
    if integer < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return integer


def _finite_vector(value, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and have shape {shape}") from exc
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _readonly_matrix(value, name: str, shape: tuple[int, ...]) -> np.ndarray:
    return _finite_vector(value, name, shape)


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    """Immutable environment realization shared by paired evaluations."""

    scenario_id: int
    episode_seed: int
    steps: int
    dt: float
    initial_eta: np.ndarray
    initial_nu: np.ndarray
    current_estimation_noise: np.ndarray
    current_delay_steps: int = 2
    current_noise_std: float = 0.01
    episode_uid: str | None = None

    def __post_init__(self) -> None:
        scenario_id = _integer(self.scenario_id, "scenario_id", minimum=1)
        if scenario_id not in {1, 2, 3}:
            raise ValueError("scenario_id must be one of 1, 2, or 3")
        episode_seed = _integer(self.episode_seed, "episode_seed", minimum=0)
        if episode_seed > _MAX_SEED:
            raise ValueError("episode_seed exceeds uint64 maximum")
        steps = _integer(self.steps, "steps", minimum=1)
        try:
            dt = float(self.dt)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("dt must be finite and positive") from exc
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        current_delay_steps = _integer(
            self.current_delay_steps, "current_delay_steps", minimum=0
        )
        try:
            current_noise_std = float(self.current_noise_std)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("current_noise_std must be finite and non-negative") from exc
        if not np.isfinite(current_noise_std) or current_noise_std < 0.0:
            raise ValueError("current_noise_std must be finite and non-negative")

        initial_eta = _finite_vector(self.initial_eta, "initial_eta", (6,))
        initial_nu = _finite_vector(self.initial_nu, "initial_nu", (6,))
        current_estimation_noise = _readonly_matrix(
            self.current_estimation_noise,
            "current_estimation_noise",
            (steps, 3),
        )
        canonical_uid = make_episode_uid(scenario_id, episode_seed)
        if self.episode_uid is not None and str(self.episode_uid) != canonical_uid:
            raise ValueError("episode_uid must depend only on scenario_id and episode_seed")

        object.__setattr__(self, "scenario_id", scenario_id)
        object.__setattr__(self, "episode_seed", episode_seed)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "initial_eta", initial_eta)
        object.__setattr__(self, "initial_nu", initial_nu)
        object.__setattr__(self, "current_estimation_noise", current_estimation_noise)
        object.__setattr__(self, "current_delay_steps", current_delay_steps)
        object.__setattr__(self, "current_noise_std", current_noise_std)
        object.__setattr__(self, "episode_uid", canonical_uid)

    @property
    def environment_seed(self) -> int:
        """Alias used by paired evaluation tables."""
        return self.episode_seed


def build_episode_spec(
        scenario_id: int,
        episode_seed: int,
        steps: int,
        dt: float,
        *,
        current_delay_steps: int = 2,
        current_noise_std: float = 0.01) -> EpisodeSpec:
    """Generate one deterministic episode realization from an environment seed."""
    seed = _integer(episode_seed, "episode_seed", minimum=0)
    if seed > _MAX_SEED:
        raise ValueError("episode_seed exceeds uint64 maximum")
    step_count = _integer(steps, "steps", minimum=1)
    try:
        noise_std = float(current_noise_std)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("current_noise_std must be finite and non-negative") from exc
    if not np.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("current_noise_std must be finite and non-negative")

    random = np.random.default_rng(seed)
    initial_eta = np.zeros(6, dtype=float)
    initial_eta[:3] = random.normal(0.0, 0.03, size=3)
    initial_nu = np.zeros(6, dtype=float)
    initial_nu[:3] = random.normal(0.0, 0.01, size=3)
    noise = random.normal(0.0, noise_std, size=(step_count, 3))
    return EpisodeSpec(
        scenario_id=scenario_id,
        episode_seed=seed,
        steps=step_count,
        dt=dt,
        initial_eta=initial_eta,
        initial_nu=initial_nu,
        current_estimation_noise=noise,
        current_delay_steps=current_delay_steps,
        current_noise_std=noise_std,
    )
