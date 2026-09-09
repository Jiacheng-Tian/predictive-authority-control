"""Paired model/environment evaluation grids."""

from __future__ import annotations

import numpy as np

from pac.evaluation.seeds import episode_uid


_MAX_SEED = 2**64 - 1


def _unique_integer_values(
        values,
        name: str,
        *,
        minimum: int,
        maximum: int | None = None) -> list[int]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of integers")
    try:
        raw = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of integers") from exc
    result: list[int] = []
    seen: set[int] = set()
    for value in raw:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)):
            raise ValueError(f"{name} must contain integer values")
        integer = int(value)
        if integer < minimum or (maximum is not None and integer > maximum):
            raise ValueError(f"{name} contains a value outside its valid range")
        if integer in seen:
            raise ValueError(f"{name} must contain unique values")
        seen.add(integer)
        result.append(integer)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def build_paired_evaluation_grid(model_seeds, episode_seeds, scenarios) -> list[dict]:
    """Build baseline and learned rows with a shared environment seed dimension."""
    model_values = _unique_integer_values(
        model_seeds, "model_seeds", minimum=0, maximum=_MAX_SEED
    )
    episode_values = _unique_integer_values(
        episode_seeds, "episode_seeds", minimum=0, maximum=_MAX_SEED
    )
    scenario_values = _unique_integer_values(scenarios, "scenarios", minimum=1)
    if set(model_values) & set(episode_values):
        raise ValueError("model_seeds and episode_seeds must be disjoint")
    rows: list[dict] = []
    for scenario_id in scenario_values:
        for environment_seed in episode_values:
            uid = episode_uid(scenario_id, environment_seed)
            for method in ("SMC", "MPC"):
                rows.append({
                    "row_type": "baseline",
                    "method": method,
                    "model_seed": None,
                    "episode_seed": environment_seed,
                    "environment_seed": environment_seed,
                    "scenario_id": scenario_id,
                    "episode_uid": uid,
                })
            for model_seed in model_values:
                rows.append({
                    "row_type": "learned",
                    "method": "PAC",
                    "model_seed": model_seed,
                    "episode_seed": environment_seed,
                    "environment_seed": environment_seed,
                    "scenario_id": scenario_id,
                    "episode_uid": uid,
                })
    return rows
