"""Paired model/environment evaluation grids."""

from __future__ import annotations

from pac.evaluation.seeds import episode_uid


_MAX_SEED = 2**64 - 1


def _positive_unique(values, name: str) -> list[int]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of positive integers")
    try:
        raw = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of positive integers") from exc
    result: list[int] = []
    seen: set[int] = set()
    for value in raw:
        if isinstance(value, bool):
            raise ValueError(f"{name} must contain positive integers")
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain positive integers") from exc
        if integer != value or integer <= 0 or integer > _MAX_SEED:
            raise ValueError(f"{name} must contain positive integers")
        if integer in seen:
            raise ValueError(f"{name} must contain unique values")
        seen.add(integer)
        result.append(integer)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def build_paired_evaluation_grid(model_seeds, episode_seeds, scenarios) -> list[dict]:
    """Build baseline and learned rows with a shared environment seed dimension."""
    model_values = _positive_unique(model_seeds, "model_seeds")
    episode_values = _positive_unique(episode_seeds, "episode_seeds")
    scenario_values = _positive_unique(scenarios, "scenarios")
    if set(model_values) & set(episode_values):
        raise ValueError("model_seeds and episode_seeds must be disjoint")
    if any(scenario not in {1, 2, 3} for scenario in scenario_values):
        raise ValueError("scenarios must be one of 1, 2, or 3")

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
