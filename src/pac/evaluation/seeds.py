"""Seed partition validation and episode identity helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


_MAX_SEED = 2**64 - 1


def validate_disjoint_seed_partitions(
    partitions: Mapping[str, Sequence[int]],
) -> None:
    """Validate that each named seed role is unique and mutually disjoint."""
    seen: dict[int, str] = {}
    for role, seeds in partitions.items():
        role_seen: set[int] = set()
        for seed in seeds:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError(f"seed in role {role} must be an integer: {seed!r}")
            if seed < 0:
                raise ValueError(f"negative seed in role {role}: {seed}")
            if seed > _MAX_SEED:
                raise ValueError(f"seed in role {role} exceeds uint64 maximum: {seed}")
            if seed in role_seen:
                raise ValueError(f"duplicate seed in role {role}: {seed}")
            role_seen.add(seed)
            previous_role = seen.get(seed)
            if previous_role is not None:
                raise ValueError(
                    f"seed overlap between roles {previous_role} and {role}: {seed}"
                )
            seen[seed] = role


def episode_uid(scenario_id: int, episode_seed: int) -> str:
    """Return a stable environment episode identity independent of model seeds."""
    return f"scn{scenario_id}_env{episode_seed}"
