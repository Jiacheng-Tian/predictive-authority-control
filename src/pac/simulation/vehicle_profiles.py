from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VehicleProfile:
    """Diagonal vehicle parameters used by the simplified 6-DOF model."""

    name: str
    rigid_body_mass: np.ndarray
    added_mass: np.ndarray
    linear_damping: np.ndarray
    quadratic_damping: np.ndarray
    restoring_stiffness: np.ndarray

    @property
    def effective_mass(self) -> np.ndarray:
        return self.rigid_body_mass + self.added_mass


def _profile(
        name,
        rigid_body_mass,
        added_mass,
        linear_damping,
        quadratic_damping=None,
        restoring_stiffness=None) -> VehicleProfile:
    return VehicleProfile(
        name=str(name),
        rigid_body_mass=np.asarray(rigid_body_mass, dtype=float),
        added_mass=np.asarray(added_mass, dtype=float),
        linear_damping=np.asarray(linear_damping, dtype=float),
        quadratic_damping=np.asarray(
            quadratic_damping if quadratic_damping is not None else np.zeros(6),
            dtype=float,
        ),
        restoring_stiffness=np.asarray(
            restoring_stiffness if restoring_stiffness is not None else np.zeros(6),
            dtype=float,
        ),
    )


PROFILES = {
    "real_10kg_v1": _profile(
        "real_10kg_v1",
        rigid_body_mass=[
            10.0,
            10.0,
            10.0,
            0.1541666667,
            0.1854166667,
            0.2354166667,
        ],
        added_mass=[
            4.0,
            6.0,
            8.0,
            0.0658333333,
            0.0745833333,
            0.1145833333,
        ],
        linear_damping=[3.0, 4.0, 5.0, 0.20, 0.24, 0.08],
        quadratic_damping=[28.0, 34.0, 48.0, 0.72, 0.88, 0.30],
        restoring_stiffness=[0.0, 0.0, 0.0, 8.0, 10.0, 0.0],
    ),
}


def get_vehicle_profile(name: str | None = None) -> VehicleProfile:
    key = str(name or "real_10kg_v1")
    try:
        return PROFILES[key]
    except KeyError as exc:
        known = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown vehicle profile '{key}'. Known: {known}") from exc
