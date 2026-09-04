from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThrusterLayout:
    """Maps six physical thruster forces to a body-frame 6-DOF wrench."""

    positions: np.ndarray
    directions: np.ndarray
    max_force: float = 35.0

    def __post_init__(self):
        object.__setattr__(
            self, "positions", np.asarray(self.positions, dtype=float).reshape(6, 3)
        )
        object.__setattr__(
            self, "directions", np.asarray(self.directions, dtype=float).reshape(6, 3)
        )
        norms = np.linalg.norm(self.directions, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError("Thruster direction vectors must be non-zero")
        object.__setattr__(self, "directions", self.directions / norms[:, None])

    @property
    def allocation_matrix(self) -> np.ndarray:
        forces = self.directions.T
        moments = np.cross(self.positions, self.directions).T
        return np.vstack([forces, moments])

    def forces_to_wrench(self, forces) -> np.ndarray:
        thrust = np.asarray(forces, dtype=float).reshape(6)
        return self.allocation_matrix @ thrust

    def normalized_action_to_forces(self, action) -> np.ndarray:
        normalized = np.clip(np.asarray(action, dtype=float).reshape(6), -1.0, 1.0)
        return normalized * float(self.max_force)

    def normalized_action_to_wrench(self, action) -> np.ndarray:
        return self.forces_to_wrench(self.normalized_action_to_forces(action))

    def allocate_wrench(self, desired_wrench) -> np.ndarray:
        desired = np.asarray(desired_wrench, dtype=float).reshape(6)
        thrust = np.linalg.pinv(self.allocation_matrix) @ desired
        thrust = np.clip(thrust, -float(self.max_force), float(self.max_force))
        return thrust / float(self.max_force)


def build_real_10kg_x_layout() -> ThrusterLayout:
    lx = 0.20
    ly = 0.175
    ly_vertical = 0.15
    c = 1.0 / np.sqrt(2.0)
    positions = np.array([
        [lx, -ly, 0.0],
        [lx, ly, 0.0],
        [-lx, -ly, 0.0],
        [-lx, ly, 0.0],
        [0.0, -ly_vertical, 0.0],
        [0.0, ly_vertical, 0.0],
    ])
    directions = np.array([
        [c, c, 0.0],
        [c, -c, 0.0],
        [c, -c, 0.0],
        [c, c, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])
    return ThrusterLayout(positions=positions, directions=directions, max_force=35.0)


def build_thruster_layout(name: str | None = None) -> ThrusterLayout:
    key = str(name or "real_10kg_x")
    if key in {"real_10kg_x", "real_10kg_v1", "x_layout"}:
        return build_real_10kg_x_layout()
    raise ValueError(f"Unknown thruster layout '{key}'")
