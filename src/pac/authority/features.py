"""PAC state and controller-context features."""

from __future__ import annotations

import numpy as np


def wrap_angle(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def phase_features(t: float, episode_end: float) -> np.ndarray:
    phase = float(np.clip(float(t) / max(float(episode_end), 1.0e-12), 0.0, 1.0))
    return np.asarray([
        phase,
        np.sin(2.0 * np.pi * phase),
        np.cos(2.0 * np.pi * phase),
    ], dtype=np.float32)


def build_alpha_feature(
        target,
        eta,
        nu,
        primary_action,
        authority_action,
        current_world,
        t: float,
        episode_end: float,
        feature_mode: str = "state_phase") -> np.ndarray:
    """Build the formal 24-dimensional PAC feature vector."""
    target = np.asarray(target, dtype=float).reshape(6)
    eta = np.asarray(eta, dtype=float).reshape(6)
    nu = np.asarray(nu, dtype=float).reshape(6)
    primary = np.asarray(primary_action, dtype=float).reshape(6)
    authority = np.asarray(authority_action, dtype=float).reshape(6)
    current = np.asarray(current_world, dtype=float).reshape(2)
    error = target - eta
    error[3:6] = [wrap_angle(value) for value in error[3:6]]
    delta_action = authority - primary
    primary_abs = np.abs(primary)
    authority_abs = np.abs(authority)
    values = [
        error[0], error[1], error[2],
        float(np.linalg.norm(error[:3])),
        nu[0], nu[1], nu[2],
        float(np.linalg.norm(nu[:3])),
        primary[0], primary[1], primary[2],
        float(np.linalg.norm(primary[:3])),
        float(np.max(primary_abs)),
        float(np.mean(primary_abs >= 0.95)),
        authority[0], authority[1], authority[2],
        float(np.linalg.norm(delta_action[:3])),
        float(np.max(authority_abs)),
        current[0], current[1],
    ]
    feature = np.asarray(values, dtype=np.float32)
    if feature_mode == "state_phase":
        return np.concatenate([feature, phase_features(t, episode_end)]).astype(np.float32)
    raise ValueError("formal PAC feature_mode must be state_phase")
