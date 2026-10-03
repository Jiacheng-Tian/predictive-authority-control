"""Warm-start replay materialization from the frozen stage-one dataset.

The stage-one transition dataset stores everything needed to rebuild the
RL warm-start replay offline:

* authority action = ``plan[:, 0, :]`` (verified to equal the MPC's applied
  action on every accepted step);
* primary action = un-blending ``requested = (1-a) primary + a authority``
  (valid for ``a < 1`` rows, ~97% of accepted transitions);
* the 24-dim alpha feature and its history windows are recomputed from the
  stored state/action/current arrays;
* rewards follow the frozen v1 definition with targets rebuilt from
  ``sample_time``.

The residual action stored for the critic warm start is
``a = clip(alpha_behavior - alpha_backbone, +-delta_max)`` — an
approximation of the deployed ``alpha = alpha_backbone + lambda * delta``
mapping that keeps the replay anchored to the behavior distribution.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from pac.authority.features import build_alpha_feature
from pac.evaluation.episodes import desired_heading
from pac.simulation.core import AUVSimulator
from pac.residual.rl.reward import compute_step_reward

_FEATURE_DIM = 24
_BLEND_EPSILON = 1.0e-6
_CHUNK = 4096


def _episode_slices(metadata):
    uids = metadata["episode_uid"].tolist()
    slices = []
    start = 0
    for index in range(1, len(uids) + 1):
        if index == len(uids) or uids[index] != uids[start]:
            slices.append((start, index))
            start = index
    return slices


def _feature_matrix(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the 24-dim features and per-row validity mask."""
    metadata = dataset.metadata
    rows = len(metadata)
    features = np.zeros((rows, _FEATURE_DIM), dtype=np.float32)
    valid = np.zeros(rows, dtype=bool)
    state = np.asarray(dataset.state, dtype=np.float64)
    requested = np.asarray(dataset.requested, dtype=np.float64)
    alpha = np.asarray(dataset.alpha, dtype=np.float64)
    plans = np.asarray(dataset.plan, dtype=np.float64)
    est_current = np.asarray(dataset.est_current, dtype=np.float64)
    accepted = metadata["plan_accepted"].astype(bool).to_numpy()
    scenario = metadata["scenario_id"].to_numpy()
    step = metadata["step"].to_numpy()
    for begin, end in _episode_slices(metadata):
        episode_steps = int(end - begin)
        episode_end_time = float(episode_steps) * 0.01
        for offset in range(episode_steps):
            row = begin + offset
            if not accepted[row] or alpha[row] > 1.0 - _BLEND_EPSILON:
                continue
            authority_action = plans[row, 0, :]
            primary_action = (
                requested[row] - alpha[row] * authority_action
            ) / (1.0 - alpha[row])
            stamp = float(step[row]) * 0.01
            target = AUVSimulator._get_target(stamp)
            features[row] = build_alpha_feature(
                target,
                state[row, :6],
                state[row, 6:],
                primary_action,
                authority_action,
                est_current[row, :2],
                stamp,
                episode_end_time,
                "state_phase",
            )
            valid[row] = True
    return features, valid


def _build_windows(features: np.ndarray, metadata, history_len: int) -> np.ndarray:
    windows = np.zeros(
        (features.shape[0], int(history_len), _FEATURE_DIM), dtype=np.float32
    )
    for begin, end in _episode_slices(metadata):
        for offset in range(end - begin):
            row = begin + offset
            window_start = begin + max(0, offset - int(history_len) + 1)
            recent = features[window_start:row + 1]
            windows[row, -len(recent):] = recent
    return windows


def _rewards(dataset, reward_config) -> np.ndarray:
    metadata = dataset.metadata
    rows = len(metadata)
    next_state = np.asarray(dataset.next_state, dtype=np.float64)
    applied = np.asarray(dataset.applied, dtype=np.float64)
    requested = np.asarray(dataset.requested, dtype=np.float64)
    alpha = np.asarray(dataset.alpha, dtype=np.float64)
    solver = np.asarray(dataset.solver, dtype=np.float64)
    rewards = np.zeros(rows, dtype=np.float64)
    for begin, end in _episode_slices(metadata):
        previous_applied = np.zeros(6, dtype=np.float64)
        previous_alpha = 0.0
        for offset in range(end - begin):
            row = begin + offset
            sample_time = float(metadata["sample_time"].iloc[row])
            target = AUVSimulator._get_target(sample_time)
            position_error = float(np.linalg.norm(
                target[:3] - next_state[row, :3]
            ))
            heading_error = abs(float(
                (desired_heading(sample_time) - next_state[row, 5] + np.pi)
                % (2.0 * np.pi) - np.pi
            ))
            saturation = float(np.mean(
                np.abs(requested[row] - applied[row]) > 1.0e-12
            ))
            rewards[row] = compute_step_reward(
                reward_config,
                position_error=position_error,
                heading_error=heading_error,
                applied_action=applied[row],
                previous_applied=previous_applied,
                alpha=float(alpha[row]),
                previous_alpha=previous_alpha,
                saturation_fraction=saturation,
                deadline_missed=bool(solver[row, 2] > 0.5),
                constraint_violated=position_error > 20.0,
            )
            previous_applied = applied[row]
            previous_alpha = float(alpha[row])
    return rewards


def materialize_warmstart(
        dataset,
        config,
        backbone,
        *,
        limit: int | None = None) -> dict[str, Any]:
    """Build the warm-start replay arrays from the frozen dataset."""
    history_len = int(config.authority_model.history_len)
    features, valid = _feature_matrix(dataset)
    windows = _build_windows(features, dataset.metadata, history_len)
    rewards = _rewards(dataset, reward_config=config.reward)

    backbone.eval()
    backbone_alphas = np.zeros(features.shape[0], dtype=np.float64)
    with torch.no_grad():
        for start in range(0, windows.shape[0], _CHUNK):
            chunk = torch.from_numpy(windows[start:start + _CHUNK])
            backbone_alphas[start:start + _CHUNK] = (
                backbone(chunk).numpy().astype(np.float64)
            )

    alpha = np.asarray(dataset.alpha, dtype=np.float64)
    delta_max = float(config.rl.delta_max)
    actions = np.clip(alpha - backbone_alphas, -delta_max, delta_max)

    dones = np.zeros(features.shape[0], dtype=bool)
    for begin, end in _episode_slices(dataset.metadata):
        dones[end - 1] = True
    next_windows = np.empty_like(windows)
    next_windows[:-1] = windows[1:]
    next_windows[-1] = windows[-1]
    for begin, end in _episode_slices(dataset.metadata):
        next_windows[end - 1] = windows[end - 1]

    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        raise ValueError("no valid warm-start transitions (no accepted, alpha<1 rows)")
    if limit is not None and valid_indices.size > int(limit):
        stride = max(1, valid_indices.size // int(limit))
        valid_indices = valid_indices[::stride][: int(limit)]
    return {
        "windows": windows[valid_indices],
        "actions": actions[valid_indices].astype(np.float32),
        "rewards": rewards[valid_indices].astype(np.float32),
        "next_windows": next_windows[valid_indices],
        "dones": dones[valid_indices],
        "backbone_alphas": backbone_alphas[valid_indices],
        "behavior_alphas": alpha[valid_indices],
        "valid_mask": valid,
        "valid_rows": int(valid_indices.size),
        "total_rows": int(features.shape[0]),
        "delta_max": delta_max,
    }


__all__ = ["materialize_warmstart"]
