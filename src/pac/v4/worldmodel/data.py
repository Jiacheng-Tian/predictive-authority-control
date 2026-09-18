"""Window construction and feature assembly for the v4 world model.

Feature layout (32 dims, frozen contract shared by collection, training,
validation, and rollout):

* ``[0:12]``  state ``[eta(6), nu(6)]`` at the step start,
* ``[12:18]`` requested (blended) actuator command,
* ``[18:24]`` applied (post-actuator) command,
* ``[24:27]`` estimated current (causal),
* ``[27:31]`` nominal context ``[mass_scale, damping_scale, amp_scale, freq_scale]``,
* ``[31:32]`` normalized base scenario id.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

FEATURE_DIM = 32
STATE_DIM = 12


def assemble_wm_feature(
        state,
        requested,
        applied,
        est_current,
        context,
        scenario_id: int) -> np.ndarray:
    """Assemble one 32-dim world-model input feature vector."""
    state_array = np.asarray(state, dtype=np.float64).reshape(STATE_DIM)
    requested_array = np.asarray(requested, dtype=np.float64).reshape(6)
    applied_array = np.asarray(applied, dtype=np.float64).reshape(6)
    current_array = np.asarray(est_current, dtype=np.float64).reshape(3)
    context_array = np.asarray(context, dtype=np.float64).reshape(4)
    values = np.concatenate([
        state_array,
        requested_array,
        applied_array,
        current_array,
        context_array,
        np.array([int(scenario_id) / 3.0]),
    ])
    if values.shape != (FEATURE_DIM,):
        raise RuntimeError(
            f"world-model feature vector has dim {values.shape[0]}, expected {FEATURE_DIM}"
        )
    return values.astype(np.float32)


@dataclass(frozen=True)
class TransitionWindows:
    """Row-aligned training tensors for one world-model split."""

    windows: np.ndarray        # (N, L, 32) float32
    state: np.ndarray          # (N, 12) float32
    physics_next_state: np.ndarray  # (N, 12) float32
    next_state: np.ndarray     # (N, 12) float32
    current_delta: np.ndarray  # (N, 2) float32 horizontal true-current change
    episode_ids: np.ndarray    # (N,) int64 episode index per row

    @property
    def residual_target(self) -> np.ndarray:
        return (self.next_state - self.physics_next_state).astype(np.float32)

    def __len__(self) -> int:
        return int(self.windows.shape[0])


def _episode_boundaries(metadata) -> list[tuple[int, int, str]]:
    uids = metadata["episode_uid"].tolist()
    boundaries: list[tuple[int, int, str]] = []
    start = 0
    for index in range(1, len(uids) + 1):
        if index == len(uids) or uids[index] != uids[start]:
            boundaries.append((start, index, str(uids[start])))
            start = index
    return boundaries


def build_transition_windows(
        dataset,
        history_len: int,
        *,
        dtype=np.float32) -> TransitionWindows:
    """Build fixed-length input windows without crossing episode boundaries.

    ``dataset`` is a :class:`pac.v4.collector.TransitionArrays`.  Windows end
    at each collected row (inclusive), matching the one-step prediction task
    ``features[t-L+1:t+1] -> x_{t+1}``.
    """
    history_len = int(history_len)
    if history_len <= 0:
        raise ValueError("history_len must be positive")
    features = np.asarray(dataset.features, dtype=dtype)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(f"expected features with dim {FEATURE_DIM}")
    windows = np.zeros((features.shape[0], history_len, FEATURE_DIM), dtype=dtype)
    episode_ids = np.zeros(features.shape[0], dtype=np.int64)
    for episode_index, (start, end, _uid) in enumerate(_episode_boundaries(dataset.metadata)):
        episode = features[start:end]
        for offset in range(episode.shape[0]):
            recent = episode[max(0, offset - history_len + 1):offset + 1]
            windows[start + offset, -recent.shape[0]:, :] = recent
        episode_ids[start:end] = episode_index
    true_current = np.asarray(dataset.true_current, dtype=np.float32)
    next_true_current = np.asarray(dataset.next_true_current, dtype=np.float32)
    return TransitionWindows(
        windows=windows,
        state=np.asarray(dataset.state, dtype=np.float32),
        physics_next_state=np.asarray(dataset.physics_next_state, dtype=np.float32),
        next_state=np.asarray(dataset.next_state, dtype=np.float32),
        current_delta=(next_true_current[:, :2] - true_current[:, :2]).astype(np.float32),
        episode_ids=episode_ids,
    )


def windows_to_torch(split: TransitionWindows, indices=None) -> dict[str, torch.Tensor]:
    """Materialize window tensors as contiguous torch tensors."""
    if indices is None:
        return {
            "windows": torch.from_numpy(np.ascontiguousarray(split.windows)),
            "state": torch.from_numpy(np.ascontiguousarray(split.state)),
            "physics_next_state": torch.from_numpy(np.ascontiguousarray(split.physics_next_state)),
            "next_state": torch.from_numpy(np.ascontiguousarray(split.next_state)),
            "current_delta": torch.from_numpy(np.ascontiguousarray(split.current_delta)),
        }
    index_array = np.asarray(indices, dtype=np.int64)
    return {
        "windows": torch.from_numpy(np.ascontiguousarray(split.windows[index_array])),
        "state": torch.from_numpy(np.ascontiguousarray(split.state[index_array])),
        "physics_next_state": torch.from_numpy(np.ascontiguousarray(split.physics_next_state[index_array])),
        "next_state": torch.from_numpy(np.ascontiguousarray(split.next_state[index_array])),
        "current_delta": torch.from_numpy(np.ascontiguousarray(split.current_delta[index_array])),
    }


__all__ = [
    "FEATURE_DIM",
    "STATE_DIM",
    "TransitionWindows",
    "assemble_wm_feature",
    "build_transition_windows",
    "windows_to_torch",
]
