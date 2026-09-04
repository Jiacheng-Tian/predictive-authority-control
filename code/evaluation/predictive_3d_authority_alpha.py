"""Predict 3D authority alpha from state/context features.

The teacher is the validated smooth startup handover. The learned policy
predicts only the scalar authority coefficient, preserving the SMC primary
controller plus sparse authority mainline.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from env.auv_env import AUVTrackingEnv
from evaluation.diagnose_3d_authority import (
    DEFAULT_WINDOWS,
    SCENARIO_NAMES,
    _add_handover_gain_columns,
    _aggregate_handover_metrics,
    _parse_csv_floats,
    _parse_csv_ints,
    _resolve_project_path,
    authority_blend_alpha,
    compute_axis_window_metrics,
    summarize_handover_windows,
)
from evaluation.current_control import _compute_metrics, _desired_heading
from evaluation.current_control import (
    _build_base_controller,
    _compute_base_action,
)
from evaluation.engineering_metrics import compute_timeseries_engineering_metrics


def _wrap_angle(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _phase_features(t: float, episode_end: float) -> np.ndarray:
    phase = float(np.clip(float(t) / max(float(episode_end), 1.0e-12), 0.0, 1.0))
    return np.asarray([
        phase,
        np.sin(2.0 * np.pi * phase),
        np.cos(2.0 * np.pi * phase),
    ], dtype=np.float32)


def teacher_alpha_for_scenario(scenario) -> float:
    """Return scenario-level alpha labels from the real10kg alpha sweep."""
    if isinstance(scenario, str):
        key = str(scenario).strip().lower()
        if key == "constant":
            return 1.0
        if key == "sinusoidal":
            return 0.5
        if key in {"step_change", "step-change", "step"}:
            return 1.0
    scenario_id = int(scenario)
    if scenario_id == 2:
        return 0.5
    return 1.0


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
    """Build compact 3D state/context features for authority-alpha prediction."""
    target = np.asarray(target, dtype=float).reshape(6)
    eta = np.asarray(eta, dtype=float).reshape(6)
    nu = np.asarray(nu, dtype=float).reshape(6)
    primary = np.asarray(primary_action, dtype=float).reshape(6)
    authority = np.asarray(authority_action, dtype=float).reshape(6)
    current = np.asarray(current_world, dtype=float).reshape(2)
    error = target - eta
    error[3:6] = [_wrap_angle(value) for value in error[3:6]]
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
    mode = str(feature_mode or "state_phase").strip().lower()
    if mode == "state":
        return feature
    if mode == "state_phase":
        return np.concatenate(
            [feature, _phase_features(t, episode_end)], axis=0
        ).astype(np.float32)
    raise ValueError(f"Unknown feature_mode: {feature_mode}")


class AlphaMLP(nn.Module):
    """Small MLP that predicts bounded authority alpha."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(int(input_dim)))
        self.register_buffer("feature_scale", torch.ones(int(input_dim)))
        drop = nn.Dropout(p=float(np.clip(dropout, 0.0, 0.95)))
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            drop,
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            drop,
            nn.Linear(int(hidden_dim), 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = (x - self.feature_mean) / self.feature_scale.clamp_min(1.0e-6)
        return self.net(x).squeeze(-1)


class TemporalAlphaTransformer(nn.Module):
    """Causal history encoder that predicts bounded authority alpha."""

    def __init__(
            self,
            input_dim: int,
            history_len: int = 100,
            embed_dim: int = 64,
            num_heads: int = 4,
            num_layers: int = 2,
            dropout: float = 0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.history_len = int(history_len)
        self.register_buffer("feature_mean", torch.zeros(self.input_dim))
        self.register_buffer("feature_scale", torch.ones(self.input_dim))
        self.input_proj = nn.Linear(self.input_dim, int(embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.history_len, int(embed_dim)))
        layer = nn.TransformerEncoderLayer(
            d_model=int(embed_dim),
            nhead=int(num_heads),
            dim_feedforward=int(embed_dim) * 4,
            dropout=float(np.clip(dropout, 0.0, 0.95)),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.head = nn.Sequential(
            nn.LayerNorm(int(embed_dim)),
            nn.Linear(int(embed_dim), 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("TemporalAlphaTransformer expects [batch, history, features]")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected feature dim {self.input_dim}, got {x.shape[-1]}")
        if x.shape[1] > self.history_len:
            x = x[:, -self.history_len:, :]
        x = (x - self.feature_mean) / self.feature_scale.clamp_min(1.0e-6)
        emb = self.input_proj(x)
        emb = emb + self.pos_embedding[:, -emb.shape[1]:, :]
        encoded = self.encoder(emb)
        return self.head(encoded[:, -1, :]).squeeze(-1)


def build_temporal_feature_window(history, history_len: int) -> np.ndarray:
    """Return a fixed-length history window with zero left padding."""
    if len(history) == 0:
        raise ValueError("history must contain at least one feature vector")
    arrays = [np.asarray(item, dtype=np.float32).reshape(-1) for item in history]
    feature_dim = int(arrays[-1].shape[0])
    window_len = int(history_len)
    if window_len <= 0:
        raise ValueError("history_len must be positive")
    recent = arrays[-window_len:]
    window = np.zeros((window_len, feature_dim), dtype=np.float32)
    window[-len(recent):, :] = np.asarray(recent, dtype=np.float32)
    return window


class TemporalFeatureWindowDataset(Dataset):
    """Lazily builds fixed history windows without crossing episode ids."""

    def __init__(
            self,
            features: np.ndarray,
            labels: np.ndarray,
            weights: np.ndarray,
            episode_ids: np.ndarray,
            history_len: int):
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.weights = np.asarray(weights, dtype=np.float32)
        self.episode_ids = np.asarray(episode_ids, dtype=np.int64)
        self.history_len = int(history_len)
        if self.features.ndim != 2:
            raise ValueError("features must be a 2D array")
        if self.history_len <= 0:
            raise ValueError("history_len must be positive")
        sample_count = int(self.features.shape[0])
        for name, array in {
                "labels": self.labels,
                "weights": self.weights,
                "episode_ids": self.episode_ids,
        }.items():
            if int(array.shape[0]) != sample_count:
                raise ValueError(
                    f"{name} length {array.shape[0]} does not match features {sample_count}"
                )

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        idx = int(index)
        episode_id = int(self.episode_ids[idx])
        start = idx
        lower_bound = max(0, idx - self.history_len + 1)
        while start > lower_bound and int(self.episode_ids[start - 1]) == episode_id:
            start -= 1
        window = build_temporal_feature_window(
            self.features[start:idx + 1],
            self.history_len,
        )
        return (
            torch.from_numpy(window),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            torch.tensor(self.weights[idx], dtype=torch.float32),
        )


def teacher_episode_ids(label_table: pd.DataFrame, sample_count: int) -> np.ndarray:
    """Build contiguous episode ids from teacher-label table metadata."""
    if label_table is None or label_table.empty:
        return np.zeros(int(sample_count), dtype=np.int64)
    key_cols = [
        "scenario",
        "seed",
        "start_time",
        "current_amplitude_scale",
        "current_frequency_scale",
    ]
    if not all(col in label_table.columns for col in key_cols):
        return np.zeros(int(sample_count), dtype=np.int64)
    keys = label_table[key_cols].copy()
    changed = keys.ne(keys.shift()).any(axis=1)
    ids = changed.cumsum().to_numpy(dtype=np.int64) - 1
    if ids.shape[0] != int(sample_count):
        raise ValueError(
            f"teacher label table length {ids.shape[0]} does not match samples {sample_count}"
        )
    return ids


def precompute_temporal_windows(
        features: np.ndarray,
        episode_ids: np.ndarray,
        history_len: int) -> np.ndarray:
    """Precompute fixed temporal windows without crossing episode boundaries."""
    feature_array = np.asarray(features, dtype=np.float32)
    ids = np.asarray(episode_ids, dtype=np.int64)
    if feature_array.ndim != 2:
        raise ValueError("features must be a 2D array")
    if ids.shape[0] != feature_array.shape[0]:
        raise ValueError("episode_ids must match feature sample count")
    window_len = int(history_len)
    if window_len <= 0:
        raise ValueError("history_len must be positive")
    sample_count, feature_dim = feature_array.shape
    windows = np.zeros((sample_count, window_len, feature_dim), dtype=np.float32)
    start = 0
    while start < sample_count:
        episode_id = int(ids[start])
        end = start + 1
        while end < sample_count and int(ids[end]) == episode_id:
            end += 1
        episode = feature_array[start:end]
        for local_idx in range(episode.shape[0]):
            recent = episode[max(0, local_idx - window_len + 1):local_idx + 1]
            windows[start + local_idx, -recent.shape[0]:, :] = recent
        start = end
    return windows


def save_teacher_cache(
        cache_path: str | Path,
        *,
        features: np.ndarray,
        labels: np.ndarray,
        label_table: pd.DataFrame,
        metadata: dict) -> Path:
    """Persist collected teacher labels for reuse across PAC architectures."""
    path = _resolve_project_path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table_csv = label_table.to_csv(index=False)
    np.savez_compressed(
        path,
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        label_table_csv=np.asarray(table_csv),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return path


def load_teacher_cache(
        cache_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    """Load a teacher cache written by :func:`save_teacher_cache`."""
    path = _resolve_project_path(cache_path)
    with np.load(path, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.float32)
        table_csv = str(data["label_table_csv"].item())
        metadata = json.loads(str(data["metadata_json"].item()))
    table = pd.read_csv(StringIO(table_csv))
    return features, labels, table, metadata


def limit_teacher_training_samples(
        features: np.ndarray,
        labels: np.ndarray,
        label_table: pd.DataFrame,
        max_samples: int,
        seed: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Deterministically cap teacher samples while retaining positive labels."""
    limit = int(max_samples)
    feature_array = np.asarray(features, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.float32)
    if limit <= 0 or limit >= int(label_array.shape[0]):
        return feature_array, label_array, label_table.reset_index(drop=True).copy()
    rng = np.random.default_rng(int(seed))
    positive = np.flatnonzero(label_array > 1.0e-6)
    non_positive = np.flatnonzero(label_array <= 1.0e-6)
    positive_keep = positive
    if positive_keep.shape[0] > limit:
        positive_keep = rng.choice(positive_keep, size=limit, replace=False)
        non_positive_keep = np.asarray([], dtype=np.int64)
    else:
        remaining = max(0, limit - int(positive_keep.shape[0]))
        if remaining > 0 and non_positive.shape[0] > 0:
            sample_size = min(remaining, int(non_positive.shape[0]))
            non_positive_keep = rng.choice(
                non_positive,
                size=sample_size,
                replace=False,
            )
        else:
            non_positive_keep = np.asarray([], dtype=np.int64)
    indices = np.sort(np.concatenate([positive_keep, non_positive_keep]).astype(np.int64))
    return (
        feature_array[indices],
        label_array[indices],
        label_table.iloc[indices].reset_index(drop=True),
    )


def apply_uncertainty_adjustment(
        alpha_mean: float,
        alpha_std: float,
        mode: str = "none",
        gain: float = 0.0,
        threshold: float = 1.0) -> float:
    """Reduce authority allocation when model uncertainty is high."""
    alpha = float(np.clip(alpha_mean, 0.0, 1.0))
    std = max(0.0, float(alpha_std))
    mode_key = str(mode or "none").strip().lower()
    if mode_key in {"none", "off"}:
        return alpha
    if mode_key == "threshold" and std > float(threshold):
        return 0.0
    if mode_key == "cap" and std > float(threshold):
        return float(min(alpha, max(0.0, float(gain))))
    if mode_key == "exponential":
        excess = max(0.0, std - float(threshold))
        return float(alpha * np.exp(-float(gain) * excess))
    raise ValueError(f"Unknown uncertainty mode: {mode}")


def predict_alpha_with_uncertainty(
        model: nn.Module,
        feature: torch.Tensor,
        samples: int = 1,
        mode: str = "none") -> tuple[float, float]:
    """Predict alpha mean/std using optional MC dropout sampling."""
    sample_count = max(1, int(samples))
    uncertainty_mode = str(mode or "none").strip().lower()
    was_training = bool(model.training)
    preds = []
    if uncertainty_mode != "none" and sample_count > 1:
        model.train()
        with torch.no_grad():
            for _ in range(sample_count):
                preds.append(model(feature).detach().float().reshape(-1))
    else:
        model.eval()
        with torch.no_grad():
            preds.append(model(feature).detach().float().reshape(-1))
    model.train(was_training)
    values = torch.cat(preds)
    return float(values.mean().item()), float(values.std(unbiased=False).item())


def blend_actions_with_predicted_alpha(
        primary_action,
        authority_action,
        alpha: float) -> tuple[np.ndarray, dict[str, float]]:
    """Blend two normalized actions with a bounded predicted authority alpha."""
    primary = np.asarray(primary_action, dtype=float).reshape(6)
    authority = np.asarray(authority_action, dtype=float).reshape(6)
    bounded_alpha = float(np.clip(float(alpha), 0.0, 1.0))
    action = (1.0 - bounded_alpha) * primary + bounded_alpha * authority
    return np.clip(action, -1.0, 1.0), {
        "authority_alpha": bounded_alpha,
        "authority_active": float(bounded_alpha > 1.0e-4),
    }


def filter_authority_alpha(
        raw_alpha: float,
        previous_alpha: float,
        smoothing: float = 0.0,
        rate_limit: float = 1.0,
        deadband: float = 0.0) -> float:
    """Apply causal temporal consistency to the scalar authority alpha."""
    raw = float(np.clip(raw_alpha, 0.0, 1.0))
    prev = float(np.clip(previous_alpha, 0.0, 1.0))
    smooth = float(np.clip(smoothing, 0.0, 0.999))
    candidate = smooth * prev + (1.0 - smooth) * raw
    limit = max(0.0, float(rate_limit))
    if limit < 1.0:
        candidate = prev + float(np.clip(candidate - prev, -limit, limit))
    if candidate < float(deadband):
        candidate = 0.0
    return float(np.clip(candidate, 0.0, 1.0))


def _tracking_cost(
        target,
        eta,
        action,
        action_delta=None,
        action_saturation_weight: float = 0.02,
        action_delta_weight: float = 0.01) -> float:
    target = np.asarray(target, dtype=float).reshape(6)
    eta = np.asarray(eta, dtype=float).reshape(6)
    error = target - eta
    error[3:6] = [_wrap_angle(value) for value in error[3:6]]
    action_arr = np.asarray(action, dtype=float).reshape(6)
    delta = (
        np.asarray(action_delta, dtype=float).reshape(6)
        if action_delta is not None else np.zeros(6, dtype=float)
    )
    xy_cost = float(np.dot(error[:2], error[:2]))
    z_cost = 0.7 * float(error[2] ** 2)
    yaw_cost = 0.08 * float(error[5] ** 2)
    sat_cost = float(action_saturation_weight) * float(
        np.mean(np.maximum(np.abs(action_arr) - 0.92, 0.0) ** 2)
    )
    delta_cost = float(action_delta_weight) * float(np.mean(delta ** 2))
    return xy_cost + z_cost + yaw_cost + sat_cost + delta_cost


def choose_oracle_alpha(
        *,
        target,
        eta,
        nu,
        primary_action,
        authority_action,
        current,
        action_to_wrench,
        dynamics_stepper,
        alpha_grid,
        previous_action=None,
        previous_alpha: float = 0.0,
        action_saturation_weight: float = 0.02,
        action_delta_weight: float = 0.01,
        alpha_delta_weight: float = 0.0) -> tuple[float, dict[str, float]]:
    """Choose alpha by one-step predicted tracking cost over a fixed grid."""
    grid = np.asarray(alpha_grid, dtype=float).reshape(-1)
    if grid.size == 0:
        raise ValueError("alpha_grid must contain at least one value")
    primary = np.asarray(primary_action, dtype=float).reshape(6)
    authority = np.asarray(authority_action, dtype=float).reshape(6)
    prev = (
        np.asarray(previous_action, dtype=float).reshape(6)
        if previous_action is not None else primary
    )
    rows = []
    for raw_alpha in grid:
        alpha = float(np.clip(raw_alpha, 0.0, 1.0))
        action, _info = blend_actions_with_predicted_alpha(primary, authority, alpha)
        wrench = action_to_wrench(action)
        pred_eta, pred_nu = dynamics_stepper(
            np.asarray(eta, dtype=float).copy(),
            np.asarray(nu, dtype=float).copy(),
            wrench,
            current,
        )
        del pred_nu
        cost = _tracking_cost(
            target,
            pred_eta,
            action,
            action_delta=action - prev,
            action_saturation_weight=action_saturation_weight,
            action_delta_weight=action_delta_weight,
        )
        cost += float(alpha_delta_weight) * abs(alpha - float(previous_alpha))
        rows.append((cost, alpha))
    rows.sort(key=lambda item: (item[0], item[1]))
    best_cost, best_alpha = rows[0]
    return float(best_alpha), {
        "candidate_count": int(grid.size),
        "best_cost": float(best_cost),
        "worst_cost": float(rows[-1][0]),
        "alpha_delta_weight": float(alpha_delta_weight),
    }


def window_alpha_schedule_from_candidates(candidates: pd.DataFrame) -> dict[str, float]:
    """Choose the fixed alpha that minimizes 3D RMSE inside each evaluation window."""
    required = {"window", "fixed_alpha", "rmse_3d"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"window alpha candidates missing columns: {sorted(missing)}")
    if candidates.empty:
        return {}
    data = candidates.copy()
    data["fixed_alpha"] = pd.to_numeric(data["fixed_alpha"], errors="raise")
    data["rmse_3d"] = pd.to_numeric(data["rmse_3d"], errors="raise")
    summary = (
        data.groupby(["window", "fixed_alpha"], as_index=False)["rmse_3d"]
        .mean()
        .sort_values(["window", "rmse_3d", "fixed_alpha"])
    )
    best = summary.groupby("window", as_index=False).first()
    return {
        str(row["window"]): float(row["fixed_alpha"])
        for _, row in best.iterrows()
    }


def window_oracle_label_at_time(
        t: float,
        schedule: dict[str, float],
        windows: list[tuple[str, float, float]] | None = None) -> float:
    """Return the window-oracle alpha label at relative time ``t``."""
    windows = list(DEFAULT_WINDOWS if windows is None else windows)
    if not windows:
        return 0.0
    time_value = float(t)
    for name, start, end in windows:
        if float(start) <= time_value < float(end):
            return float(schedule.get(str(name), 0.0))
    if time_value < float(windows[0][1]):
        return float(schedule.get(str(windows[0][0]), 0.0))
    return float(schedule.get(str(windows[-1][0]), 0.0))


def schedule_alpha_at_time(
        t: float,
        schedule: dict[str, float],
        windows: list[tuple[str, float, float]] | None = None) -> float:
    """Return alpha from a named evaluation-window schedule."""
    return window_oracle_label_at_time(t, schedule, windows=windows)


def calibrate_alpha_with_bias_schedule(
        alpha: float,
        t: float,
        bias_schedule: dict[str, float] | None,
        windows: list[tuple[str, float, float]] | None = None) -> float:
    """Apply an evaluation-window bias schedule to a model alpha prediction."""
    value = float(alpha)
    if bias_schedule:
        value += schedule_alpha_at_time(t, bias_schedule, windows=windows)
    return float(np.clip(value, 0.0, 1.0))


def hybrid_oracle_label(
        t: float,
        base_label: float,
        schedule: dict[str, float],
        floor_alpha: float = 0.75) -> float:
    """Apply conservative window-aware alpha floors to the short-horizon oracle."""
    window_name = window_oracle_window_at_time(t)
    schedule_alpha = float(schedule.get(window_name, 0.0))
    label = float(base_label)
    if window_name in {"pre_step_3_10s", "post_step_13_20p9s"} and schedule_alpha >= float(floor_alpha):
        label = max(label, float(floor_alpha))
    return float(np.clip(label, 0.0, 1.0))


def window_oracle_window_at_time(
        t: float,
        windows: list[tuple[str, float, float]] | None = None) -> str:
    """Return the evaluation-window name used for a relative time."""
    windows = list(DEFAULT_WINDOWS if windows is None else windows)
    if not windows:
        return ""
    time_value = float(t)
    for name, start, end in windows:
        if float(start) <= time_value < float(end):
            return str(name)
    if time_value < float(windows[0][1]):
        return str(windows[0][0])
    return str(windows[-1][0])


def build_window_oracle_alpha_schedule(
        *,
        scenario: int,
        seed: int,
        steps: int,
        mass_scale_xy: float,
        damping_scale_xy: float,
        current_amplitude_scale: float,
        current_frequency_scale: float,
        vertical_current: float,
        primary_controller: str,
        authority_controller: str,
        feature_mode: str,
        alpha_grid: np.ndarray,
        alpha_gain: float,
        vehicle_profile: str,
        action_mode: str,
        thruster_layout: str,
        start_time: float = 0.0) -> tuple[dict[str, float], pd.DataFrame]:
    """Build per-window fixed-alpha labels from full-episode sweep results."""
    candidate_frames = []
    for raw_alpha in np.asarray(alpha_grid, dtype=float).reshape(-1):
        metrics = run_predictive_alpha_episode(
            model=None,
            scenario=int(scenario),
            seed=int(seed),
            steps=int(steps),
            mass_scale_xy=float(mass_scale_xy),
            damping_scale_xy=float(damping_scale_xy),
            current_amplitude_scale=float(current_amplitude_scale),
            current_frequency_scale=float(current_frequency_scale),
            vertical_current=float(vertical_current),
            primary_controller=str(primary_controller),
            authority_controller=str(authority_controller),
            feature_mode=str(feature_mode),
            start_time=float(start_time),
            fixed_alpha=float(raw_alpha),
            alpha_gain=float(alpha_gain),
            alpha_smoothing=0.0,
            alpha_rate_limit=1.0,
            alpha_deadband=0.0,
            policy_architecture="fixed_alpha",
            history_len=1,
            vehicle_profile=str(vehicle_profile),
            action_mode=str(action_mode),
            thruster_layout=str(thruster_layout),
            save_ts=True,
        )
        window = compute_axis_window_metrics(metrics["ts"], dt=0.01)
        window["fixed_alpha"] = float(raw_alpha)
        candidate_frames.append(window)
    candidates = (
        pd.concat(candidate_frames, ignore_index=True)
        if candidate_frames else pd.DataFrame(columns=["window", "fixed_alpha", "rmse_3d"])
    )
    return window_alpha_schedule_from_candidates(candidates), candidates


def scheduled_alpha_candidates_from_window_schedule(
        window_schedule: dict[str, float],
        alpha_grid: np.ndarray) -> list[dict[str, float]]:
    """Return compact schedule candidates for episode-level teacher selection."""
    window_names = [str(item[0]) for item in DEFAULT_WINDOWS]
    candidates: list[dict[str, float]] = []

    def add(schedule: dict[str, float]):
        normalized = {
            name: float(np.clip(schedule.get(name, 0.0), 0.0, 1.0))
            for name in window_names
        }
        if normalized not in candidates:
            candidates.append(normalized)

    add(window_schedule)
    conservative = dict(window_schedule)
    for name in ("step_recovery_10_13s",):
        conservative[name] = min(float(conservative.get(name, 0.0)), 0.25)
    add(conservative)
    for alpha in np.asarray(alpha_grid, dtype=float).reshape(-1):
        add({name: float(alpha) for name in window_names})
    return candidates


def build_schedule_oracle_alpha_schedule(
        *,
        scenario: int,
        seed: int,
        steps: int,
        mass_scale_xy: float,
        damping_scale_xy: float,
        current_amplitude_scale: float,
        current_frequency_scale: float,
        vertical_current: float,
        primary_controller: str,
        authority_controller: str,
        feature_mode: str,
        alpha_grid: np.ndarray,
        alpha_gain: float,
        alpha_smoothing: float,
        alpha_rate_limit: float,
        vehicle_profile: str,
        action_mode: str,
        thruster_layout: str,
        start_time: float = 0.0) -> tuple[dict[str, float], pd.DataFrame]:
    """Choose an alpha schedule by whole-episode rollout cost."""
    window_schedule, window_candidates = build_window_oracle_alpha_schedule(
        scenario=scenario,
        seed=seed,
        steps=steps,
        mass_scale_xy=mass_scale_xy,
        damping_scale_xy=damping_scale_xy,
        current_amplitude_scale=current_amplitude_scale,
        current_frequency_scale=current_frequency_scale,
        vertical_current=vertical_current,
        primary_controller=primary_controller,
        authority_controller=authority_controller,
        feature_mode=feature_mode,
        alpha_grid=alpha_grid,
        alpha_gain=1.0,
        vehicle_profile=vehicle_profile,
        action_mode=action_mode,
        thruster_layout=thruster_layout,
        start_time=start_time,
    )
    rows = []
    for idx, candidate in enumerate(scheduled_alpha_candidates_from_window_schedule(
            window_schedule,
            alpha_grid,
    )):
        metrics = run_predictive_alpha_episode(
            model=None,
            scenario=int(scenario),
            seed=int(seed),
            steps=int(steps),
            mass_scale_xy=float(mass_scale_xy),
            damping_scale_xy=float(damping_scale_xy),
            current_amplitude_scale=float(current_amplitude_scale),
            current_frequency_scale=float(current_frequency_scale),
            vertical_current=float(vertical_current),
            primary_controller=str(primary_controller),
            authority_controller=str(authority_controller),
            feature_mode=str(feature_mode),
            start_time=float(start_time),
            alpha_schedule=candidate,
            alpha_gain=float(alpha_gain),
            alpha_smoothing=float(alpha_smoothing),
            alpha_rate_limit=float(alpha_rate_limit),
            alpha_deadband=0.0,
            policy_architecture="scheduled_alpha",
            history_len=1,
            vehicle_profile=str(vehicle_profile),
            action_mode=str(action_mode),
            thruster_layout=str(thruster_layout),
            save_ts=False,
        )
        row = {"candidate_id": int(idx), **candidate, **_scalar_metrics(metrics)}
        rows.append(row)
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return window_schedule, window_candidates
    best = candidates.sort_values(["rmse", "candidate_id"]).iloc[0]
    schedule = {
        name: float(best[name])
        for name, _start, _end in DEFAULT_WINDOWS
    }
    candidates["source_window_candidate_count"] = int(len(window_candidates))
    return schedule, candidates


def collect_alpha_teacher_dataset(
        scenarios: list[int],
        seeds: list[int],
        steps: int,
        mass_scale_xy: float,
        damping_scale_xy: float,
        current_amplitude_scale: float,
        current_frequency_scale: float,
        vertical_current: float,
        primary_controller: str,
        authority_controller: str,
        authority_until: float,
        blend_duration: float,
        blend_curve: str,
        feature_mode: str,
        teacher_mode: str = "handover",
        oracle_alpha_grid: str = "0,0.25,0.5,0.75,1",
        oracle_horizon_steps: int = 10,
        oracle_action_saturation_weight: float = 0.02,
        oracle_action_delta_weight: float = 0.01,
        oracle_alpha_delta_weight: float = 0.0,
        window_oracle_alpha_gain: float = 1.0,
        vehicle_profile: str = "real_10kg_v1",
        action_mode: str = "wrench",
        thruster_layout: str = "real_10kg_x",
        start_times: list[float] | None = None,
        current_amplitude_scales: list[float] | None = None,
        current_frequency_scales: list[float] | None = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Collect state features with smooth-handover teacher alpha labels."""
    features, labels, rows = [], [], []
    episode_end = float(steps) * 0.01
    alpha_grid = np.asarray(_parse_csv_floats(oracle_alpha_grid), dtype=float)
    start_grid = (
        [float(v) for v in start_times]
        if start_times is not None else [0.0]
    )
    amp_grid = (
        [float(v) for v in current_amplitude_scales]
        if current_amplitude_scales is not None
        else [float(current_amplitude_scale)]
    )
    freq_grid = (
        [float(v) for v in current_frequency_scales]
        if current_frequency_scales is not None
        else [float(current_frequency_scale)]
    )
    for scenario in scenarios:
        for seed in seeds:
            for start_time in start_grid:
                for amp_scale in amp_grid:
                    for freq_scale in freq_grid:
                        mode = str(teacher_mode or "handover").strip().lower()
                        window_alpha_schedule = {}
                        if mode in {"window_oracle", "hybrid_oracle"}:
                            window_alpha_schedule, _window_candidates = build_window_oracle_alpha_schedule(
                                scenario=int(scenario),
                                seed=int(seed),
                                steps=int(steps),
                                mass_scale_xy=float(mass_scale_xy),
                                damping_scale_xy=float(damping_scale_xy),
                                current_amplitude_scale=float(amp_scale),
                                current_frequency_scale=float(freq_scale),
                                vertical_current=float(vertical_current),
                                primary_controller=str(primary_controller),
                                authority_controller=str(authority_controller),
                                feature_mode=str(feature_mode),
                                alpha_grid=alpha_grid,
                                alpha_gain=float(window_oracle_alpha_gain),
                                vehicle_profile=str(vehicle_profile),
                                action_mode=str(action_mode),
                                thruster_layout=str(thruster_layout),
                                start_time=float(start_time),
                            )
                        elif mode == "schedule_oracle":
                            window_alpha_schedule, _schedule_candidates = build_schedule_oracle_alpha_schedule(
                                scenario=int(scenario),
                                seed=int(seed),
                                steps=int(steps),
                                mass_scale_xy=float(mass_scale_xy),
                                damping_scale_xy=float(damping_scale_xy),
                                current_amplitude_scale=float(amp_scale),
                                current_frequency_scale=float(freq_scale),
                                vertical_current=float(vertical_current),
                                primary_controller=str(primary_controller),
                                authority_controller=str(authority_controller),
                                feature_mode=str(feature_mode),
                                alpha_grid=alpha_grid,
                                alpha_gain=float(window_oracle_alpha_gain),
                                alpha_smoothing=0.5,
                                alpha_rate_limit=0.0125,
                                vehicle_profile=str(vehicle_profile),
                                action_mode=str(action_mode),
                                thruster_layout=str(thruster_layout),
                                start_time=float(start_time),
                            )
                        env = AUVTrackingEnv(
                            scenario=int(scenario),
                            max_steps=int(steps),
                            trajectory3d=True,
                            start_time=float(start_time),
                            mass_scale_xy=float(mass_scale_xy),
                            damping_scale_xy=float(damping_scale_xy),
                            current_amplitude_scale=float(amp_scale),
                            current_frequency_scale=float(freq_scale),
                            vertical_current=float(vertical_current),
                            vehicle_profile=str(vehicle_profile),
                            action_mode=str(action_mode),
                            thruster_layout=str(thruster_layout),
                        )
                        primary_name, primary = _build_base_controller(primary_controller)
                        authority_name, authority = _build_base_controller(authority_controller)
                        for controller in (primary, authority):
                            if hasattr(controller, "reset"):
                                controller.reset()
                            if hasattr(controller, "set_trajectory3d"):
                                controller.set_trajectory3d(True)
                        env.reset(seed=int(seed))
                        done = False
                        prev_action = np.zeros(6, dtype=float)
                        prev_alpha = 0.0

                        def action_to_wrench(action):
                            return env._action_to_wrench(action)[0]

                        def dynamics_stepper(eta0, nu0, wrench, current):
                            eta_save = env.dynamics.eta.copy()
                            nu_save = env.dynamics.nu.copy()
                            env.dynamics.eta = np.asarray(eta0, dtype=float).copy()
                            env.dynamics.nu = np.asarray(nu0, dtype=float).copy()
                            pred_eta, pred_nu = env.dynamics.step(wrench, current)
                            for _ in range(max(0, int(oracle_horizon_steps) - 1)):
                                pred_eta, pred_nu = env.dynamics.step(wrench, current)
                            env.dynamics.eta = eta_save
                            env.dynamics.nu = nu_save
                            return pred_eta, pred_nu

                        while not done:
                            dyn = env.dynamics
                            t = env.start_time + env.current_step * dyn.dt
                            target = env._get_target(t)
                            current_for_controller = env.privileged_state[:3]
                            current_world = current_for_controller[:2]
                            primary_action = _compute_base_action(
                                primary,
                                primary_name,
                                target,
                                dyn.eta,
                                dyn.nu,
                                t,
                                dyn.dt,
                                current_for_controller,
                            )
                            authority_action = _compute_base_action(
                                authority,
                                authority_name,
                                target,
                                dyn.eta,
                                dyn.nu,
                                t,
                                dyn.dt,
                                current_for_controller,
                            )
                            oracle_info = {}
                            if mode == "scenario_fixed":
                                label = teacher_alpha_for_scenario(scenario)
                            elif mode in {"oracle", "hybrid_oracle"}:
                                oracle_target = env._get_target(
                                    t + max(1, int(oracle_horizon_steps)) * dyn.dt
                                )
                                label, oracle_info = choose_oracle_alpha(
                                    target=oracle_target,
                                    eta=dyn.eta,
                                    nu=dyn.nu,
                                    primary_action=primary_action,
                                    authority_action=authority_action,
                                    current=current_for_controller,
                                    action_to_wrench=action_to_wrench,
                                    dynamics_stepper=dynamics_stepper,
                                    alpha_grid=alpha_grid,
                                    previous_action=prev_action,
                                    previous_alpha=prev_alpha,
                                    action_saturation_weight=oracle_action_saturation_weight,
                                    action_delta_weight=oracle_action_delta_weight,
                                    alpha_delta_weight=oracle_alpha_delta_weight,
                                )
                                if mode == "hybrid_oracle":
                                    label = hybrid_oracle_label(
                                        t - env.start_time,
                                        label,
                                        window_alpha_schedule,
                                    )
                            elif mode in {"window_oracle", "schedule_oracle"}:
                                label = window_oracle_label_at_time(
                                    t - env.start_time,
                                    window_alpha_schedule,
                                )
                            else:
                                label = authority_blend_alpha(
                                    t - env.start_time,
                                    authority_until=authority_until,
                                    blend_duration=blend_duration,
                                    curve=blend_curve,
                                )
                            feature = build_alpha_feature(
                                target=target,
                                eta=dyn.eta,
                                nu=dyn.nu,
                                primary_action=primary_action,
                                authority_action=authority_action,
                                current_world=current_world,
                                t=t - env.start_time,
                                episode_end=episode_end,
                                feature_mode=feature_mode,
                            )
                            features.append(feature)
                            labels.append(float(label))
                            window_name = window_oracle_window_at_time(t - env.start_time)
                            rows.append({
                                "scenario": int(scenario),
                                "seed": int(seed),
                                "start_time": float(start_time),
                                "current_amplitude_scale": float(amp_scale),
                                "current_frequency_scale": float(freq_scale),
                                "step": int(env.current_step),
                                "time": float(t - env.start_time),
                                "absolute_time": float(t),
                                "teacher_alpha": float(label),
                                "oracle_best_cost": float(oracle_info.get("best_cost", np.nan)),
                                "oracle_worst_cost": float(oracle_info.get("worst_cost", np.nan)),
                                "window_oracle_window": window_name,
                                "window_oracle_alpha": float(
                                    window_alpha_schedule.get(window_name, np.nan)
                                ),
                                "error_norm": float(np.linalg.norm((target - dyn.eta)[:3])),
                                "vehicle_profile": str(vehicle_profile),
                                "action_mode": str(action_mode),
                                "thruster_layout": str(thruster_layout),
                            })
                            action, _info = blend_actions_with_predicted_alpha(
                                primary_action,
                                authority_action,
                                label,
                            )
                            _obs, _reward, done, truncated, _step_info = env.step(action)
                            done = done or truncated
                            prev_action = np.asarray(action, dtype=float).copy()
                            prev_alpha = float(label)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        pd.DataFrame(rows),
    )


def train_alpha_model(
        features: np.ndarray,
        labels: np.ndarray,
        hidden_dim: int,
        epochs: int,
        batch_size: int,
        lr: float,
        seed: int,
        policy_architecture: str = "mlp",
        history_len: int = 1,
        transformer_embed_dim: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        model_dropout: float = 0.0,
        label_table: pd.DataFrame | None = None) -> tuple[nn.Module, dict]:
    """Train alpha regressor against soft-handover teacher labels."""
    torch.manual_seed(int(seed))
    architecture = str(policy_architecture or "mlp").strip().lower()
    feature_array = features.astype(np.float32)
    y = torch.from_numpy(labels.astype(np.float32))
    weights = torch.where(y > 1.0e-6, torch.tensor(8.0), torch.tensor(1.0))
    if architecture == "transformer":
        model = TemporalAlphaTransformer(
            input_dim=int(features.shape[1]),
            history_len=int(history_len),
            embed_dim=int(transformer_embed_dim),
            num_heads=int(transformer_heads),
            num_layers=int(transformer_layers),
            dropout=float(model_dropout),
        )
        episode_ids = teacher_episode_ids(
            label_table,
            sample_count=int(feature_array.shape[0]),
        )
        estimated_window_bytes = (
            int(feature_array.shape[0])
            * int(history_len)
            * int(feature_array.shape[1])
            * np.dtype(np.float32).itemsize
        )
        if estimated_window_bytes <= 2_000_000_000:
            x = torch.from_numpy(precompute_temporal_windows(
                features=feature_array,
                episode_ids=episode_ids,
                history_len=int(history_len),
            ))
            dataset = TensorDataset(x, y, weights)
            temporal_dataset_mode = "precomputed"
        else:
            dataset = TemporalFeatureWindowDataset(
                features=feature_array,
                labels=labels,
                weights=weights.numpy(),
                episode_ids=episode_ids,
                history_len=int(history_len),
            )
            temporal_dataset_mode = "lazy"
        eval_loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
        )
    elif architecture == "mlp":
        x = torch.from_numpy(feature_array)
        model = AlphaMLP(
            input_dim=int(features.shape[1]),
            hidden_dim=int(hidden_dim),
            dropout=float(model_dropout),
        )
        dataset = TensorDataset(x, y, weights)
        temporal_dataset_mode = "single_step"
        eval_loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
        )
    else:
        raise ValueError(f"Unknown policy_architecture: {policy_architecture}")
    with torch.no_grad():
        feature_tensor = torch.from_numpy(feature_array)
        model.feature_mean.copy_(feature_tensor.mean(dim=0))
        model.feature_scale.copy_(feature_tensor.std(dim=0).clamp_min(1.0e-3))
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1.0e-4)
    for _epoch in range(int(epochs)):
        for xb, yb, wb in loader:
            pred = model(xb)
            loss = torch.mean(wb * (pred - yb) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        pred_batches = []
        y_batches = []
        for xb, yb, _wb in eval_loader:
            pred_batches.append(model(xb).detach().reshape(-1))
            y_batches.append(yb.detach().reshape(-1))
        pred = torch.cat(pred_batches)
        y_eval = torch.cat(y_batches)
        mse = torch.mean((pred - y_eval) ** 2)
        positive = y_eval > 1.0e-6
        metrics = {
            "train_mse": float(mse.item()),
            "train_mae": float(torch.mean(torch.abs(pred - y_eval)).item()),
            "teacher_positive_fraction": float(positive.float().mean().item()),
            "pred_alpha_mean": float(pred.mean().item()),
            "teacher_alpha_mean": float(y_eval.mean().item()),
            "policy_architecture": architecture,
            "history_len": int(history_len),
            "temporal_dataset_mode": temporal_dataset_mode,
            "positive_mae": (
                float(torch.mean(torch.abs(pred[positive] - y_eval[positive])).item())
                if bool(torch.any(positive)) else 0.0
            ),
        }
    return model, metrics


def load_alpha_model_checkpoint(
        checkpoint_path: str | Path,
        expected_feature_mode: str | None = None) -> tuple[nn.Module, dict]:
    """Load a saved alpha regressor and validate feature compatibility."""
    path = _resolve_project_path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Invalid alpha model checkpoint: {path}")

    input_dim = int(checkpoint.get("input_dim", 0))
    hidden_dim = int(checkpoint.get("hidden_dim", 64))
    feature_mode = str(checkpoint.get("feature_mode", "state_phase"))
    architecture = str(checkpoint.get("policy_architecture", "mlp"))
    if input_dim <= 0:
        raise ValueError(f"Alpha checkpoint is missing a valid input_dim: {path}")
    if expected_feature_mode is not None and feature_mode != str(expected_feature_mode):
        raise ValueError(
            "Alpha checkpoint feature_mode mismatch: "
            f"expected {expected_feature_mode}, got {feature_mode}"
        )

    if architecture == "transformer":
        model = TemporalAlphaTransformer(
            input_dim=input_dim,
            history_len=int(checkpoint.get("history_len", 1)),
            embed_dim=int(checkpoint.get("transformer_embed_dim", 64)),
            num_heads=int(checkpoint.get("transformer_heads", 4)),
            num_layers=int(checkpoint.get("transformer_layers", 2)),
            dropout=float(checkpoint.get("model_dropout", 0.0)),
        )
    elif architecture == "mlp":
        model = AlphaMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=float(checkpoint.get("model_dropout", 0.0)),
        )
    else:
        raise ValueError(f"Unknown alpha checkpoint architecture: {architecture}")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    metadata = {
        "checkpoint_path": str(path),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "feature_mode": feature_mode,
        "policy_architecture": architecture,
        "history_len": int(checkpoint.get("history_len", 1)),
        "transformer_embed_dim": int(checkpoint.get("transformer_embed_dim", 64)),
        "transformer_heads": int(checkpoint.get("transformer_heads", 4)),
        "transformer_layers": int(checkpoint.get("transformer_layers", 2)),
        "model_dropout": float(checkpoint.get("model_dropout", 0.0)),
    }
    return model, metadata


def run_predictive_alpha_episode(
        model: AlphaMLP | None,
        scenario: int,
        seed: int,
        steps: int,
        mass_scale_xy: float,
        damping_scale_xy: float,
        current_amplitude_scale: float,
        current_frequency_scale: float,
        vertical_current: float,
        primary_controller: str,
        authority_controller: str,
        feature_mode: str,
        initial_position_std: float = 0.0,
        initial_velocity_std: float = 0.0,
        start_time: float = 0.0,
        fixed_alpha: float | None = None,
        alpha_schedule: dict[str, float] | None = None,
        alpha_bias_schedule: dict[str, float] | None = None,
        alpha_mode: str = "model",
        oracle_alpha_grid: str = "0,0.05,0.1,0.2,0.35,0.5,0.65,0.8,0.9,1.0",
        oracle_horizon_steps: int = 1,
        oracle_action_saturation_weight: float = 0.02,
        oracle_action_delta_weight: float = 0.01,
        oracle_alpha_delta_weight: float = 0.0,
        alpha_gain: float = 1.0,
        alpha_threshold: float = 0.0,
        alpha_smoothing: float = 0.0,
        alpha_rate_limit: float = 1.0,
        alpha_deadband: float = 0.0,
        policy_architecture: str = "mlp",
        history_len: int = 1,
        uncertainty_mode: str = "none",
        uncertainty_samples: int = 1,
        uncertainty_gain: float = 0.0,
        uncertainty_threshold: float = 1.0,
        vehicle_profile: str = "real_10kg_v1",
        action_mode: str = "wrench",
        thruster_layout: str = "real_10kg_x",
        save_ts: bool = False) -> dict:
    """Evaluate SMC primary plus predicted-alpha MPC_3D authority."""
    env = AUVTrackingEnv(
        scenario=int(scenario),
        max_steps=int(steps),
        trajectory3d=True,
        start_time=float(start_time),
        mass_scale_xy=float(mass_scale_xy),
        damping_scale_xy=float(damping_scale_xy),
        current_amplitude_scale=float(current_amplitude_scale),
        current_frequency_scale=float(current_frequency_scale),
        initial_position_std=float(initial_position_std),
        initial_velocity_std=float(initial_velocity_std),
        vertical_current=float(vertical_current),
        vehicle_profile=str(vehicle_profile),
        action_mode=str(action_mode),
        thruster_layout=str(thruster_layout),
    )
    primary_name, primary = _build_base_controller(primary_controller)
    authority_name, authority = _build_base_controller(authority_controller)
    for controller in (primary, authority):
        if hasattr(controller, "reset"):
            controller.reset()
        if hasattr(controller, "set_trajectory3d"):
            controller.set_trajectory3d(True)
    env.reset(seed=int(seed))
    if model is not None:
        model.eval()
    episode_end = float(steps) * env.dynamics.dt
    errors, energies, actions = [], [], []
    thruster_forces = []
    headings, desired_headings, step_rows, time_rows = [], [], [], []
    rolls, pitches, desired_rolls, desired_pitches = [], [], [], []
    xs, ys, zs, target_zs, z_errors = [], [], [], [], []
    authority_active, authority_alpha = [], []
    alpha_raw_values, alpha_uncertainty_values = [], []
    feature_history = []
    prev_action = np.zeros(6, dtype=float)
    done = False
    alpha_mode = str(alpha_mode or "model").strip().lower()
    alpha_grid = np.asarray(_parse_csv_floats(oracle_alpha_grid), dtype=float)
    prev_alpha = 0.0

    def action_to_wrench(action):
        return env._action_to_wrench(action)[0]

    def dynamics_stepper(eta0, nu0, wrench, current):
        eta_save = env.dynamics.eta.copy()
        nu_save = env.dynamics.nu.copy()
        env.dynamics.eta = np.asarray(eta0, dtype=float).copy()
        env.dynamics.nu = np.asarray(nu0, dtype=float).copy()
        pred_eta, pred_nu = env.dynamics.step(wrench, current)
        for _ in range(max(0, int(oracle_horizon_steps) - 1)):
            pred_eta, pred_nu = env.dynamics.step(wrench, current)
        env.dynamics.eta = eta_save
        env.dynamics.nu = nu_save
        return pred_eta, pred_nu

    while not done:
        dyn = env.dynamics
        t = env.start_time + env.current_step * dyn.dt
        target = env._get_target(t)
        current_for_controller = env.privileged_state[:3]
        current_world = current_for_controller[:2]
        primary_action = _compute_base_action(
            primary,
            primary_name,
            target,
            dyn.eta,
            dyn.nu,
            t,
            dyn.dt,
            current_for_controller,
        )
        authority_action = _compute_base_action(
            authority,
            authority_name,
            target,
            dyn.eta,
            dyn.nu,
            t,
            dyn.dt,
            current_for_controller,
        )
        if alpha_mode == "oracle":
            oracle_target = env._get_target(
                t + max(1, int(oracle_horizon_steps)) * dyn.dt
            )
            alpha, _oracle_info = choose_oracle_alpha(
                target=oracle_target,
                eta=dyn.eta,
                nu=dyn.nu,
                primary_action=primary_action,
                authority_action=authority_action,
                current=current_for_controller,
                action_to_wrench=action_to_wrench,
                dynamics_stepper=dynamics_stepper,
                alpha_grid=alpha_grid,
                previous_action=prev_action,
                previous_alpha=prev_alpha,
                action_saturation_weight=oracle_action_saturation_weight,
                action_delta_weight=oracle_action_delta_weight,
                alpha_delta_weight=oracle_alpha_delta_weight,
            )
        elif alpha_schedule is not None:
            alpha = schedule_alpha_at_time(t - env.start_time, alpha_schedule)
        elif fixed_alpha is not None:
            alpha = float(fixed_alpha)
        elif model is None:
            alpha = 0.0
            alpha_raw = 0.0
            alpha_uncertainty = 0.0
        else:
            feature = build_alpha_feature(
                target=target,
                eta=dyn.eta,
                nu=dyn.nu,
                primary_action=primary_action,
                authority_action=authority_action,
                current_world=current_world,
                t=t - env.start_time,
                episode_end=episode_end,
                feature_mode=feature_mode,
            )
            feature_history.append(feature)
            if str(policy_architecture or "mlp").strip().lower() == "transformer":
                window = build_temporal_feature_window(feature_history, history_len)
                model_input = torch.from_numpy(window).unsqueeze(0)
            else:
                model_input = torch.from_numpy(feature).unsqueeze(0)
            alpha_raw, alpha_uncertainty = predict_alpha_with_uncertainty(
                model,
                model_input,
                samples=uncertainty_samples,
                mode=uncertainty_mode,
            )
            alpha = apply_uncertainty_adjustment(
                alpha_mean=alpha_raw,
                alpha_std=alpha_uncertainty,
                mode=uncertainty_mode if uncertainty_mode != "mc_dropout" else "exponential",
                gain=uncertainty_gain,
                threshold=uncertainty_threshold,
            )
            alpha = calibrate_alpha_with_bias_schedule(
                alpha,
                t - env.start_time,
                alpha_bias_schedule,
            )
        if fixed_alpha is not None or alpha_schedule is not None or alpha_mode == "oracle":
            alpha_raw = float(alpha)
            alpha_uncertainty = 0.0
        alpha = float(np.clip(alpha * float(alpha_gain), 0.0, 1.0))
        if alpha < float(alpha_threshold):
            alpha = 0.0
        alpha = filter_authority_alpha(
            raw_alpha=alpha,
            previous_alpha=prev_alpha,
            smoothing=alpha_smoothing,
            rate_limit=alpha_rate_limit,
            deadband=alpha_deadband,
        )
        action, info = blend_actions_with_predicted_alpha(
            primary_action,
            authority_action,
            alpha,
        )
        step_rows.append(int(env.current_step))
        time_rows.append(float(t))
        rolls.append(float(dyn.eta[3]))
        pitches.append(float(dyn.eta[4]))
        headings.append(float(dyn.eta[5]))
        desired_rolls.append(float(target[3]))
        desired_pitches.append(float(target[4]))
        desired_headings.append(float(_desired_heading(t)))
        xs.append(float(dyn.eta[0]))
        ys.append(float(dyn.eta[1]))
        zs.append(float(dyn.eta[2]))
        target_zs.append(float(target[2]))
        authority_active.append(float(info["authority_active"]))
        authority_alpha.append(float(info["authority_alpha"]))
        alpha_raw_values.append(float(alpha_raw))
        alpha_uncertainty_values.append(float(alpha_uncertainty))
        _obs, _reward, done, truncated, step_info = env.step(action)
        done = done or truncated
        errors.append(float(step_info["dist_error"]))
        energies.append(float(step_info["energy"]))
        z_errors.append(float(step_info.get("z_error", target[2] - dyn.eta[2])))
        actions.append(np.asarray(action, dtype=float).copy())
        if "thruster_forces" in step_info:
            thruster_forces.append(np.asarray(step_info["thruster_forces"], dtype=float).copy())
        prev_action = np.asarray(action, dtype=float).copy()
        prev_alpha = float(alpha)
    metrics = _compute_metrics(
        errors,
        energies,
        actions,
        headings,
        desired_headings,
        env.dynamics.dt,
        z_errors=z_errors,
    )
    metrics["primary_controller"] = primary_name
    metrics["authority_controller"] = authority_name
    metrics["authority_until"] = -1.0 if fixed_alpha is None and alpha_schedule is None else 0.0
    metrics["blend_duration"] = -1.0
    if alpha_schedule is not None:
        metrics["blend_curve"] = "scheduled_alpha"
    elif fixed_alpha is not None:
        metrics["blend_curve"] = "fixed"
    elif alpha_mode == "oracle":
        metrics["blend_curve"] = "oracle_dynamic_alpha"
    elif alpha_bias_schedule is not None:
        metrics["blend_curve"] = "predictive_alpha_sspo"
    else:
        metrics["blend_curve"] = "predictive_alpha"
    if alpha_bias_schedule is not None:
        metrics["alpha_bias_schedule_json"] = json.dumps(alpha_bias_schedule, sort_keys=True)
        metrics["alpha_bias_abs_mean"] = float(np.mean(np.abs(list(alpha_bias_schedule.values()))))
    metrics["authority_active_fraction"] = float(np.mean(authority_active))
    metrics["authority_alpha_mean"] = float(np.mean(authority_alpha))
    metrics["authority_alpha_std"] = float(np.std(authority_alpha))
    metrics["alpha_raw_mean"] = float(np.mean(alpha_raw_values))
    metrics["alpha_uncertainty_std"] = float(np.mean(alpha_uncertainty_values))
    metrics["policy_architecture"] = str(policy_architecture)
    metrics["history_len"] = int(history_len)
    metrics["trajectory3d"] = True
    metrics["start_time"] = float(start_time)
    metrics["current_amplitude_scale"] = float(current_amplitude_scale)
    metrics["current_frequency_scale"] = float(current_frequency_scale)
    metrics["initial_position_std"] = float(initial_position_std)
    metrics["initial_velocity_std"] = float(initial_velocity_std)
    metrics["vertical_current"] = float(vertical_current)
    metrics["vehicle_profile"] = str(vehicle_profile)
    metrics["action_mode"] = str(action_mode)
    metrics["thruster_layout"] = str(thruster_layout)
    ts = pd.DataFrame({
        "step": step_rows,
        "time": time_rows,
        "error": errors,
        "energy": energies,
        "roll": rolls,
        "pitch": pitches,
        "yaw": headings,
        "desired_roll": desired_rolls,
        "desired_pitch": desired_pitches,
        "desired_yaw": desired_headings,
        "heading": headings,
        "desired_heading": desired_headings,
        "x": xs,
        "y": ys,
        "z": zs,
        "target_z": target_zs,
        "z_error": z_errors,
        "authority_active": authority_active,
        "authority_alpha": authority_alpha,
        "alpha_raw": alpha_raw_values,
        "alpha_uncertainty": alpha_uncertainty_values,
    })
    action_arr = np.asarray(actions, dtype=float)
    for idx in range(action_arr.shape[1]):
        ts[f"action_{idx}"] = action_arr[:, idx]
    if thruster_forces:
        thruster_arr = np.asarray(thruster_forces, dtype=float)
        for idx in range(thruster_arr.shape[1]):
            ts[f"thruster_force_{idx}"] = thruster_arr[:, idx]
    ts["vehicle_profile"] = str(vehicle_profile)
    ts["action_mode"] = str(action_mode)
    ts["thruster_layout"] = str(thruster_layout)
    metrics.update(compute_timeseries_engineering_metrics(ts, dt=env.dynamics.dt))
    if save_ts:
        metrics["ts"] = ts
    return metrics


def _scalar_metrics(metrics: dict) -> dict:
    return {
        key: value
        for key, value in metrics.items()
        if key != "ts" and np.isscalar(value)
    }


def _float_grid_arg(args, grid_name: str, fallback_name: str, fallback_default: float) -> list[float]:
    value = getattr(args, grid_name, None)
    if value is not None and str(value).strip():
        return [float(item) for item in _parse_csv_floats(value)]
    fallback = getattr(args, fallback_name, fallback_default)
    return [float(fallback)]


def _plot_predictive_summary(overall: pd.DataFrame, figure_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    if overall.empty:
        return paths
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
    labels = overall["method"].tolist()
    colors = ["#4C6F91", "#54946B", "#C36B3C", "#8B6BAE"][:len(labels)]
    gain_col = (
        "rmse_gain_vs_authority0_pct"
        if "rmse_gain_vs_authority0_pct" in overall.columns
        else "rmse_gain_pct"
    )
    for ax, metric, title in [
            (axes[0], "rmse", "3D RMSE"),
            (axes[1], gain_col, "RMSE gain (%)"),
            (axes[2], "action_jerk_mean", "Action jerk")]:
        ax.bar(np.arange(len(overall)), overall[metric].to_numpy(dtype=float),
               color=colors)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "predictive_alpha_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def run_experiment(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = _resolve_project_path(args.out_dir)
    figure_dir = _resolve_project_path(args.figure_dir)
    ts_dir = out_dir / "timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    ts_dir.mkdir(parents=True, exist_ok=True)

    train_scenarios = _parse_csv_ints(args.train_scenarios)
    train_seeds = _parse_csv_ints(args.train_seeds)
    eval_scenarios = _parse_csv_ints(args.eval_scenarios)
    eval_seeds = _parse_csv_ints(args.eval_seeds)
    train_start_times = _float_grid_arg(args, "train_start_times", "start_time", 0.0)
    eval_start_times = _float_grid_arg(args, "eval_start_times", "start_time", 0.0)
    train_amp_scales = _float_grid_arg(
        args,
        "train_current_amplitude_scales",
        "current_amplitude_scale",
        2.5,
    )
    eval_amp_scales = _float_grid_arg(
        args,
        "eval_current_amplitude_scales",
        "current_amplitude_scale",
        2.5,
    )
    train_freq_scales = _float_grid_arg(
        args,
        "train_current_frequency_scales",
        "current_frequency_scale",
        1.0,
    )
    eval_freq_scales = _float_grid_arg(
        args,
        "eval_current_frequency_scales",
        "current_frequency_scale",
        1.0,
    )
    pretrained_path = str(getattr(args, "pretrained_alpha_model", "") or "").strip()
    teacher_cache_path = str(getattr(args, "teacher_cache_path", "") or "").strip()
    save_timeseries = bool(getattr(args, "save_timeseries", True))
    requested_methods = {
        item.strip()
        for item in str(getattr(
            args,
            "eval_methods",
            "smc_baseline,predictive_alpha",
        )).split(",")
        if item.strip()
    }
    valid_methods = {"smc_baseline", "predictive_alpha"}
    if not requested_methods:
        raise ValueError("eval_methods must include at least one method")
    unknown_methods = requested_methods - valid_methods
    if unknown_methods:
        raise ValueError(f"Unknown eval_methods: {sorted(unknown_methods)}")
    if pretrained_path:
        model, checkpoint_meta = load_alpha_model_checkpoint(
            pretrained_path,
            expected_feature_mode=args.feature_mode,
        )
        features = np.empty((0, int(checkpoint_meta["input_dim"])), dtype=np.float32)
        labels = np.empty((0,), dtype=np.float32)
        dataset_sample_count = 0
        training_sample_count = 0
        pd.DataFrame([{
            "pretrained_alpha_model": checkpoint_meta["checkpoint_path"],
            "feature_mode": checkpoint_meta["feature_mode"],
            "input_dim": checkpoint_meta["input_dim"],
        }]).to_csv(out_dir / "teacher_alpha_labels.csv", index=False)
        train_metrics = {
            "training_skipped": True,
            "pretrained_alpha_model": checkpoint_meta["checkpoint_path"],
            "input_dim": int(checkpoint_meta["input_dim"]),
            "hidden_dim": int(checkpoint_meta["hidden_dim"]),
            "feature_mode": checkpoint_meta["feature_mode"],
            "policy_architecture": checkpoint_meta["policy_architecture"],
            "history_len": checkpoint_meta["history_len"],
        }
        model_hidden_dim = int(checkpoint_meta["hidden_dim"])
        model_policy_architecture = str(checkpoint_meta["policy_architecture"])
        model_history_len = int(checkpoint_meta["history_len"])
        model_transformer_embed_dim = int(checkpoint_meta["transformer_embed_dim"])
        model_transformer_heads = int(checkpoint_meta["transformer_heads"])
        model_transformer_layers = int(checkpoint_meta["transformer_layers"])
        model_dropout = float(checkpoint_meta["model_dropout"])
    else:
        cache_loaded = False
        if teacher_cache_path and _resolve_project_path(teacher_cache_path).exists():
            features, labels, label_table, cache_meta = load_teacher_cache(teacher_cache_path)
            cache_loaded = True
            if str(cache_meta.get("feature_mode", args.feature_mode)) != str(args.feature_mode):
                raise ValueError(
                    "Teacher cache feature_mode mismatch: "
                    f"expected {args.feature_mode}, got {cache_meta.get('feature_mode')}"
                )
        else:
            features, labels, label_table = collect_alpha_teacher_dataset(
                scenarios=train_scenarios,
                seeds=train_seeds,
                steps=args.steps,
                mass_scale_xy=args.mass_scale_xy,
                damping_scale_xy=args.damping_scale_xy,
                current_amplitude_scale=train_amp_scales[0],
                current_frequency_scale=train_freq_scales[0],
                vertical_current=args.vertical_current,
                primary_controller=args.primary_controller,
                authority_controller=args.authority_controller,
                authority_until=args.teacher_authority_until,
                blend_duration=args.teacher_blend_duration,
                blend_curve=args.teacher_blend_curve,
                feature_mode=args.feature_mode,
                teacher_mode=args.teacher_mode,
                oracle_alpha_grid=args.oracle_alpha_grid,
                oracle_horizon_steps=args.oracle_horizon_steps,
                oracle_action_saturation_weight=args.oracle_action_saturation_weight,
                oracle_action_delta_weight=args.oracle_action_delta_weight,
                oracle_alpha_delta_weight=args.oracle_alpha_delta_weight,
                window_oracle_alpha_gain=args.window_oracle_alpha_gain,
                vehicle_profile=args.vehicle_profile,
                action_mode=args.action_mode,
                thruster_layout=args.thruster_layout,
                start_times=train_start_times,
                current_amplitude_scales=train_amp_scales,
                current_frequency_scales=train_freq_scales,
            )
            if teacher_cache_path:
                save_teacher_cache(
                    teacher_cache_path,
                    features=features,
                    labels=labels,
                    label_table=label_table,
                    metadata={
                        "feature_mode": str(args.feature_mode),
                        "steps": int(args.steps),
                        "train_scenarios": train_scenarios,
                        "train_seeds": train_seeds,
                        "train_start_times": train_start_times,
                        "train_current_amplitude_scales": train_amp_scales,
                        "train_current_frequency_scales": train_freq_scales,
                        "teacher_mode": str(args.teacher_mode),
                        "oracle_alpha_grid": str(args.oracle_alpha_grid),
                        "oracle_horizon_steps": int(args.oracle_horizon_steps),
                        "window_oracle_alpha_gain": float(args.window_oracle_alpha_gain),
                    },
                )
        label_table.to_csv(out_dir / "teacher_alpha_labels.csv", index=False)
        train_features = features
        train_labels = labels
        train_label_table = label_table
        max_train_samples = int(getattr(args, "max_train_samples", 0) or 0)
        if max_train_samples > 0:
            train_features, train_labels, train_label_table = limit_teacher_training_samples(
                features,
                labels,
                label_table,
                max_samples=max_train_samples,
                seed=int(args.train_seed),
            )
        model, train_metrics = train_alpha_model(
            features=train_features,
            labels=train_labels,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.train_seed,
            policy_architecture=args.policy_architecture,
            history_len=args.history_len,
            transformer_embed_dim=args.transformer_embed_dim,
            transformer_heads=args.transformer_heads,
            transformer_layers=args.transformer_layers,
            model_dropout=args.model_dropout,
            label_table=train_label_table,
        )
        train_metrics["teacher_cache_path"] = (
            str(_resolve_project_path(teacher_cache_path)) if teacher_cache_path else ""
        )
        train_metrics["teacher_cache_loaded"] = bool(cache_loaded)
        train_metrics["teacher_samples_total"] = int(labels.size)
        train_metrics["teacher_samples_used"] = int(train_labels.size)
        train_metrics["max_train_samples"] = int(max_train_samples)
        dataset_sample_count = int(labels.size)
        training_sample_count = int(train_labels.size)
        model_hidden_dim = int(args.hidden_dim)
        model_policy_architecture = str(args.policy_architecture)
        model_history_len = int(args.history_len)
        model_transformer_embed_dim = int(args.transformer_embed_dim)
        model_transformer_heads = int(args.transformer_heads)
        model_transformer_layers = int(args.transformer_layers)
        model_dropout = float(args.model_dropout)
    rows, window_frames, ts_frames = [], [], []
    progress_every = max(0, int(getattr(args, "progress_every", 0) or 0))
    methods = [item for item in [
        ("smc_baseline", None, 0.0, 1.0, 0.0),
        ("predictive_alpha", model, None, args.alpha_gain, args.alpha_threshold),
    ] if item[0] in requested_methods]
    total_eval_episodes = (
        len(methods)
        * len(eval_scenarios)
        * len(eval_seeds)
        * len(eval_start_times)
        * len(eval_amp_scales)
        * len(eval_freq_scales)
    )
    completed_eval_episodes = 0
    eval_start_wall = time.time()
    for method, model_ref, fixed_alpha, alpha_gain, alpha_threshold in methods:
        for scenario in eval_scenarios:
            for seed in eval_seeds:
                for start_time in eval_start_times:
                    for amp_scale in eval_amp_scales:
                        for freq_scale in eval_freq_scales:
                            metrics = run_predictive_alpha_episode(
                                model=model_ref,
                                scenario=scenario,
                                seed=seed,
                                steps=args.steps,
                                mass_scale_xy=args.mass_scale_xy,
                                damping_scale_xy=args.damping_scale_xy,
                                current_amplitude_scale=amp_scale,
                                current_frequency_scale=freq_scale,
                                vertical_current=args.vertical_current,
                                primary_controller=args.primary_controller,
                                authority_controller=args.authority_controller,
                                feature_mode=args.feature_mode,
                                initial_position_std=args.eval_initial_position_std,
                                initial_velocity_std=args.eval_initial_velocity_std,
                                start_time=start_time,
                                fixed_alpha=fixed_alpha,
                                alpha_gain=alpha_gain,
                                alpha_threshold=alpha_threshold,
                                alpha_mode=args.eval_alpha_mode if fixed_alpha is None else "model",
                                oracle_alpha_grid=args.oracle_alpha_grid,
                                oracle_horizon_steps=args.oracle_horizon_steps,
                                oracle_action_saturation_weight=args.oracle_action_saturation_weight,
                                oracle_action_delta_weight=args.oracle_action_delta_weight,
                                oracle_alpha_delta_weight=args.oracle_alpha_delta_weight,
                                vehicle_profile=args.vehicle_profile,
                                action_mode=args.action_mode,
                                thruster_layout=args.thruster_layout,
                                alpha_smoothing=args.alpha_smoothing,
                                alpha_rate_limit=args.alpha_rate_limit,
                                alpha_deadband=args.alpha_deadband,
                                policy_architecture=model_policy_architecture,
                                history_len=model_history_len,
                                uncertainty_mode=args.uncertainty_mode,
                                uncertainty_samples=args.uncertainty_samples,
                                uncertainty_gain=args.uncertainty_gain,
                                uncertainty_threshold=args.uncertainty_threshold,
                                save_ts=True,
                            )
                            row = {
                                "method": method,
                                "scenario": SCENARIO_NAMES.get(int(scenario), str(scenario)),
                                "scenario_id": int(scenario),
                                "seed": int(seed),
                                "start_time": float(start_time),
                                "mass_scale_xy": float(args.mass_scale_xy),
                                "damping_scale_xy": float(args.damping_scale_xy),
                                "initial_position_std": float(args.eval_initial_position_std),
                                "initial_velocity_std": float(args.eval_initial_velocity_std),
                                **_scalar_metrics(metrics),
                            }
                            rows.append(row)
                            ts = metrics["ts"].copy()
                            ts["relative_time"] = ts["step"].to_numpy(dtype=float) * 0.01
                            ts["method"] = method
                            ts["scenario_id"] = int(scenario)
                            ts["seed"] = int(seed)
                            ts["start_time"] = float(start_time)
                            ts["current_amplitude_scale"] = float(amp_scale)
                            ts["current_frequency_scale"] = float(freq_scale)
                            ts["initial_position_std"] = float(args.eval_initial_position_std)
                            ts["initial_velocity_std"] = float(args.eval_initial_velocity_std)
                            if save_timeseries:
                                ts_frames.append(ts)
                            window = compute_axis_window_metrics(ts, dt=0.01)
                            for key, value in row.items():
                                if np.isscalar(value) and key not in window.columns:
                                    window[key] = value
                                elif np.isscalar(value):
                                    window[f"episode_{key}"] = value
                            window_frames.append(window)
                            completed_eval_episodes += 1
                            if progress_every and (
                                    completed_eval_episodes == 1
                                    or completed_eval_episodes % progress_every == 0
                                    or completed_eval_episodes == total_eval_episodes):
                                elapsed = time.time() - eval_start_wall
                                print(
                                    "[predictive_alpha] "
                                    f"eval {completed_eval_episodes}/{total_eval_episodes} "
                                    f"method={method} scenario={scenario} seed={seed} "
                                    f"start={start_time} amp={amp_scale} freq={freq_scale} "
                                    f"elapsed_s={elapsed:.1f}",
                                    flush=True,
                                )
    raw = _add_handover_gain_columns(
        pd.DataFrame(rows).assign(
            blend_curve=lambda df: df["blend_curve"].fillna("predictive_alpha"),
            blend_duration=lambda df: df["blend_duration"].fillna(-1.0),
        ),
        reference_until=0.0,
    )
    # The helper compares within blend_curve. For predictive-vs-baseline, add
    # direct baseline gains by scenario/seed.
    baseline = raw[raw["method"].eq("smc_baseline")][
        [
            "scenario_id",
            "seed",
            "start_time",
            "current_amplitude_scale",
            "current_frequency_scale",
            "initial_position_std",
            "initial_velocity_std",
            "rmse",
            "post_startup_rmse",
        ]
    ].rename(columns={
        "rmse": "_smc_rmse",
        "post_startup_rmse": "_smc_post",
    })
    if baseline.empty:
        raw["rmse_gain_pct"] = np.nan
        raw["post_startup_gain_pct"] = np.nan
    else:
        raw = raw.merge(
            baseline,
            on=[
                "scenario_id",
                "seed",
                "start_time",
                "current_amplitude_scale",
                "current_frequency_scale",
                "initial_position_std",
                "initial_velocity_std",
            ],
            how="left",
        )
        raw["rmse_gain_pct"] = (
            (raw["_smc_rmse"] - raw["rmse"]) / raw["_smc_rmse"].clip(lower=1.0e-12) * 100.0
        )
        raw["post_startup_gain_pct"] = (
            (raw["_smc_post"] - raw["post_startup_rmse"]) / raw["_smc_post"].clip(lower=1.0e-12) * 100.0
        )
        raw = raw.drop(columns=["_smc_rmse", "_smc_post"], errors="ignore")
    aggregate = {
        "rmse": "mean",
        "post_startup_rmse": "mean",
        "tail_rmse": "mean",
        "mean_error": "mean",
        "max_error": "mean",
        "energy": "mean",
        "z_rmse": "mean",
        "authority_active_fraction": "mean",
        "authority_alpha_mean": "mean",
        "authority_alpha_std": "mean",
        "action_saturation_step_fraction": "mean",
        "action_tv_mean": "mean",
        "action_jerk_mean": "mean",
        "alpha_raw_mean": "mean",
        "alpha_uncertainty_std": "mean",
        "rmse_gain_pct": "mean",
        "post_startup_gain_pct": "mean",
    }
    overall = raw.groupby(["method"], as_index=False).agg(aggregate).sort_values("rmse")
    by_scenario = raw.groupby(["method", "scenario"], as_index=False).agg({
        "rmse": "mean",
        "post_startup_rmse": "mean",
        "rmse_gain_pct": "mean",
        "post_startup_gain_pct": "mean",
        "authority_alpha_mean": "mean",
        "authority_alpha_std": "mean",
        "action_jerk_mean": "mean",
        "alpha_raw_mean": "mean",
        "alpha_uncertainty_std": "mean",
    }).sort_values(["method", "scenario"])
    window_metrics = pd.concat(window_frames, ignore_index=True)
    raw.to_csv(out_dir / "raw_metrics.csv", index=False)
    overall.to_csv(out_dir / "overall_summary.csv", index=False)
    by_scenario.to_csv(out_dir / "by_scenario.csv", index=False)
    window_metrics.to_csv(out_dir / "window_metrics.csv", index=False)
    if save_timeseries and ts_frames:
        timeseries = pd.concat(ts_frames, ignore_index=True)
        timeseries.to_csv(ts_dir / "timeseries_3d.csv", index=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(features.shape[1]),
            "feature_mode": str(args.feature_mode),
            "hidden_dim": int(model_hidden_dim),
            "policy_architecture": model_policy_architecture,
            "history_len": int(model_history_len),
            "transformer_embed_dim": int(model_transformer_embed_dim),
            "transformer_heads": int(model_transformer_heads),
            "transformer_layers": int(model_transformer_layers),
            "model_dropout": float(model_dropout),
        },
        out_dir / "predictive_alpha_model.pt",
    )
    manifest = {
        "protocol": vars(args),
        "resolved_grids": {
            "train_start_times": train_start_times,
            "eval_start_times": eval_start_times,
            "train_current_amplitude_scales": train_amp_scales,
            "eval_current_amplitude_scales": eval_amp_scales,
            "train_current_frequency_scales": train_freq_scales,
            "eval_current_frequency_scales": eval_freq_scales,
        },
        "dataset_samples": int(dataset_sample_count),
        "training_samples": int(training_sample_count),
        "save_timeseries": bool(save_timeseries),
        "feature_dim": int(features.shape[1]),
        "train_metrics": train_metrics,
        "model": {
            "policy_architecture": model_policy_architecture,
            "feature_mode": str(args.feature_mode),
            "input_dim": int(features.shape[1]),
            "hidden_dim": int(model_hidden_dim),
            "history_len": int(model_history_len),
            "transformer_embed_dim": int(model_transformer_embed_dim),
            "transformer_heads": int(model_transformer_heads),
            "transformer_layers": int(model_transformer_layers),
            "model_dropout": float(model_dropout),
            "uncertainty_mode": str(args.uncertainty_mode),
            "uncertainty_samples": int(args.uncertainty_samples),
            "uncertainty_gain": float(args.uncertainty_gain),
            "uncertainty_threshold": float(args.uncertainty_threshold),
        },
        "formal_scope_guardrail": (
            "The learned model predicts only authority alpha for an existing "
            "SMC primary plus MPC_3D authority blend. It is not a global neural controller."
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    figure_paths = _plot_predictive_summary(overall, figure_dir)
    report = [
        "# 3D Predictive Authority Alpha",
        "",
        "## Overall",
        "```text",
        overall.to_string(index=False),
        "```",
        "",
        "## Train Metrics",
        "```json",
        json.dumps(train_metrics, indent=2),
        "```",
        "",
        "## Figures",
    ]
    report.extend([f"- {path}" for path in figure_paths])
    (out_dir / "predictive_alpha_summary.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")
    print(overall.to_string(index=False))
    print(json.dumps(train_metrics, indent=2))
    return raw, overall


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the predictive-alpha CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="results/3d_authority_diagnosis/predictive_alpha_scale2p5",
    )
    parser.add_argument(
        "--figure-dir",
        default="results/figures/3d_authority_diagnosis/predictive_alpha_scale2p5",
    )
    parser.add_argument("--steps", type=int, default=2100)
    parser.add_argument("--train-scenarios", default="1,2,3")
    parser.add_argument("--train-seeds", default="0,1,2")
    parser.add_argument("--eval-scenarios", default="1,2,3")
    parser.add_argument("--eval-seeds", default="0,1,2,3,4")
    parser.add_argument("--mass-scale-xy", type=float, default=1.5)
    parser.add_argument("--damping-scale-xy", type=float, default=0.5)
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--current-amplitude-scale", type=float, default=2.5)
    parser.add_argument("--current-frequency-scale", type=float, default=1.0)
    parser.add_argument("--eval-initial-position-std", type=float, default=0.0)
    parser.add_argument("--eval-initial-velocity-std", type=float, default=0.0)
    parser.add_argument(
        "--train-start-times",
        default="",
        help="Comma-separated training start times. Defaults to --start-time.",
    )
    parser.add_argument(
        "--eval-start-times",
        default="",
        help="Comma-separated evaluation start times. Defaults to --start-time.",
    )
    parser.add_argument(
        "--train-current-amplitude-scales",
        default="",
        help="Comma-separated training current amplitude scales.",
    )
    parser.add_argument(
        "--eval-current-amplitude-scales",
        default="",
        help="Comma-separated evaluation current amplitude scales.",
    )
    parser.add_argument(
        "--train-current-frequency-scales",
        default="",
        help="Comma-separated training current frequency scales.",
    )
    parser.add_argument(
        "--eval-current-frequency-scales",
        default="",
        help="Comma-separated evaluation current frequency scales.",
    )
    parser.add_argument("--vertical-current", type=float, default=0.0)
    parser.add_argument("--vehicle-profile", default="real_10kg_v1")
    parser.add_argument("--action-mode", default="wrench", choices=["wrench", "thruster"])
    parser.add_argument("--thruster-layout", default="real_10kg_x")
    parser.add_argument("--primary-controller", default="real10kg_smc_steady")
    parser.add_argument("--authority-controller", default="real10kg_mpc_event")
    parser.add_argument("--teacher-authority-until", type=float, default=2.0)
    parser.add_argument("--teacher-blend-duration", type=float, default=0.5)
    parser.add_argument(
        "--teacher-blend-curve",
        default="smoothstep",
        choices=["linear", "smoothstep"],
    )
    parser.add_argument(
        "--teacher-mode",
        default="handover",
        choices=[
            "handover",
            "scenario_fixed",
            "oracle",
            "window_oracle",
            "schedule_oracle",
            "hybrid_oracle",
        ],
    )
    parser.add_argument(
        "--feature-mode",
        default="state_phase",
        choices=["state", "state_phase"],
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--policy-architecture",
        default="mlp",
        choices=["mlp", "transformer"],
    )
    parser.add_argument("--history-len", type=int, default=1)
    parser.add_argument("--transformer-embed-dim", type=int, default=64)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--model-dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--alpha-gain", type=float, default=1.0)
    parser.add_argument("--alpha-threshold", type=float, default=0.0)
    parser.add_argument(
        "--eval-alpha-mode",
        default="model",
        choices=["model", "oracle"],
    )
    parser.add_argument(
        "--eval-methods",
        default="smc_baseline,predictive_alpha",
        help="Comma-separated methods to evaluate: smc_baseline,predictive_alpha.",
    )
    parser.add_argument(
        "--oracle-alpha-grid",
        default="0,0.25,0.5,0.75,1",
    )
    parser.add_argument("--oracle-horizon-steps", type=int, default=10)
    parser.add_argument("--oracle-action-saturation-weight", type=float, default=0.02)
    parser.add_argument("--oracle-action-delta-weight", type=float, default=0.01)
    parser.add_argument("--oracle-alpha-delta-weight", type=float, default=0.0)
    parser.add_argument(
        "--window-oracle-alpha-gain",
        type=float,
        default=1.0,
        help="Alpha gain used only when sweeping fixed-alpha labels for window_oracle teacher.",
    )
    parser.add_argument("--alpha-smoothing", type=float, default=0.0)
    parser.add_argument("--alpha-rate-limit", type=float, default=1.0)
    parser.add_argument("--alpha-deadband", type=float, default=0.0)
    parser.add_argument(
        "--uncertainty-mode",
        default="none",
        choices=["none", "mc_dropout", "threshold", "cap", "exponential"],
    )
    parser.add_argument("--uncertainty-samples", type=int, default=1)
    parser.add_argument("--uncertainty-gain", type=float, default=0.0)
    parser.add_argument("--uncertainty-threshold", type=float, default=1.0)
    parser.add_argument(
        "--pretrained-alpha-model",
        default="",
        help="Optional saved alpha model checkpoint. When set, skip teacher collection and training.",
    )
    parser.add_argument(
        "--teacher-cache-path",
        default="",
        help="Optional shared teacher dataset cache path for comparing PAC architectures.",
    )
    parser.add_argument(
        "--save-timeseries",
        dest="save_timeseries",
        action="store_true",
        default=True,
        help="Export concatenated per-step evaluation timeseries.",
    )
    parser.add_argument(
        "--no-save-timeseries",
        dest="save_timeseries",
        action="store_false",
        help="Skip concatenated per-step timeseries export while keeping metrics.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print evaluation progress every N completed episodes.",
    )
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())






