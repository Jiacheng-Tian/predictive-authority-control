"""Transformer model, temporal windows, and checkpoint I/O for PAC."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TemporalAlphaTransformer(nn.Module):
    """History encoder that predicts one bounded authority coefficient."""

    def __init__(
            self,
            input_dim: int,
            history_len: int,
            embed_dim: int,
            num_heads: int,
            num_layers: int,
            dropout: float):
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
        embedding = self.input_proj(x)
        embedding = embedding + self.pos_embedding[:, -embedding.shape[1]:, :]
        encoded = self.encoder(embedding)
        return self.head(encoded[:, -1, :]).squeeze(-1)


def build_temporal_feature_window(history, history_len: int) -> np.ndarray:
    """Return a fixed-length history window with formal-v2 zero padding."""
    if not history:
        raise ValueError("history must contain at least one feature vector")
    arrays = [np.asarray(item, dtype=np.float32).reshape(-1) for item in history]
    window_len = int(history_len)
    if window_len <= 0:
        raise ValueError("history_len must be positive")
    recent = arrays[-window_len:]
    window = np.zeros((window_len, arrays[-1].shape[0]), dtype=np.float32)
    window[-len(recent):, :] = np.asarray(recent, dtype=np.float32)
    return window


def teacher_episode_ids(label_table: pd.DataFrame, sample_count: int) -> np.ndarray:
    """Build contiguous episode identifiers from teacher metadata."""
    key_columns = [
        "scenario",
        "seed",
        "start_time",
        "current_amplitude_scale",
        "current_frequency_scale",
    ]
    if label_table.empty or not all(column in label_table for column in key_columns):
        return np.zeros(int(sample_count), dtype=np.int64)
    changed = label_table[key_columns].ne(label_table[key_columns].shift()).any(axis=1)
    identifiers = changed.cumsum().to_numpy(dtype=np.int64) - 1
    if identifiers.shape[0] != int(sample_count):
        raise ValueError("teacher label table length does not match feature count")
    return identifiers


def precompute_temporal_windows(
        features: np.ndarray,
        episode_ids: np.ndarray,
        history_len: int) -> np.ndarray:
    """Precompute temporal windows without crossing episode boundaries."""
    feature_array = np.asarray(features, dtype=np.float32)
    identifiers = np.asarray(episode_ids, dtype=np.int64)
    if feature_array.ndim != 2 or identifiers.shape[0] != feature_array.shape[0]:
        raise ValueError("features and episode_ids must have matching sample counts")
    window_len = int(history_len)
    if window_len <= 0:
        raise ValueError("history_len must be positive")
    windows = np.zeros(
        (feature_array.shape[0], window_len, feature_array.shape[1]),
        dtype=np.float32,
    )
    start = 0
    while start < feature_array.shape[0]:
        episode_id = int(identifiers[start])
        end = start + 1
        while end < feature_array.shape[0] and int(identifiers[end]) == episode_id:
            end += 1
        episode = feature_array[start:end]
        for index in range(episode.shape[0]):
            recent = episode[max(0, index - window_len + 1):index + 1]
            windows[start + index, -recent.shape[0]:, :] = recent
        start = end
    return windows


def train_alpha_model(
        features: np.ndarray,
        labels: np.ndarray,
        label_table: pd.DataFrame,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        seed: int,
        history_len: int,
        embed_dim: int,
        heads: int,
        layers: int,
        dropout: float) -> tuple[TemporalAlphaTransformer, dict[str, float | int | str]]:
    """Train the formal Transformer against oracle authority labels."""
    torch.manual_seed(int(seed))
    feature_array = np.asarray(features, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.float32)
    model = TemporalAlphaTransformer(
        input_dim=feature_array.shape[1],
        history_len=history_len,
        embed_dim=embed_dim,
        num_heads=heads,
        num_layers=layers,
        dropout=dropout,
    )
    identifiers = teacher_episode_ids(label_table, feature_array.shape[0])
    windows = precompute_temporal_windows(feature_array, identifiers, history_len)
    x = torch.from_numpy(windows)
    y = torch.from_numpy(label_array)
    weights = torch.where(y > 1.0e-6, torch.tensor(8.0), torch.tensor(1.0))
    dataset = TensorDataset(x, y, weights)
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=1.0e-4,
    )
    for _epoch in range(int(epochs)):
        for xb, yb, wb in loader:
            prediction = model(xb)
            loss = torch.mean(wb * (prediction - yb) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    evaluation_loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    with torch.no_grad():
        predictions = []
        targets = []
        for xb, yb, _weights in evaluation_loader:
            predictions.append(model(xb).detach().reshape(-1))
            targets.append(yb.detach().reshape(-1))
        prediction = torch.cat(predictions)
        target = torch.cat(targets)
        positive = target > 1.0e-6
        metrics = {
            "train_mse": float(torch.mean((prediction - target) ** 2).item()),
            "train_mae": float(torch.mean(torch.abs(prediction - target)).item()),
            "teacher_positive_fraction": float(positive.float().mean().item()),
            "pred_alpha_mean": float(prediction.mean().item()),
            "teacher_alpha_mean": float(target.mean().item()),
            "policy_architecture": "transformer",
            "history_len": int(history_len),
            "temporal_dataset_mode": "precomputed",
            "positive_mae": (
                float(torch.mean(torch.abs(prediction[positive] - target[positive])).item())
                if bool(torch.any(positive)) else 0.0
            ),
        }
    return model, metrics


def load_alpha_model_checkpoint(
        checkpoint_path: str | Path,
        expected_feature_mode: str | None = None) -> tuple[TemporalAlphaTransformer, dict]:
    """Load a formal Transformer checkpoint and validate feature compatibility."""
    path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Invalid alpha model checkpoint: {path}")
    feature_mode = str(checkpoint.get("feature_mode", "state_phase"))
    if expected_feature_mode is not None and feature_mode != expected_feature_mode:
        raise ValueError(
            f"Alpha checkpoint feature_mode mismatch: expected {expected_feature_mode}, got {feature_mode}"
        )
    architecture = str(checkpoint.get("policy_architecture", ""))
    if architecture != "transformer":
        raise ValueError("formal PAC checkpoints must use transformer architecture")
    input_dim = int(checkpoint["input_dim"])
    model = TemporalAlphaTransformer(
        input_dim=input_dim,
        history_len=int(checkpoint["history_len"]),
        embed_dim=int(checkpoint["transformer_embed_dim"]),
        num_heads=int(checkpoint["transformer_heads"]),
        num_layers=int(checkpoint["transformer_layers"]),
        dropout=float(checkpoint["model_dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, {
        "checkpoint_path": str(path),
        "input_dim": input_dim,
        "hidden_dim": int(checkpoint.get("hidden_dim", 64)),
        "feature_mode": feature_mode,
        "policy_architecture": architecture,
        "history_len": int(checkpoint["history_len"]),
        "transformer_embed_dim": int(checkpoint["transformer_embed_dim"]),
        "transformer_heads": int(checkpoint["transformer_heads"]),
        "transformer_layers": int(checkpoint["transformer_layers"]),
        "model_dropout": float(checkpoint["model_dropout"]),
    }
