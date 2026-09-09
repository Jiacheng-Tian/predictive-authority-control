"""Transformer model, temporal windows, and checkpoint I/O for PAC."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile
from typing import Any, Mapping

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
    if not isinstance(label_table, pd.DataFrame):
        raise TypeError("label_table must be a pandas DataFrame")
    if "episode_uid" in label_table.columns:
        if len(label_table) != int(sample_count):
            raise ValueError("teacher label table length does not match feature count")
        identifiers, _ = pd.factorize(label_table["episode_uid"], sort=False)
        if bool((identifiers < 0).any()):
            raise ValueError("episode_uid values must not be null")
        return identifiers.astype(np.int64, copy=False)
    if "episode_fingerprint" in label_table.columns:
        if len(label_table) != int(sample_count):
            raise ValueError("teacher label table length does not match feature count")
        identifiers, _ = pd.factorize(label_table["episode_fingerprint"], sort=False)
        if bool((identifiers < 0).any()):
            raise ValueError("episode_fingerprint values must not be null")
        return identifiers.astype(np.int64, copy=False)
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


def _authority_model_values(config: Any) -> dict[str, Any]:
    """Resolve a model config to the semantic fields used by V3 artifacts."""
    if hasattr(config, "authority_model"):
        config = config.authority_model
    elif isinstance(config, Mapping) and "authority_model" in config:
        config = config["authority_model"]
    if is_dataclass(config):
        source = asdict(config)
    elif isinstance(config, Mapping):
        source = dict(config)
    else:
        source = {
            name: getattr(config, name)
            for name in (
                "architecture",
                "feature_mode",
                "history_len",
                "embed_dim",
                "heads",
                "layers",
                "dropout",
                "alpha_gain",
                "alpha_smoothing",
                "alpha_rate_limit",
                "alpha_deadband",
            )
            if hasattr(config, name)
        }
    required = (
        "architecture",
        "feature_mode",
        "history_len",
        "embed_dim",
        "heads",
        "layers",
        "dropout",
        "alpha_gain",
        "alpha_smoothing",
        "alpha_rate_limit",
        "alpha_deadband",
    )
    missing = [name for name in required if name not in source]
    if missing:
        raise ValueError(f"authority model config missing fields: {', '.join(missing)}")
    values: dict[str, Any] = {}
    for name in required:
        value = source[name]
        if name in {"architecture", "feature_mode"}:
            values[name] = str(value)
        elif name in {"history_len", "embed_dim", "heads", "layers"}:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"authority model config {name} must be an integer")
            values[name] = int(value)
        else:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"authority model config {name} must be numeric")
            values[name] = float(value)
    if values["architecture"] != "transformer":
        raise ValueError("authority model architecture must be transformer")
    if values["feature_mode"] != "state_phase":
        raise ValueError("authority model feature_mode must be state_phase")
    if not 1 <= values["history_len"] <= 512:
        raise ValueError("authority model history_len must be in [1, 512]")
    if not 1 <= values["embed_dim"] <= 2048:
        raise ValueError("authority model embed_dim must be in [1, 2048]")
    if not 1 <= values["heads"] <= values["embed_dim"]:
        raise ValueError("authority model heads must be in [1, embed_dim]")
    if values["embed_dim"] % values["heads"]:
        raise ValueError("authority model embed_dim must be divisible by heads")
    if not 1 <= values["layers"] <= 64:
        raise ValueError("authority model layers must be in [1, 64]")
    if not 0.0 <= values["dropout"] < 1.0:
        raise ValueError("authority model dropout must be in [0, 1)")
    if not all(
        math.isfinite(values[name])
        for name in ("dropout", "alpha_gain", "alpha_smoothing", "alpha_rate_limit", "alpha_deadband")
    ):
        raise ValueError("authority model floating-point values must be finite")
    if values["alpha_gain"] < 0.0:
        raise ValueError("authority model alpha_gain must be non-negative")
    for name in ("alpha_smoothing", "alpha_rate_limit", "alpha_deadband"):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"authority model {name} must be in [0, 1]")
    return values


def _semantic_config_hash(config: Any) -> str:
    values = _authority_model_values(config)
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config_hash_candidates(config: Any) -> tuple[str, ...]:
    """Accept model-only and full V3 config hashes at the artifact boundary."""
    model_hash = _semantic_config_hash(config)
    if not hasattr(config, "protocol"):
        return (model_hash,)
    full_values = asdict(config) if is_dataclass(config) else config
    encoded = json.dumps(full_values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return (model_hash, hashlib.sha256(encoded).hexdigest())


def _seed_all(seed: int) -> None:
    """Set every random source used by the CPU training path."""
    value = int(seed)
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _validated_model_seed(seed: Any) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("model_seed must be an integer")
    value = int(seed)
    if not 0 <= value <= 2**64 - 1:
        raise ValueError("model_seed must be in [0, 2**64 - 1]")
    return value


def _weighted_mse(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb, wb in loader:
            prediction = model(xb)
            total += float(torch.sum(wb * (prediction - yb) ** 2).item())
            count += int(yb.numel())
    if count <= 0:
        raise ValueError("training split must contain at least one sample")
    return total / count


def _metadata_episode_ids(metadata: pd.DataFrame, sample_count: int) -> np.ndarray:
    """Use the same episode-id/window path for both old and V3 metadata."""
    identifiers = teacher_episode_ids(metadata, sample_count)
    if identifiers.shape != (int(sample_count),):
        raise ValueError("teacher episode identifiers have an invalid shape")
    return identifiers


def _train_alpha_model_v3_impl(
        dataset: Any,
        config: Any,
        model_seed: int,
        max_epochs: int | None = None,
        patience: int | None = None,
) -> tuple[TemporalAlphaTransformer, pd.DataFrame, dict[str, Any]]:
    """Train the V3 authority model using episode-isolated train/validation data."""
    model_seed = _validated_model_seed(model_seed)
    _seed_all(model_seed)
    values = _authority_model_values(config)
    if isinstance(dataset, (tuple, list)) and len(dataset) == 3:
        dataset = type("Dataset", (), {
            "features": dataset[0],
            "labels": dataset[1],
            "metadata": dataset[2],
        })()
    features = np.array(getattr(dataset, "features", None), dtype=np.float32, copy=True)
    labels = np.array(getattr(dataset, "labels", None), dtype=np.float32, copy=True)
    metadata = getattr(dataset, "metadata", None)
    if features.ndim != 2 or features.shape[1] != 24:
        raise ValueError("V3 authority training requires features with shape (N, 24)")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must have shape (N,) matching features")
    if not np.isfinite(features).all() or not np.isfinite(labels).all():
        raise ValueError("features and labels must be finite")
    if not isinstance(metadata, pd.DataFrame):
        raise ValueError("V3 authority training requires dataset metadata")
    if len(metadata) != features.shape[0] or "split" not in metadata:
        raise ValueError("dataset metadata must align with features and include split")
    split_values = set(metadata["split"].tolist())
    if not {"train", "val"}.issubset(split_values):
        raise ValueError("dataset metadata must contain train and val splits")
    from pac.authority.dataset import assert_episode_split_disjoint

    assert_episode_split_disjoint(metadata)
    train_indices = np.flatnonzero(metadata["split"].to_numpy() == "train")
    val_indices = np.flatnonzero(metadata["split"].to_numpy() == "val")
    if train_indices.size == 0 or val_indices.size == 0:
        raise ValueError("V3 training requires non-empty train and val episode splits")

    identifiers = _metadata_episode_ids(metadata, features.shape[0])
    windows = precompute_temporal_windows(
        features,
        identifiers,
        values["history_len"],
    )
    x = torch.from_numpy(windows)
    y = torch.from_numpy(labels)
    weights = torch.where(y > 0.0, torch.tensor(8.0), torch.tensor(1.0))
    train_dataset = TensorDataset(x[train_indices], y[train_indices], weights[train_indices])
    val_dataset = TensorDataset(x[val_indices], y[val_indices], weights[val_indices])
    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(model_seed),
        num_workers=0,
    )
    train_eval_loader = DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0)

    model = TemporalAlphaTransformer(
        input_dim=24,
        history_len=values["history_len"],
        embed_dim=values["embed_dim"],
        num_heads=values["heads"],
        num_layers=values["layers"],
        dropout=values["dropout"],
    )
    with torch.no_grad():
        train_features = torch.from_numpy(features[train_indices])
        model.feature_mean.copy_(train_features.mean(dim=0))
        model.feature_scale.copy_(train_features.std(dim=0, unbiased=False).clamp_min(1.0e-3))

    epochs = 180 if max_epochs is None else int(max_epochs)
    stop_patience = 20 if patience is None else int(patience)
    if epochs <= 0 or stop_patience <= 0:
        raise ValueError("max_epochs and patience must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    history_rows: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = 0
    no_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb, wb in train_loader:
            prediction = model(xb)
            loss = torch.mean(wb * (prediction - yb) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        train_mse = _weighted_mse(model, train_eval_loader)
        val_mse = _weighted_mse(model, val_loader)
        history_rows.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        if val_mse < best_val - 1.0e-6:
            best_val = val_mse
            best_epoch = epoch
            no_improvement = 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            no_improvement += 1
            if no_improvement >= stop_patience:
                break
    if best_state is None:
        raise RuntimeError("V3 training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    history = pd.DataFrame(history_rows, columns=["epoch", "train_mse", "val_mse"])
    restored_val_mse = _weighted_mse(model, val_loader)
    restored_train_mse = _weighted_mse(model, train_eval_loader)
    with torch.no_grad():
        val_prediction = torch.cat([model(xb) for xb, _yb, _wb in val_loader])
        val_target = torch.cat([yb for _xb, yb, _wb in val_loader])
    metrics: dict[str, Any] = {
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(history_rows)),
        "train_mse": float(restored_train_mse),
        "val_mse": float(restored_val_mse),
        "best_val_mse": float(restored_val_mse),
        "val_mae": float(torch.mean(torch.abs(val_prediction - val_target)).item()),
        "train_samples": int(train_indices.size),
        "val_samples": int(val_indices.size),
        "model_seed": model_seed,
        "param_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "policy_architecture": "transformer",
        "feature_mode": values["feature_mode"],
        "history_len": int(values["history_len"]),
        "input_dim": 24,
        "temporal_dataset_mode": "precomputed",
    }
    if metrics["param_count"] != 14113:
        raise ValueError(
            "V3 authority model contract requires 14113 parameters, "
            f"got {metrics['param_count']}"
        )
    return model, history, metrics


def train_alpha_model_v3(
        dataset: Any,
        config: Any,
        model_seed: int,
        max_epochs: int | None = None,
        patience: int | None = None,
) -> tuple[TemporalAlphaTransformer, pd.DataFrame, dict[str, Any]]:
    """Train V3 while restoring every process-global random/configuration state."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
        if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled")
        else False
    )
    cudnn_deterministic = getattr(torch.backends.cudnn, "deterministic", None)
    cudnn_benchmark = getattr(torch.backends.cudnn, "benchmark", None)
    try:
        return _train_alpha_model_v3_impl(
            dataset,
            config,
            model_seed=model_seed,
            max_epochs=max_epochs,
            patience=patience,
        )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        if cudnn_deterministic is not None:
            torch.backends.cudnn.deterministic = cudnn_deterministic
        if cudnn_benchmark is not None:
            torch.backends.cudnn.benchmark = cudnn_benchmark


def _default_git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[3]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _default_git_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    commit = _default_git_commit()
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("unable to collect git provenance") from exc
    return {
        "git_commit": commit,
        "git_dirty": bool(status.strip()),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _default_dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "pandas", "scipy", "osqp", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_finite_number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _require_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _validate_training_metrics(
        value: Any,
        parameter_count: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("training_metrics must be a non-empty mapping")
    metrics = dict(value)
    best_key = next(
        (key for key in ("best_val_mse", "val_mse", "best") if key in metrics),
        None,
    )
    if best_key is None:
        raise ValueError("training_metrics must contain best_val_mse, val_mse, or best")
    _require_finite_number(metrics[best_key], f"training_metrics.{best_key}")
    best_epoch = _require_integer(metrics.get("best_epoch"), "training_metrics.best_epoch")
    epochs_ran = _require_integer(metrics.get("epochs_ran"), "training_metrics.epochs_ran")
    if best_epoch <= 0 or epochs_ran <= 0 or best_epoch > epochs_ran:
        raise ValueError("training_metrics best_epoch/epochs_ran must be positive and ordered")
    parameter_key = "param_count" if "param_count" in metrics else "parameter_count"
    if parameter_key not in metrics:
        raise ValueError("training_metrics must contain param_count or parameter_count")
    metric_parameter_count = _require_integer(
        metrics[parameter_key], f"training_metrics.{parameter_key}"
    )
    if metric_parameter_count != int(parameter_count):
        raise ValueError("training_metrics parameter count does not match checkpoint model")
    return metrics


def _validate_checkpoint_artifact_metadata(
        git_commit: Any,
        dependency_versions: Any,
        training_metrics: Any,
        parameter_count: int,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    commit = _require_nonempty_string(git_commit, "git_commit")
    if not isinstance(dependency_versions, Mapping) or not dependency_versions:
        raise ValueError("dependency_versions must be a non-empty mapping")
    dependencies: dict[str, str] = {}
    for key, value in dependency_versions.items():
        name = _require_nonempty_string(key, "dependency_versions key")
        version = _require_nonempty_string(value, f"dependency_versions[{name}]")
        dependencies[name] = version
    metrics = _validate_training_metrics(training_metrics, parameter_count)
    return commit, dependencies, metrics


def save_v3_checkpoint(
        checkpoint_path: str | Path,
        model: TemporalAlphaTransformer,
        config: Any,
        model_seed: int,
        dataset_hash: str,
        config_semantic_sha256: str | None = None,
        git_commit: str | None = None,
        dependency_versions: Mapping[str, str] | None = None,
        training_metrics: Mapping[str, Any] | None = None,
        git_dirty: bool | None = None,
        git_diff_sha256: str | None = None,
        protocol_version: str = "formal_true_mpc_v3") -> dict[str, Any]:
    """Write a new V3 checkpoint, refusing every pre-existing target."""
    if isinstance(checkpoint_path, TemporalAlphaTransformer) and not isinstance(model, TemporalAlphaTransformer):
        checkpoint_path, model = model, checkpoint_path
    if not isinstance(model, TemporalAlphaTransformer):
        raise TypeError("model must be a TemporalAlphaTransformer")
    model_seed = _validated_model_seed(model_seed)
    values = _authority_model_values(config)
    target = Path(checkpoint_path).resolve()
    if target.exists():
        raise FileExistsError(f"checkpoint target already exists: {target}")
    if protocol_version != "formal_true_mpc_v3":
        raise ValueError("unsupported V3 checkpoint protocol_version")
    if not isinstance(dataset_hash, str) or not dataset_hash:
        raise ValueError("dataset_hash must be a non-empty string")
    model_config_hash = _semantic_config_hash(config)
    semantic_hash = str(config_semantic_sha256) if config_semantic_sha256 is not None else model_config_hash
    if semantic_hash not in _config_hash_candidates(config):
        raise ValueError("config_semantic_sha256 does not match model config")
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    if parameter_count != 14113:
        raise ValueError(f"V3 checkpoint requires 14113 parameters, got {parameter_count}")
    if int(model.input_dim) != 24:
        raise ValueError("V3 checkpoint requires input_dim=24")
    if git_commit is None or git_dirty is None or git_diff_sha256 is None:
        git_provenance = _default_git_provenance()
    else:
        git_provenance = {
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "git_diff_sha256": git_diff_sha256,
        }
    git_value = git_provenance["git_commit"]
    if not isinstance(git_provenance["git_dirty"], bool):
        raise ValueError("git_dirty must be a boolean")
    git_diff_value = _require_nonempty_string(
        git_provenance["git_diff_sha256"], "git_diff_sha256"
    )
    dependency_value = (
        dependency_versions
        if dependency_versions is not None
        else _default_dependency_versions()
    )
    git_value, dependency_value, metrics_value = _validate_checkpoint_artifact_metadata(
        git_value,
        dependency_value,
        training_metrics,
        parameter_count,
    )
    payload: dict[str, Any] = {
        "protocol_version": protocol_version,
        "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "input_dim": 24,
        "architecture": values["architecture"],
        "policy_architecture": values["architecture"],
        "feature_mode": values["feature_mode"],
        "history_len": values["history_len"],
        "embed_dim": values["embed_dim"],
        "transformer_embed_dim": values["embed_dim"],
        "heads": values["heads"],
        "transformer_heads": values["heads"],
        "layers": values["layers"],
        "transformer_layers": values["layers"],
        "dropout": values["dropout"],
        "model_dropout": values["dropout"],
        "alpha_gain": values["alpha_gain"],
        "alpha_smoothing": values["alpha_smoothing"],
        "alpha_rate_limit": values["alpha_rate_limit"],
        "alpha_deadband": values["alpha_deadband"],
        "model_seed": model_seed,
        "dataset_hash": dataset_hash,
        "config_semantic_sha256": semantic_hash,
        "authority_model_config_semantic_sha256": model_config_hash,
        "git_commit": git_value,
        "git_dirty": git_provenance["git_dirty"],
        "git_diff_sha256": git_diff_value,
        "dependency_versions": dependency_value,
        "training_metrics": metrics_value,
        "param_count": parameter_count,
        "parameter_count": parameter_count,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.tmp-",
            dir=target.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists():
            raise FileExistsError(f"checkpoint target already exists: {target}")
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    return payload


def _load_v3_checkpoint_impl(
        checkpoint_path: str | Path,
        config: Any | None = None,
        *,
        dataset_hash: str | None = None,
        model_seed: int | None = None,
        config_semantic_sha256: str | None = None,
        config_hash: str | None = None,
        expected_dataset_hash: str | None = None,
        expected_model_seed: int | None = None,
        expected_config_semantic_sha256: str | None = None,
        expected_config_hash: str | None = None) -> tuple[TemporalAlphaTransformer, dict[str, Any]]:
    """Load and validate a V3 checkpoint contract without byte comparisons."""
    path = Path(checkpoint_path).resolve()
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, ValueError) as exc:
        raise ValueError(f"invalid V3 checkpoint: {path}") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("protocol_version") != "formal_true_mpc_v3":
        raise ValueError("checkpoint is not a formal_true_mpc_v3 checkpoint")
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("V3 checkpoint state_dict is missing")
    if checkpoint.get("architecture") != "transformer" or checkpoint.get("policy_architecture") != "transformer":
        raise ValueError("V3 checkpoint must use transformer architecture")
    if checkpoint.get("feature_mode") != "state_phase":
        raise ValueError("V3 checkpoint must use state_phase features")
    if int(checkpoint.get("input_dim", -1)) != 24:
        raise ValueError("V3 checkpoint must have input_dim=24")
    required_keys = (
        "architecture", "policy_architecture", "input_dim", "feature_mode",
        "history_len", "embed_dim", "transformer_embed_dim", "heads",
        "transformer_heads", "layers", "transformer_layers", "dropout",
        "model_dropout",
        "model_seed", "dataset_hash", "config_semantic_sha256", "param_count",
        "git_commit", "git_dirty", "git_diff_sha256", "dependency_versions",
        "training_metrics",
    )
    if any(key not in checkpoint for key in required_keys):
        raise ValueError("V3 checkpoint metadata is incomplete")
    stored_parameter_count = _require_integer(checkpoint["param_count"], "param_count")
    if stored_parameter_count != 14113:
        raise ValueError("V3 checkpoint parameter count must be 14113")
    if "parameter_count" in checkpoint:
        if _require_integer(checkpoint["parameter_count"], "parameter_count") != stored_parameter_count:
            raise ValueError("V3 checkpoint parameter_count mismatch")
    _validate_checkpoint_artifact_metadata(
        checkpoint["git_commit"],
        checkpoint["dependency_versions"],
        checkpoint["training_metrics"],
        stored_parameter_count,
    )
    if not isinstance(checkpoint["git_dirty"], bool):
        raise ValueError("V3 checkpoint git_dirty must be a boolean")
    _require_nonempty_string(checkpoint["git_diff_sha256"], "git_diff_sha256")
    if checkpoint["architecture"] != "transformer":
        raise ValueError("V3 checkpoint architecture must be transformer")
    if checkpoint["policy_architecture"] != checkpoint["architecture"]:
        raise ValueError("V3 checkpoint policy_architecture alias mismatch")
    if checkpoint["feature_mode"] != "state_phase":
        raise ValueError("V3 checkpoint feature_mode must be state_phase")
    if _require_integer(checkpoint["input_dim"], "input_dim") != 24:
        raise ValueError("V3 checkpoint input_dim must be 24")
    history_len = _require_integer(checkpoint["history_len"], "history_len")
    embed_dim = _require_integer(checkpoint["embed_dim"], "embed_dim")
    heads = _require_integer(checkpoint["heads"], "heads")
    layers = _require_integer(checkpoint["layers"], "layers")
    dropout = _require_finite_number(checkpoint["dropout"], "dropout")
    if _require_integer(checkpoint["transformer_embed_dim"], "transformer_embed_dim") != embed_dim:
        raise ValueError("V3 checkpoint embed_dim alias mismatch")
    if _require_integer(checkpoint["transformer_heads"], "transformer_heads") != heads:
        raise ValueError("V3 checkpoint heads alias mismatch")
    if _require_integer(checkpoint["transformer_layers"], "transformer_layers") != layers:
        raise ValueError("V3 checkpoint layers alias mismatch")
    if _require_finite_number(checkpoint["model_dropout"], "model_dropout") != dropout:
        raise ValueError("V3 checkpoint dropout alias mismatch")
    if config is not None:
        expected_values = _authority_model_values(config)
        expected_fields = {
            "architecture": expected_values["architecture"],
            "feature_mode": expected_values["feature_mode"],
            "history_len": expected_values["history_len"],
            "embed_dim": expected_values["embed_dim"],
            "heads": expected_values["heads"],
            "layers": expected_values["layers"],
            "dropout": expected_values["dropout"],
        }
        actual_fields = {
            "architecture": checkpoint["architecture"],
            "feature_mode": checkpoint["feature_mode"],
            "history_len": history_len,
            "embed_dim": embed_dim,
            "heads": heads,
            "layers": layers,
            "dropout": dropout,
        }
        if actual_fields != expected_fields:
            raise ValueError("V3 checkpoint architecture does not match expected config")
    stored_model_seed = _validated_model_seed(checkpoint["model_seed"])
    if not isinstance(checkpoint["dataset_hash"], str) or not checkpoint["dataset_hash"]:
        raise ValueError("V3 checkpoint dataset_hash is invalid")
    stored_hash = str(checkpoint["config_semantic_sha256"])
    if len(stored_hash) != 64:
        raise ValueError("V3 checkpoint config_semantic_sha256 is invalid")
    expected_dataset_hash = expected_dataset_hash if expected_dataset_hash is not None else dataset_hash
    expected_model_seed = expected_model_seed if expected_model_seed is not None else model_seed
    expected_config_semantic_sha256 = (
        expected_config_semantic_sha256
        if expected_config_semantic_sha256 is not None
        else (config_semantic_sha256 if config_semantic_sha256 is not None else config_hash)
    )
    if expected_config_semantic_sha256 is None:
        expected_config_semantic_sha256 = expected_config_hash
    if expected_dataset_hash is not None and str(expected_dataset_hash) != checkpoint["dataset_hash"]:
        raise ValueError("V3 checkpoint dataset_hash mismatch")
    if expected_model_seed is not None and _validated_model_seed(expected_model_seed) != stored_model_seed:
        raise ValueError("V3 checkpoint model_seed mismatch")
    if config is not None:
        if stored_hash not in _config_hash_candidates(config):
            raise ValueError("V3 checkpoint config_semantic_sha256 mismatch")
    if expected_config_semantic_sha256 is not None and str(expected_config_semantic_sha256) != stored_hash:
        raise ValueError("V3 checkpoint config_semantic_sha256 mismatch")
    model = TemporalAlphaTransformer(
        input_dim=24,
        history_len=history_len,
        embed_dim=embed_dim,
        num_heads=heads,
        num_layers=layers,
        dropout=dropout,
    )
    if int(sum(parameter.numel() for parameter in model.parameters())) != 14113:
        raise ValueError("V3 checkpoint model shape does not have 14113 parameters")
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("V3 checkpoint state_dict does not match model shape") from exc
    model.eval()
    metadata = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    metadata["checkpoint_path"] = str(path)
    return model, metadata


def load_v3_checkpoint(
        checkpoint_path: str | Path,
        config: Any | None = None,
        **kwargs: Any) -> tuple[TemporalAlphaTransformer, dict[str, Any]]:
    """Load a V3 checkpoint and normalize all contract failures to ValueError."""
    path = Path(checkpoint_path).resolve()
    try:
        return _load_v3_checkpoint_impl(path, config=config, **kwargs)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        if isinstance(exc, ValueError) and "V3 checkpoint contract" in str(exc):
            raise
        raise ValueError(f"invalid V3 checkpoint contract at {path}: {exc}") from exc


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
