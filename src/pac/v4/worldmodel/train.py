"""Supervised training for the physics + residual world-model ensemble."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from pac.v4.worldmodel.data import FEATURE_DIM, build_transition_windows, windows_to_torch
from pac.v4.worldmodel.model import (
    EnsembleDynamicsModel,
    ResidualDynamicsModel,
    WM_PROTOCOL_VERSION,
    load_world_model,
    save_world_model,
)
from pac.v4.collector import load_transition_dataset

MEMBER_SEED_BASE = 91000
_RESIDUAL_SCALE_MIN = 1.0e-4


def dataset_content_hash(dataset_dir: str | Path) -> str:
    """Content address for the transition dataset (npz + metadata hashes)."""
    root = Path(dataset_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    hashes = manifest.get("file_sha256", {})
    for name in ("transitions.npz", "metadata.csv"):
        if name not in hashes:
            raise ValueError(f"transition dataset manifest missing hash for {name}")
    digest = hashlib.sha256()
    digest.update(b"pac-v4-wm-dataset-v1\0")
    digest.update(hashes["transitions.npz"].encode("ascii"))
    digest.update(hashes["metadata.csv"].encode("ascii"))
    return digest.hexdigest()


def split_indices(metadata: pd.DataFrame, split: str) -> np.ndarray:
    values = metadata["split"].astype(str).to_numpy()
    return np.flatnonzero(values == split)


def _compute_normalization(train_rows: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    windows = train_rows["windows"].numpy()
    feature_mean = windows.reshape(-1, windows.shape[-1]).mean(axis=0)
    feature_scale = windows.reshape(-1, windows.shape[-1]).std(axis=0)
    residual = (train_rows["next_state"] - train_rows["physics_next_state"]).numpy()
    current_delta = train_rows["current_delta"].numpy()
    target = np.concatenate([residual, current_delta], axis=1)
    residual_mean = target.mean(axis=0)
    residual_scale = target.std(axis=0)
    feature_scale = np.clip(feature_scale, 1.0e-3, None)
    residual_scale = np.clip(residual_scale, _RESIDUAL_SCALE_MIN, None)
    return {
        "feature_mean": feature_mean.astype(np.float32),
        "feature_scale": feature_scale.astype(np.float32),
        "residual_mean": residual_mean.astype(np.float32),
        "residual_scale": residual_scale.astype(np.float32),
    }


def _member_targets(rows: dict[str, torch.Tensor], normalization: dict[str, np.ndarray]) -> torch.Tensor:
    residual = rows["next_state"] - rows["physics_next_state"]
    target = torch.cat([residual, rows["current_delta"]], dim=1)
    mean = torch.from_numpy(normalization["residual_mean"])
    scale = torch.from_numpy(normalization["residual_scale"])
    return (target - mean) / scale


def _loss_weight(device: torch.device, state_dim: int, current_weight: float) -> torch.Tensor:
    weights = torch.ones(state_dim + 2, dtype=torch.float32, device=device)
    weights[state_dim:] = float(current_weight)
    return weights


def train_member(
        member_seed: int,
        train_rows: dict[str, torch.Tensor],
        val_rows: dict[str, torch.Tensor],
        *,
        architecture: dict[str, int],
        normalization: dict[str, np.ndarray],
        max_epochs: int,
        patience: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        current_aux_weight: float,
        device: torch.device,
        log=lambda message: None) -> tuple[ResidualDynamicsModel, dict[str, Any]]:
    """Train one ensemble member with early stopping on validation loss."""
    torch.manual_seed(int(member_seed))
    random.seed(int(member_seed))
    np.random.seed(int(member_seed) % (2**32))
    model = ResidualDynamicsModel(**architecture)
    model.configure_normalization(**normalization)
    model.to(device)

    target_train = _member_targets(train_rows, normalization).to(device)
    target_val = _member_targets(val_rows, normalization).to(device)
    weights = _loss_weight(device, model.state_dim, current_aux_weight)

    train_dataset = TensorDataset(
        train_rows["windows"].to(device), target_train
    )
    generator = torch.Generator().manual_seed(int(member_seed))
    loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )

    def evaluate(model_: ResidualDynamicsModel) -> float:
        model_.eval()
        windows_val = val_rows["windows"].to(device)
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, windows_val.shape[0], 4096):
                prediction = model_(windows_val[start:start + 4096])
                loss = torch.mean(
                    weights * (prediction - target_val[start:start + 4096]) ** 2
                )
                total += float(loss.item()) * prediction.shape[0]
                count += prediction.shape[0]
        return total / max(count, 1)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    no_improvement = 0
    started = time.perf_counter()
    history: list[dict[str, float]] = []
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for window_batch, target_batch in loader:
            prediction = model(window_batch)
            loss = torch.mean(weights * (prediction - target_batch) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        val_loss = evaluate(model)
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss / max(batches, 1),
            "val_loss": val_loss,
        })
        if val_loss < best_val - 1.0e-6:
            best_val = val_loss
            best_epoch = epoch
            no_improvement = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            no_improvement += 1
            if no_improvement >= int(patience):
                break
        if epoch % 5 == 0 or epoch == 1:
            log(
                f"[wm-train seed={member_seed}] epoch={epoch} "
                f"train={history[-1]['train_loss']:.6f} val={val_loss:.6f}"
            )
    if best_state is None:
        raise RuntimeError("world-model member training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    metrics = {
        "member_seed": int(member_seed),
        "best_epoch": int(best_epoch),
        "epochs_ran": len(history),
        "best_val_loss": float(best_val),
        "final_train_loss": float(history[-1]["train_loss"]),
        "train_seconds": float(time.perf_counter() - started),
        "param_count": int(sum(p.numel() for p in model.parameters())),
    }
    return model, metrics


def train_world_model(
        config,
        dataset_dir: str | Path,
        output_dir: str | Path,
        *,
        device: str = "cpu",
        log=lambda message: None) -> dict[str, Any]:
    """Train the configured ensemble and persist the checkpoint atomically."""
    dataset = load_transition_dataset(dataset_dir)
    dataset_hash = dataset_content_hash(dataset_dir)
    wm = config.world_model
    resolved_device = torch.device(device)

    splits: dict[str, Any] = {}
    normalization_source = None
    for split in ("train", "val", "test"):
        indices = split_indices(dataset.metadata, split)
        if indices.size == 0:
            raise ValueError(f"transition dataset has no rows for split '{split}'")
        splits[split] = windows_to_torch(
            build_transition_windows(dataset, wm.history_len), indices
        )
        if split == "train":
            normalization_source = splits[split]
    normalization = _compute_normalization(normalization_source)

    architecture = {
        "feature_dim": FEATURE_DIM,
        "state_dim": 12,
        "history_len": int(wm.history_len),
        "hidden_dim": int(wm.hidden_dim),
        "num_layers": int(wm.num_layers),
        "residual_hidden_dim": int(wm.residual_hidden_dim),
    }
    member_seeds = [MEMBER_SEED_BASE + index for index in range(int(wm.members))]
    members: list[ResidualDynamicsModel] = []
    member_metrics: list[dict[str, Any]] = []
    for member_seed in member_seeds:
        member, metrics = train_member(
            member_seed,
            splits["train"],
            splits["val"],
            architecture=architecture,
            normalization=normalization,
            max_epochs=int(wm.max_epochs),
            patience=int(wm.patience),
            batch_size=int(wm.batch_size),
            learning_rate=float(wm.learning_rate),
            weight_decay=float(wm.weight_decay),
            current_aux_weight=float(wm.current_aux_weight),
            device=resolved_device,
            log=log,
        )
        members.append(member)
        member_metrics.append(metrics)
        log(
            f"[wm-train] member seed={member_seed} best_epoch={metrics['best_epoch']} "
            f"val={metrics['best_val_loss']:.6f}"
        )
    ensemble = EnsembleDynamicsModel(members, member_seeds)

    output_path = Path(output_dir) / "world_model.pt"
    payload = save_world_model(
        output_path,
        ensemble,
        dataset_hash=dataset_hash,
        member_training_metrics=member_metrics,
        config_hash=_config_content_hash(config),
        validation_metrics={},
    )
    summary = {
        "protocol_version": WM_PROTOCOL_VERSION,
        "dataset_dir": str(dataset_dir),
        "dataset_hash": dataset_hash,
        "checkpoint_path": str(output_path),
        "members": int(wm.members),
        "member_seeds": member_seeds,
        "member_metrics": member_metrics,
        "train_rows": int(splits["train"]["windows"].shape[0]),
        "val_rows": int(splits["val"]["windows"].shape[0]),
        "test_rows": int(splits["test"]["windows"].shape[0]),
        "device": str(resolved_device),
    }
    (Path(output_dir) / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def _config_content_hash(config) -> str:
    from dataclasses import asdict

    encoded = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_trained_world_model(
        checkpoint_path: str | Path,
        dataset_dir: str | Path | None = None):
    """Load an ensemble, optionally verifying the dataset hash binding."""
    expected = None
    if dataset_dir is not None:
        expected = dataset_content_hash(dataset_dir)
    ensemble, metadata = load_world_model(checkpoint_path, expected_dataset_hash=expected)
    return ensemble, metadata


__all__ = [
    "MEMBER_SEED_BASE",
    "dataset_content_hash",
    "load_trained_world_model",
    "split_indices",
    "train_member",
    "train_world_model",
]
