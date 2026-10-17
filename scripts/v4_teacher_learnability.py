#!/usr/bin/env python3
"""S0 learnability gate: can the backbone architecture learn teacher
labels on disturbance episodes?  Trains the existing transformer on GPU
and reports validation R^2 overall / per family / per difficulty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pac.authority.model import TemporalAlphaTransformer
from pac.v4.config import load_v4_config
from pac.v4.teacher import build_teacher_windows


def r2(predictions: np.ndarray, targets: np.ndarray) -> float:
    variance = float(np.var(targets))
    if variance <= 0.0:
        return float("nan")
    return 1.0 - float(np.mean((predictions - targets) ** 2)) / variance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pac_v4.yaml")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args(argv)

    config = load_v4_config(arguments.config)
    device = torch.device(arguments.device)
    data = np.load(Path(arguments.dataset_dir) / "teacher.npz")
    metadata = pd.read_csv(Path(arguments.dataset_dir) / "metadata.csv")
    features, labels = data["features"], data["labels"]
    windows = build_teacher_windows(
        features, metadata, int(config.authority_model.history_len)
    )
    # episode-disjoint split: even episode indices train, odd validate
    episode_index = metadata["episode_index"].to_numpy()
    train_mask = (episode_index % 2) == 0
    val_mask = ~train_mask

    model = TemporalAlphaTransformer(
        input_dim=windows.shape[-1],
        history_len=int(config.authority_model.history_len),
        embed_dim=int(config.authority_model.embed_dim),
        num_heads=int(config.authority_model.heads),
        num_layers=int(config.authority_model.layers),
        dropout=0.1,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()

    def loader(mask: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(windows[mask]),
            torch.from_numpy(labels[mask]),
        )
        return DataLoader(dataset, batch_size=arguments.batch_size,
                          shuffle=shuffle, drop_last=False)

    train_loader = loader(train_mask, True)
    val_loader = loader(val_mask, False)
    best_val = float("inf")
    model.train()
    for epoch in range(int(arguments.epochs)):
        for batch_windows, batch_labels in train_loader:
            batch_windows = batch_windows.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_windows), batch_labels)
            loss.backward()
            optimizer.step()
        if epoch % 5 == 4 or epoch == arguments.epochs - 1:
            model.eval()
            with torch.no_grad():
                errors = []
                for batch_windows, batch_labels in val_loader:
                    predictions = model(batch_windows.to(device)).cpu().numpy()
                    errors.append(
                        (predictions - batch_labels.numpy()) ** 2
                    )
                val_mse = float(np.mean(np.concatenate(errors)))
            best_val = min(best_val, val_mse)
            model.train()

    model.eval()
    with torch.no_grad():
        predictions = np.concatenate([
            model(batch_windows.to(device)).cpu().numpy()
            for batch_windows, _ in val_loader
        ])
    targets = labels[val_mask]
    val_meta = metadata[val_mask].reset_index(drop=True)

    report = {
        "device": str(device),
        "val_mse": best_val,
        "label_variance": float(np.var(targets)),
        "overall_r2": r2(predictions, targets),
        "low_alpha_fraction_labels": float((labels <= 0.3).mean()),
        "per_family_r2": {
            family: r2(predictions[val_meta["family"] == family],
                       targets[val_meta["family"] == family])
            for family in sorted(val_meta["family"].unique())
        },
        "per_difficulty_r2": {
            difficulty: r2(predictions[val_meta["difficulty"] == difficulty],
                           targets[val_meta["difficulty"] == difficulty])
            for difficulty in sorted(val_meta["difficulty"].unique())
        },
        "label_mean_per_difficulty": {
            difficulty: float(targets[val_meta["difficulty"] == difficulty].mean())
            for difficulty in sorted(val_meta["difficulty"].unique())
        },
        "low_alpha_fraction_per_difficulty": {
            difficulty: float((targets[val_meta["difficulty"] == difficulty] <= 0.3).mean())
            for difficulty in sorted(val_meta["difficulty"].unique())
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
