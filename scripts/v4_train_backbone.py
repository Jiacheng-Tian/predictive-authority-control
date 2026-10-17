#!/usr/bin/env python3
"""Train a disturbance-aware supervised backbone (backbone v2) on GPU.

Consumes a v4 oracle-teacher dataset (``teacher.npz`` + ``metadata.csv``
from ``v4_oracle_dataset.py``), trains ``TemporalAlphaTransformer`` with
an episode-disjoint split, and saves a checkpoint compatible with
``load_alpha_model_checkpoint`` so the entire downstream stack (formal
evaluation, residual RL) works unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from pac.authority.model import TemporalAlphaTransformer
from pac.v4.config import load_v4_config  # noqa: E402
from pac.v4.teacher import build_teacher_windows  # noqa: E402


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "pac_v4.yaml"))
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--history-len", type=int, default=None,
                        help="override authority_model.history_len (64 ablation)")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = load_v4_config(arguments.config)
    am = config.authority_model
    history_len = int(arguments.history_len or am.history_len)
    device = torch.device(arguments.device)

    data = np.load(Path(arguments.dataset_dir) / "teacher.npz")
    metadata = pd.read_csv(Path(arguments.dataset_dir) / "metadata.csv")
    features, labels = data["features"], data["labels"]
    windows = build_teacher_windows(features, metadata, history_len)

    # episode-disjoint split: holdout = the highest episode_index per
    # (family, difficulty) group (last seed of each) plus the last two
    # structured episodes
    episode_index = metadata["episode_index"].to_numpy()
    group = metadata.assign(_ep=episode_index).groupby(
        ["family", "difficulty"]
    )["_ep"].max()
    val_episodes = set(group.tolist())
    structured_eps = sorted(
        metadata[metadata["family"] == "structured"]["episode_index"].unique()
    )
    val_episodes.update(structured_eps[-2:])
    val_mask = np.isin(episode_index, sorted(val_episodes))
    train_mask = ~val_mask

    torch.manual_seed(int(arguments.model_seed))
    np.random.seed(int(arguments.model_seed))
    model = TemporalAlphaTransformer(
        input_dim=windows.shape[-1],
        history_len=history_len,
        embed_dim=int(am.embed_dim),
        num_heads=int(am.heads),
        num_layers=int(am.layers),
        dropout=float(am.dropout),
    ).to(device)
    with torch.no_grad():
        flat_train = windows[train_mask].reshape(-1, windows.shape[-1])
        model.feature_mean.copy_(
            torch.from_numpy(flat_train.mean(axis=0)).float()
        )
        model.feature_scale.copy_(
            torch.from_numpy(flat_train.std(axis=0)).float().clamp_min(1.0e-6)
        )

    def loader(mask: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(windows[mask]),
            torch.from_numpy(labels[mask]),
        )
        return DataLoader(dataset, batch_size=arguments.batch_size,
                          shuffle=shuffle, drop_last=False)

    train_loader = loader(train_mask, True)
    val_loader = loader(val_mask, False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(arguments.lr))
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    for epoch in range(int(arguments.max_epochs)):
        model.train()
        for batch_windows, batch_labels in train_loader:
            batch_windows = batch_windows.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_windows), batch_labels)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            errors = np.concatenate([
                ((model(batch_windows.to(device)).cpu().numpy()
                  - batch_labels.numpy()) ** 2)
                for batch_windows, batch_labels in val_loader
            ])
        val_mse = float(np.mean(errors))
        if val_mse < best_val - 1.0e-6:
            best_val, best_epoch, stale = val_mse, epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= int(arguments.patience):
                break

    out_dir = Path(arguments.out_dir) / f"pac_train_seed_{int(arguments.model_seed)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_version": "backbone_v2_disturbance",
        "input_dim": int(windows.shape[-1]),
        "architecture": "transformer",
        "policy_architecture": "transformer",
        "feature_mode": str(am.feature_mode),
        "history_len": history_len,
        "embed_dim": int(am.embed_dim),
        "transformer_embed_dim": int(am.embed_dim),
        "heads": int(am.heads),
        "transformer_heads": int(am.heads),
        "layers": int(am.layers),
        "transformer_layers": int(am.layers),
        "dropout": float(am.dropout),
        "model_dropout": float(am.dropout),
        "alpha_gain": float(am.alpha_gain),
        "alpha_smoothing": float(am.alpha_smoothing),
        "alpha_rate_limit": float(am.alpha_rate_limit),
        "alpha_deadband": float(am.alpha_deadband),
        "model_seed": int(arguments.model_seed),
        "dataset_dir": str(Path(arguments.dataset_dir).resolve()),
        "state_dict": best_state,
        "training_summary": {
            "best_epoch": int(best_epoch),
            "best_val_mse": float(best_val),
            "label_variance_val": float(np.var(labels[val_mask])),
            "r2_val": float(1.0 - best_val / max(np.var(labels[val_mask]), 1e-9)),
            "train_rows": int(train_mask.sum()),
            "val_rows": int(val_mask.sum()),
            "val_episodes": sorted(int(e) for e in val_episodes),
            "device": str(device),
        },
        "git_commit": _git_commit(),
    }
    checkpoint = out_dir / "checkpoint.pt"
    torch.save(payload, checkpoint)
    print(json.dumps({
        "checkpoint": str(checkpoint),
        "best_epoch": best_epoch,
        "val_mse": round(best_val, 6),
        "val_r2": payload["training_summary"]["r2_val"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
