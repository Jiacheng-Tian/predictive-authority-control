"""Residual dynamics model and ensemble uncertainty wrapper for v4.

The world model follows the physics + residual form required by the v4 plan:

    x_{t+1} = f_physics(x_t, u_t, d_hat_t) + r_theta(history)

The residual head consumes a fixed-length history of 32-dim step features
(see :mod:`pac.v4.worldmodel.data`), produces the next-state residual
(12 dims) and an auxiliary horizontal-current increment (2 dims).  An
ensemble of members provides predictive mean and variance.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from pac.v4.worldmodel.data import FEATURE_DIM, STATE_DIM

WM_PROTOCOL_VERSION = "predictive_authority_v4"
WM_CHECKPOINT_KIND = "world_model"


class ResidualDynamicsModel(nn.Module):
    """GRU history encoder with a zero-initialized residual output head."""

    def __init__(
            self,
            *,
            feature_dim: int = FEATURE_DIM,
            state_dim: int = STATE_DIM,
            history_len: int,
            hidden_dim: int,
            num_layers: int,
            residual_hidden_dim: int):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.state_dim = int(state_dim)
        self.history_len = int(history_len)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.residual_hidden_dim = int(residual_hidden_dim)
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.residual_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.residual_hidden_dim, self.state_dim + 2),
        )
        self.register_buffer("feature_mean", torch.zeros(self.feature_dim))
        self.register_buffer("feature_scale", torch.ones(self.feature_dim))
        self.register_buffer("residual_mean", torch.zeros(self.state_dim + 2))
        self.register_buffer("residual_scale", torch.ones(self.state_dim + 2))
        self._zero_init_head()

    def _zero_init_head(self) -> None:
        final = self.head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Return normalized residual predictions ``(B, state_dim + 2)``."""
        if windows.ndim != 3 or windows.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected windows with shape (B, L, {self.feature_dim})"
            )
        if windows.shape[1] > self.history_len:
            windows = windows[:, -self.history_len:, :]
        normalized = (windows - self.feature_mean) / self.feature_scale.clamp_min(1.0e-6)
        _, hidden = self.gru(normalized)
        return self.head(hidden[-1])

    def predict_delta(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return denormalized ``(delta_state (B,12), delta_current (B,2))``."""
        normalized = self.forward(windows)
        denormalized = normalized * self.residual_scale + self.residual_mean
        return denormalized[:, : self.state_dim], denormalized[:, self.state_dim:]

    def configure_normalization(
            self,
            feature_mean: np.ndarray,
            feature_scale: np.ndarray,
            residual_mean: np.ndarray,
            residual_scale: np.ndarray) -> None:
        for name, value in (
            ("feature_mean", feature_mean),
            ("feature_scale", feature_scale),
            ("residual_mean", residual_mean),
            ("residual_scale", residual_scale),
        ):
            array = np.asarray(value, dtype=np.float32)
            expected = self.feature_dim if name.startswith("feature") else self.state_dim + 2
            if array.shape != (expected,):
                raise ValueError(f"{name} must have shape ({expected},)")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite")
            getattr(self, name).copy_(torch.from_numpy(array))


@dataclass(frozen=True)
class EnsemblePrediction:
    """Ensemble-aggregated world-model prediction."""

    mean_delta_state: torch.Tensor   # (..., 12)
    mean_delta_current: torch.Tensor  # (..., 2)
    std_delta_state: torch.Tensor    # (..., 12)
    std_delta_current: torch.Tensor  # (..., 2)


class EnsembleDynamicsModel:
    """Container of independently trained residual members."""

    def __init__(self, members: list[ResidualDynamicsModel], member_seeds: list[int]):
        if not members:
            raise ValueError("ensemble requires at least one member")
        reference = members[0]
        for member in members[1:]:
            for attribute in ("feature_dim", "state_dim", "history_len", "hidden_dim", "num_layers"):
                if getattr(member, attribute) != getattr(reference, attribute):
                    raise ValueError("ensemble members must share the architecture contract")
        self.members = list(members)
        self.member_seeds = [int(seed) for seed in member_seeds]
        if len(self.member_seeds) != len(self.members):
            raise ValueError("member_seeds length must match members")

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def history_len(self) -> int:
        return self.members[0].history_len

    def to(self, device: torch.device) -> "EnsembleDynamicsModel":
        self.members = [member.to(device) for member in self.members]
        return self

    def eval(self) -> "EnsembleDynamicsModel":
        for member in self.members:
            member.eval()
        return self

    def predict_delta(self, windows: torch.Tensor) -> EnsemblePrediction:
        with torch.no_grad():
            deltas = [member.predict_delta(windows) for member in self.members]
        states = torch.stack([item[0] for item in deltas], dim=0)
        currents = torch.stack([item[1] for item in deltas], dim=0)
        return EnsemblePrediction(
            mean_delta_state=states.mean(dim=0),
            mean_delta_current=currents.mean(dim=0),
            std_delta_state=states.std(dim=0, unbiased=False),
            std_delta_current=currents.std(dim=0, unbiased=False),
        )

    def member(self, index: int) -> ResidualDynamicsModel:
        return self.members[index]

    def parameters_count(self) -> list[int]:
        return [
            int(sum(parameter.numel() for parameter in member.parameters()))
            for member in self.members
        ]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _git_commit() -> str | None:
    import subprocess

    try:
        root = Path(__file__).resolve().parents[4]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _validated_seed(seed: Any) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("member seed must be an integer")
    return int(seed)


def save_world_model(
        checkpoint_path: str | Path,
        ensemble: EnsembleDynamicsModel,
        *,
        dataset_hash: str,
        member_training_metrics: list[dict[str, Any]],
        config_hash: str,
        feature_dim: int = FEATURE_DIM,
        state_dim: int = STATE_DIM,
        validation_metrics: dict[str, Any] | None = None,
        protocol_version: str = WM_PROTOCOL_VERSION) -> dict[str, Any]:
    """Persist the ensemble with the full v4 provenance contract."""
    if protocol_version != WM_PROTOCOL_VERSION:
        raise ValueError("unsupported world-model protocol_version")
    target = Path(checkpoint_path)
    if target.exists():
        raise FileExistsError(f"world-model checkpoint already exists: {target}")
    reference = ensemble.members[0]
    member_states = []
    for member, metrics in zip(ensemble.members, member_training_metrics):
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("member training metrics must be non-empty mappings")
        member_states.append(
            {name: value.detach().cpu().clone() for name, value in member.state_dict().items()}
        )
    payload = {
        "protocol_version": protocol_version,
        "kind": WM_CHECKPOINT_KIND,
        "feature_dim": int(feature_dim),
        "state_dim": int(state_dim),
        "history_len": int(reference.history_len),
        "hidden_dim": int(reference.hidden_dim),
        "num_layers": int(reference.num_layers),
        "residual_hidden_dim": int(reference.residual_hidden_dim),
        "members": len(ensemble.members),
        "member_seeds": [_validated_seed(seed) for seed in ensemble.member_seeds],
        "member_state_dicts": member_states,
        "param_counts": ensemble.parameters_count(),
        "dataset_hash": str(dataset_hash),
        "config_hash": str(config_hash),
        "git_commit": _git_commit(),
        "member_training_metrics": [
            {key: value for key, value in metrics.items()}
            for metrics in member_training_metrics
        ],
        "validation_metrics": dict(validation_metrics or {}),
        "payload_sha256": "",
    }
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    payload["payload_sha256"] = hashlib.sha256(_canonical_json({
        key: (value if key != "member_state_dicts" else "state_dicts")
        for key, value in body.items()
    })).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return payload


def load_world_model(
        checkpoint_path: str | Path,
        *,
        expected_dataset_hash: str | None = None,
        expected_config_hash: str | None = None) -> tuple[EnsembleDynamicsModel, dict[str, Any]]:
    """Load and validate a v4 world-model checkpoint."""
    path = Path(checkpoint_path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, ValueError) as exc:
        raise ValueError(f"invalid world-model checkpoint: {path}") from exc
    if not isinstance(payload, dict) or payload.get("kind") != WM_CHECKPOINT_KIND:
        raise ValueError("checkpoint is not a v4 world_model checkpoint")
    if payload.get("protocol_version") != WM_PROTOCOL_VERSION:
        raise ValueError("world-model protocol_version mismatch")
    for key in (
        "feature_dim", "state_dim", "history_len", "hidden_dim", "num_layers",
        "residual_hidden_dim", "members", "member_seeds", "member_state_dicts",
        "dataset_hash", "config_hash", "payload_sha256",
    ):
        if key not in payload:
            raise ValueError(f"world-model checkpoint missing field: {key}")
    if int(payload["feature_dim"]) != FEATURE_DIM or int(payload["state_dim"]) != STATE_DIM:
        raise ValueError("world-model checkpoint feature/state dims mismatch")
    member_count = int(payload["members"])
    member_seeds = [_validated_seed(seed) for seed in payload["member_seeds"]]
    if len(member_seeds) != member_count or len(payload["member_state_dicts"]) != member_count:
        raise ValueError("world-model checkpoint member count mismatch")
    members = []
    for state_dict in payload["member_state_dicts"]:
        member = ResidualDynamicsModel(
            feature_dim=int(payload["feature_dim"]),
            state_dim=int(payload["state_dim"]),
            history_len=int(payload["history_len"]),
            hidden_dim=int(payload["hidden_dim"]),
            num_layers=int(payload["num_layers"]),
            residual_hidden_dim=int(payload["residual_hidden_dim"]),
        )
        member.load_state_dict(state_dict, strict=True)
        member.eval()
        members.append(member)
    ensemble = EnsembleDynamicsModel(members, member_seeds)
    if expected_dataset_hash is not None and str(expected_dataset_hash) != str(payload["dataset_hash"]):
        raise ValueError("world-model checkpoint dataset_hash mismatch")
    if expected_config_hash is not None and str(expected_config_hash) != str(payload["config_hash"]):
        raise ValueError("world-model checkpoint config_hash mismatch")
    metadata = {key: value for key, value in payload.items() if key != "member_state_dicts"}
    metadata["checkpoint_path"] = str(path.resolve())
    return ensemble, metadata


__all__ = [
    "WM_PROTOCOL_VERSION",
    "WM_CHECKPOINT_KIND",
    "EnsembleDynamicsModel",
    "EnsemblePrediction",
    "ResidualDynamicsModel",
    "load_world_model",
    "save_world_model",
]
