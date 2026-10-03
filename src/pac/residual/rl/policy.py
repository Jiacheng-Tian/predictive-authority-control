"""Frozen-backbone residual alpha policy for v4 constrained TD3.

Deployment mapping (v4 plan):

    alpha_raw = alpha_backbone(z_t) + lambda_blend * delta_max * tanh(h(z_t, wm))

The head is zero-initialized so an untrained policy is bitwise the frozen
supervised backbone; round 1 freezes every backbone parameter, round 2 may unfreeze
the last encoder block and the backbone head.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch
from torch import nn

RL_PROTOCOL_VERSION = "pac_residual_runs"
RL_CHECKPOINT_KIND = "residual_rl_policy"
WM_FEATURE_DIM = 5
_ROUND2_UNFREEZE_BLOCKS = ("encoder_last", "head")


def unfreeze_round2_blocks(policy: "ResidualAlphaPolicy", blocks) -> None:
    """Thaw exactly the configured round-2 backbone blocks.

    ``encoder_last`` maps to ``backbone.encoder.layers[-1]`` and ``head``
    to ``backbone.head``; every other backbone parameter stays frozen at
    its checkpoint value.  Unknown block names raise so a typo in the
    config cannot silently unfreeze nothing (or everything).
    """
    for name in blocks:
        if name == "encoder_last":
            target = policy.backbone.encoder.layers[-1]
        elif name == "head":
            target = policy.backbone.head
        else:
            raise ValueError(
                f"unknown rl.unfreeze_blocks_round2 entry: {name!r} "
                f"(expected any of {_ROUND2_UNFREEZE_BLOCKS})"
            )
        for parameter in target.parameters():
            parameter.requires_grad_(True)


class ResidualAlphaPolicy(nn.Module):
    """Residual head on top of the frozen ``TemporalAlphaTransformer``."""

    def __init__(
            self,
            backbone: nn.Module,
            *,
            delta_max: float,
            lambda_blend: float,
            hidden_dim: int = 64,
            wm_feature_dim: int = 0,
            freeze_backbone: bool = True):
        super().__init__()
        if int(backbone.input_dim) != 24:
            raise ValueError("residual policy requires a 24-dim backbone")
        self.backbone = backbone
        self.delta_max = float(delta_max)
        self.lambda_blend = float(lambda_blend)
        self.wm_feature_dim = int(wm_feature_dim)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not bool(freeze_backbone))
        embed_dim = int(backbone.input_proj.out_features)
        head_input = embed_dim + self.wm_feature_dim
        self.head = nn.Sequential(
            nn.Linear(head_input, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    # -- backbone plumbing -------------------------------------------------
    def _embed(self, windows: torch.Tensor) -> torch.Tensor:
        """Replicate ``TemporalAlphaTransformer`` up to the encoder output."""
        backbone = self.backbone
        x = windows
        if x.ndim != 3 or x.shape[-1] != int(backbone.input_dim):
            raise ValueError("expected windows with shape (B, L, 24)")
        if x.shape[1] > int(backbone.history_len):
            x = x[:, -int(backbone.history_len):, :]
        normalized = (x - backbone.feature_mean) / backbone.feature_scale.clamp_min(1.0e-6)
        embedding = backbone.input_proj(normalized)
        embedding = embedding + backbone.pos_embedding[:, -embedding.shape[1]:, :]
        encoded = backbone.encoder(embedding)
        return encoded[:, -1, :]

    def backbone_alpha(self, windows: torch.Tensor) -> torch.Tensor:
        """Raw backbone alpha (before gain/filter)."""
        with torch.no_grad():
            embedding = self._embed(windows)
            return self.backbone.head(embedding).squeeze(-1)

    def forward(
            self,
            windows: torch.Tensor,
            wm_features: torch.Tensor | None = None) -> torch.Tensor:
        """Residual delta in ``[-delta_max, delta_max]`` for a batch."""
        embedding = self._embed(windows)
        if self.wm_feature_dim:
            if wm_features is None:
                wm_features = torch.zeros(
                    windows.shape[0], self.wm_feature_dim,
                    dtype=embedding.dtype, device=embedding.device,
                )
            elif wm_features.shape != (windows.shape[0], self.wm_feature_dim):
                raise ValueError(
                    f"wm_features must have shape (B, {self.wm_feature_dim})"
                )
            embedding = torch.cat([embedding, wm_features.to(embedding.dtype)], dim=1)
        return self.delta_max * torch.tanh(self.head(embedding)).squeeze(-1)

    def raw_alpha(
            self,
            windows: torch.Tensor,
            wm_features: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Deployed raw alpha and the residual delta."""
        delta = self.forward(windows, wm_features)
        base = self.backbone_alpha(windows)
        return base + self.lambda_blend * delta, delta


def build_wm_features(advice) -> np.ndarray:
    """Encode a :class:`~pac.residual.eval.runner.WMAdvice` into head features.

    Layout (5): ``[log1p(best_cost), log1p(spread), log10(1+uncertainty),
    gate_open, advice_valid]``.  ``None`` advice yields the invalid vector.
    """
    if advice is None:
        return np.zeros(WM_FEATURE_DIM, dtype=np.float32)
    # Cost fields can be NaN when the uncertainty gate is closed (no
    # rollout was run); they are floored to 0 so the features stay finite
    # while the gate_open flag carries the semantic difference.
    def _finite(value: float) -> float:
        number = float(value)
        return number if np.isfinite(number) else 0.0

    values = [
        np.log1p(max(_finite(advice.best_cost), 0.0)),
        np.log1p(max(_finite(advice.cost_spread), 0.0)),
        np.log10(1.0 + max(_finite(advice.uncertainty), 0.0)),
        1.0 if bool(advice.gate_open) else 0.0,
        1.0,
    ]
    return np.asarray(values, dtype=np.float32)


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[4]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_rl_checkpoint(
        checkpoint_path: str | Path,
        policy: ResidualAlphaPolicy,
        *,
        model_seed: int,
        round_index: int,
        backbone_checkpoint: str,
        reward_version: str,
        wm_version: str,
        training_summary: dict[str, Any]) -> dict[str, Any]:
    """Persist the residual policy with its provenance contract."""
    target = Path(checkpoint_path)
    if target.exists():
        raise FileExistsError(f"RL checkpoint already exists: {target}")
    payload = {
        "protocol_version": RL_PROTOCOL_VERSION,
        "kind": RL_CHECKPOINT_KIND,
        "model_seed": int(model_seed),
        "round": int(round_index),
        "backbone_checkpoint": str(backbone_checkpoint),
        "delta_max": float(policy.delta_max),
        "lambda_blend": float(policy.lambda_blend),
        "wm_feature_dim": int(policy.wm_feature_dim),
        "freeze_backbone": bool(not any(
            p.requires_grad for p in policy.backbone.parameters()
        )),
        "reward_version": str(reward_version),
        "wm_version": str(wm_version),
        "controller_pair": "real10kg_smc_steady+real10kg_mpc_ltv_v3",
        "alpha_limits": {
            "gain": 1.2, "smoothing": 0.5, "rate_limit": 0.0125, "deadband": 0.0,
        },
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
        },
        "param_count": int(sum(p.numel() for p in policy.parameters())),
        "training_summary": dict(training_summary),
        "git_commit": _git_commit(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return payload


def load_rl_checkpoint(
        checkpoint_path: str | Path,
        *,
        backbone: nn.Module,
        expected_model_seed: int | None = None) -> tuple[ResidualAlphaPolicy, dict[str, Any]]:
    """Load and validate a residual RL checkpoint against its backbone."""
    path = Path(checkpoint_path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, ValueError) as exc:
        raise ValueError(f"invalid RL checkpoint: {path}") from exc
    if not isinstance(payload, dict) or payload.get("kind") != RL_CHECKPOINT_KIND:
        raise ValueError("checkpoint is not a residual RL policy")
    for key in (
        "model_seed", "round", "delta_max", "lambda_blend", "wm_feature_dim",
        "state_dict", "training_summary", "reward_version", "wm_version",
    ):
        if key not in payload:
            raise ValueError(f"RL checkpoint missing field: {key}")
    if expected_model_seed is not None and int(payload["model_seed"]) != int(expected_model_seed):
        raise ValueError("RL checkpoint model_seed mismatch")
    policy = ResidualAlphaPolicy(
        backbone,
        delta_max=float(payload["delta_max"]),
        lambda_blend=float(payload["lambda_blend"]),
        wm_feature_dim=int(payload["wm_feature_dim"]),
        freeze_backbone=bool(payload.get("freeze_backbone", True)),
    )
    policy.load_state_dict(payload["state_dict"], strict=True)
    policy.eval()
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    metadata["checkpoint_path"] = str(path.resolve())
    return policy, metadata


__all__ = [
    "RL_CHECKPOINT_KIND",
    "RL_PROTOCOL_VERSION",
    "WM_FEATURE_DIM",
    "ResidualAlphaPolicy",
    "build_wm_features",
    "load_rl_checkpoint",
    "save_rl_checkpoint",
    "unfreeze_round2_blocks",
]
