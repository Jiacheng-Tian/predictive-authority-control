"""Conservative anchor wrapper for the residual base alpha (Stage B)."""

from __future__ import annotations

import torch
from torch import nn


class AnchoredBackbone(nn.Module):
    """Module wrapper exposing ``c + T*(backbone(x) - c)`` as its alpha.

    Used where downstream code calls the raw backbone module for base
    alphas (warm-start materialization), so anchored and identity runs
    share one code path.
    """

    def __init__(self, backbone: nn.Module, *, center: float, temperature: float):
        super().__init__()
        self.backbone = backbone
        self.center = float(center)
        self.temperature = float(temperature)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        raw = self.backbone(windows)
        if self.center == 0.5 and self.temperature == 1.0:
            return raw
        return self.center + self.temperature * (raw - self.center)
