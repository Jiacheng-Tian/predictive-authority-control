#!/usr/bin/env python3
"""Acceptance gate: fixed-TD3 checkpoints keep the backbone bitwise frozen.

Compares every ``backbone.*`` tensor in a residual-RL checkpoint against
the frozen supervised baseline checkpoint of the same model seed and fails on any
difference.  Used before formal evaluation of the retrained arms (P0
gate) and reusable for round-2 checkpoints (frozen blocks must still
match; pass --round 2 to check only the never-unfrozen tensors).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

# Blocks that stay frozen in every round under the preregistered protocol.
_ALWAYS_FROZEN_PREFIXES = (
    "backbone.input_proj.",
    "backbone.pos_embedding",
    "backbone.feature_",
)
_ROUND2_FROZEN_EXTRAS = (
    # rounds >= 2 thaw encoder.layers[-1] and backbone.head; every other
    # backbone tensor must remain bitwise identical to the baseline
    "backbone.encoder.norm",  # empty for the flat TransformerEncoder
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--round", type=int, default=1, choices=[1, 2])
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu",
                            weights_only=True)
    baseline = torch.load(arguments.baseline, map_location="cpu",
                          weights_only=True)
    baseline_sd = baseline.get(
        "model_state_dict", baseline.get("state_dict")
    )
    state = checkpoint["state_dict"]
    checked = 0
    for key, reference in baseline_sd.items():
        full = f"backbone.{key}"
        if full not in state:
            raise SystemExit(f"FAIL: checkpoint missing tensor {full}")
        if arguments.round == 1:
            must_match = True
        else:
            must_match = (
                full.startswith(_ALWAYS_FROZEN_PREFIXES)
                or full.startswith(_ROUND2_FROZEN_EXTRAS)
            )
            if full.startswith("backbone.encoder.layers."):
                layer_index = int(full.split(".")[3])
                last_index = None  # resolved below via key scan
                layer_ids = {
                    int(name.split(".")[3])
                    for name in state
                    if name.startswith("backbone.encoder.layers.")
                }
                last_index = max(layer_ids)
                must_match = layer_index != last_index
        if must_match:
            checked += 1
            if not torch.equal(state[full], reference):
                raise SystemExit(
                    f"FAIL: backbone tensor {full} changed during training"
                )
    print(f"OK: {checked} frozen backbone tensors bitwise identical "
          f"({arguments.checkpoint}, round {arguments.round})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
