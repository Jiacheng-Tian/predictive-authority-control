#!/usr/bin/env python3
"""Train and evaluate the capacity-matched pure-learning world-model control.

The pure control zeroes the stored physics baseline so the same
architecture, ensemble size, and training pipeline must predict the full
next state from history alone.  Comparing its one-step / multistep errors
against the physics+residual model isolates the structural contribution of
the physics prior from extra learning capacity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

from pac.v4.config import load_v4_config  # noqa: E402
from pac.v4.worldmodel.train import train_world_model  # noqa: E402
from pac.v4.worldmodel.validate import evaluate_world_model  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "pac_v4.yaml"))
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--reference-summary",
        required=True,
        help="one_step_summary.csv of the physics+residual validation to compare against",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    arguments = build_parser().parse_args(argv)
    config = load_v4_config(arguments.config)
    output = Path(arguments.out_dir).resolve()
    train_dir = output / "pure-control"
    validation_dir = output / "pure-control-validation"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    training = train_world_model(
        config,
        arguments.dataset_dir,
        train_dir,
        device=arguments.device,
        physics_baseline="zero",
        log=lambda message: print(message, file=sys.stderr, flush=True),
    )
    evaluate_world_model(
        config,
        arguments.dataset_dir,
        train_dir / "world_model.pt",
        validation_dir,
        device="cpu",
        ranking_splits=(),
        physics_baseline="zero",
        log=lambda message: print(message, file=sys.stderr, flush=True),
    )

    pure = pd.read_csv(validation_dir / "one_step_summary.csv")
    reference = pd.read_csv(arguments.reference_summary)
    merged = pure.merge(
        reference,
        on=["split", "group", "label"],
        suffixes=("_pure", "_physics_residual"),
    )
    merged["degradation_factor"] = (
        merged["ensemble_position_rmse_pure"]
        / merged["ensemble_position_rmse_physics_residual"]
    )
    merged.to_csv(output / "capacity_control_comparison.csv", index=False)
    interesting = merged[merged["group"].isin(["all", "family"])][[
        "split", "group", "label",
        "ensemble_position_rmse_pure",
        "ensemble_position_rmse_physics_residual",
        "degradation_factor",
    ]]
    print(interesting.to_string(index=False))
    (output / "summary.json").write_text(
        json.dumps({
            "training_checkpoint": training["checkpoint_path"],
            "physics_baseline": "zero",
            "reference_summary": str(Path(arguments.reference_summary).resolve()),
            "worst_degradation_factor": float(
                merged["degradation_factor"].max()
            ),
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
