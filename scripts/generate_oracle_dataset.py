#!/usr/bin/env python3
"""Generate the immutable v3 rollout-oracle dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from pac.authority.dataset import save_oracle_dataset
from pac.authority.training import collect_teacher_dataset_v3
from pac.experiment_config import load_v3_config


def _plan(config, profile: str) -> dict[str, object]:
    profile = str(profile).lower()
    if profile == "dry":
        return {
            "profile": "dry",
            "scenarios": list(config.environment.scenarios),
            "train_seeds": list(config.training.oracle_train_seeds),
            "val_seeds": list(config.training.oracle_val_seeds),
            "steps": config.environment.steps,
            "episodes": len(config.environment.scenarios) * (
                len(config.training.oracle_train_seeds)
                + len(config.training.oracle_val_seeds)
            ),
            "sample_bound": config.environment.steps * len(config.environment.scenarios) * (
                len(config.training.oracle_train_seeds)
                + len(config.training.oracle_val_seeds)
            ),
        }
    if profile == "short":
        return {
            "profile": "short",
            "scenarios": [1],
            "train_seeds": [11000],
            "val_seeds": [12000],
            "steps": 20,
            "episodes": 2,
            "sample_bound": 40,
        }
    if profile == "formal":
        return {
            "profile": "formal",
            "scenarios": list(config.environment.scenarios),
            "train_seeds": list(config.training.oracle_train_seeds),
            "val_seeds": list(config.training.oracle_val_seeds),
            "steps": config.environment.steps,
            "episodes": len(config.environment.scenarios) * (
                len(config.training.oracle_train_seeds)
                + len(config.training.oracle_val_seeds)
            ),
            "sample_bound": config.environment.steps * len(config.environment.scenarios) * (
                len(config.training.oracle_train_seeds)
                + len(config.training.oracle_val_seeds)
            ),
        }
    raise ValueError("profile must be dry, short, or formal")


def _reject_output_path(path: Path) -> None:
    if any(part.lower() == "results" for part in path.parts):
        raise ValueError("oracle dataset output must not be under a results path")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "pac_v3.yaml"))
    parser.add_argument("--profile", choices=("dry", "short", "formal"), default="dry")
    parser.add_argument("--out-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).resolve()
    output = Path(args.out_dir).resolve()
    _reject_output_path(output)
    config = load_v3_config(config_path)
    plan = _plan(config, args.profile)
    if args.profile == "dry":
        print(json.dumps(plan, sort_keys=True))
        return 0

    dataset = collect_teacher_dataset_v3(config, args.profile)
    metadata = dataset.metadata
    provenance = {
        "config_hash": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "config_path": str(config_path),
        "oracle": asdict(config.oracle),
        "seed_partitions": {
            "train": list(config.training.oracle_train_seeds),
            "val": list(config.training.oracle_val_seeds),
        },
        "profile": args.profile,
    }
    manifest = save_oracle_dataset(dataset, output, provenance)
    print(json.dumps({
        "profile": args.profile,
        "out_dir": str(output),
        "sample_count": manifest["sample_count"],
        "episodes": int(metadata["episode_uid"].nunique()),
        "dataset_hash": manifest["dataset_hash"],
        "files": manifest["file_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
