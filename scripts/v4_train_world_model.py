#!/usr/bin/env python3
"""Train the v4 physics + residual world-model ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from pac.v4.config import load_v4_config  # noqa: E402
from pac.v4.worldmodel.train import train_world_model  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "pac_v4.yaml"))
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = load_v4_config(arguments.config)
    output = Path(arguments.out_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = train_world_model(
        config,
        arguments.dataset_dir,
        output,
        device=arguments.device,
        log=lambda message: print(message, file=sys.stderr, flush=True),
    )
    print(json.dumps({
        "checkpoint": summary["checkpoint_path"],
        "members": summary["members"],
        "train_rows": summary["train_rows"],
        "val_rows": summary["val_rows"],
        "test_rows": summary["test_rows"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
