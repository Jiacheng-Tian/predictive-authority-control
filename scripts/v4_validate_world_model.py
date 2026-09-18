#!/usr/bin/env python3
"""Run stage-1 world-model validation and produce the gate report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from pac.v4.config import load_v4_config  # noqa: E402
from pac.v4.worldmodel.validate import nan_to_none, evaluate_world_model  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "pac_v4.yaml"))
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--ranking-splits", default="val,test",
        help="comma-separated splits for the (expensive) ranking evaluation",
    )
    parser.add_argument(
        "--max-ranking-windows", type=int, default=None,
        help="optional cap on ranking windows per split (debug/time control)",
    )
    parser.add_argument(
        "--gates-only-ranking", action="store_true",
        help="restrict ranking evaluation to the test split",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = load_v4_config(arguments.config)
    ranking_splits = tuple(
        split.strip() for split in arguments.ranking_splits.split(",") if split.strip()
    )
    if arguments.gates_only_ranking:
        ranking_splits = ("test",)
    report = evaluate_world_model(
        config,
        arguments.dataset_dir,
        arguments.checkpoint,
        arguments.out_dir,
        device=arguments.device,
        ranking_splits=ranking_splits,
        max_ranking_windows=arguments.max_ranking_windows,
        log=lambda message: print(message, file=sys.stderr, flush=True),
    )
    gates = report["gates"]
    print(json.dumps(nan_to_none({
        name: {
            "pass": value["pass"],
            **{
                key: item for key, item in value.items()
                if key not in {"pass", "description"}
            },
        }
        for name, value in gates.items()
    }), sort_keys=True, allow_nan=False))
    return 0 if all(value["pass"] for value in gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
