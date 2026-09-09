#!/usr/bin/env python3
"""Run the paired, simulation-only formal true-MPC v3 evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pac.evaluation.formal_v3 import run_formal_v3


ROOT = Path(__file__).resolve().parents[1]


def _config_path(value: str) -> Path:
    if str(value).strip().lower() in {"pac_v3", "pac_v3.yaml"}:
        return ROOT / "config" / "pac_v3.yaml"
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pac_v3")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--profile", choices=("dry", "short", "formal"), required=True)
    parser.add_argument("--progress-every", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_formal_v3(
        config_path=_config_path(args.config),
        dataset_dir=args.dataset_dir,
        checkpoints_dir=args.checkpoints_dir,
        output_root=args.output_root,
        profile=args.profile,
        run_id=args.run_id,
        progress_every=max(0, int(args.progress_every)),
    )
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps({"run_dir": str(result)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
