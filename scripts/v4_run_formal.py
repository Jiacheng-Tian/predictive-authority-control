#!/usr/bin/env python3
"""Run the paired formal v4 evaluation grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

from pac.v4.eval.formal import run_v4_formal  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "pac_v4.yaml"))
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--rl-dir", required=True)
    parser.add_argument("--wm-checkpoint", required=True)
    parser.add_argument("--sspo-schedule", required=True)
    parser.add_argument("--output-root", default=str(ROOT / "runs" / "predictive_authority_v4" / "formal"))
    parser.add_argument("--profile", default="dry", choices=["dry", "short", "formal"])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--include-wm-rl", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_v4_formal(
        config_path=arguments.config,
        dataset_dir=arguments.dataset_dir,
        rl_dir=arguments.rl_dir,
        wm_checkpoint=arguments.wm_checkpoint,
        sspo_schedule_path=arguments.sspo_schedule,
        output_root=arguments.output_root,
        profile=arguments.profile,
        run_id=arguments.run_id,
        include_wm_rl=arguments.include_wm_rl,
        progress_every=arguments.progress_every,
        log=lambda message: print(message, file=sys.stderr, flush=True),
    )
    print(json.dumps(result if isinstance(result, dict) else str(result),
                     indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
