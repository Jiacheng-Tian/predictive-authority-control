#!/usr/bin/env python3
"""Train the v4 direct-thruster RL control arm (one model seed)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Pin every numeric library to one thread per process (see v4_train_rl.py).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parents[1]

import torch  # noqa: E402

torch.set_num_threads(1)

from pac.v4.config import load_v4_config  # noqa: E402
from pac.v4.rl.direct import train_direct_rl  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "pac_v4.yaml"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument(
        "--max-env-steps", type=int, default=None,
        help="override the configured step budget (smoke runs)",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = load_v4_config(arguments.config)
    summary = train_direct_rl(
        config,
        arguments.out_dir,
        model_seed=arguments.model_seed,
        max_env_steps=arguments.max_env_steps,
        device=arguments.device,
        log=lambda message: print(message, file=sys.stderr, flush=True),
    )
    training = summary["training_summary"]
    print(json.dumps({
        "checkpoint": summary["checkpoint_path"],
        "model_seed": summary["model_seed"],
        "env_steps": training["env_steps"],
        "gradient_updates": training["gradient_updates"],
        "final_critic_loss": training["final_critic_loss"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
