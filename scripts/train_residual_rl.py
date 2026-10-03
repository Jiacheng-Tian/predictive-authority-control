#!/usr/bin/env python3
"""Train the v4 constrained TD3 residual policy (one model seed)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Pin every numeric library to one thread per process: five training
# processes each defaulting to all-core torch/BLAS threads oversubscribe
# the CPU for hours and have triggered hardware-corrected errors (WHEA 17)
# followed by an automatic reboot on this machine.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parents[1]

import torch  # noqa: E402

torch.set_num_threads(1)

from pac.residual.config import load_residual_config  # noqa: E402
from pac.residual.rl.train import train_residual_rl  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "pac_residual.yaml"))
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--round", type=int, default=1, choices=[1, 2])
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="round-1 checkpoint to continue from when running round 2",
    )
    parser.add_argument(
        "--wm-checkpoint",
        default=None,
        help="stage-one world-model checkpoint; enables the WM-feature head (v4-WM-RL)",
    )
    parser.add_argument(
        "--max-env-steps", type=int, default=None,
        help="override the configured step budget (smoke runs)",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = load_residual_config(arguments.config)
    summary = train_residual_rl(
        config,
        arguments.dataset_dir,
        arguments.out_dir,
        model_seed=arguments.model_seed,
        round_index=arguments.round,
        max_env_steps=arguments.max_env_steps,
        resume_checkpoint=arguments.resume_checkpoint,
        wm_checkpoint=arguments.wm_checkpoint,
        device=arguments.device,
        log=lambda message: print(message, file=sys.stderr, flush=True),
    )
    training = summary["training_summary"]
    print(json.dumps({
        "checkpoint": summary["checkpoint_path"] if "checkpoint_path" in summary
        else training.get("checkpoint"),
        "model_seed": summary["model_seed"],
        "round": summary["round"],
        "env_steps": training["env_steps"],
        "gradient_updates": training["gradient_updates"],
        "final_critic_loss": training["final_critic_loss"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
