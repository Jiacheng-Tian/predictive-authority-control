# -*- coding: utf-8 -*-
"""Collect full-vehicle trajectory timeseries for the motion-performance figures.

Replays two representative paired episodes with the frozen checkpoints:
  ID  : structured sinusoidal current, held-out environment seed 42000
  OOD : actuator delay + noise family, held-out environment seed 43000
Methods: SMC, MPC, fixed blend, supervised PAC, residual-RL PAC.
The simulator is deterministic given the seed, so regenerated episode metrics
must reproduce the archived case-study values (printed for verification).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from pac.v4.config import load_v4_config  # noqa: E402
from pac.v4.eval.formal import EpisodeTask, MethodFactory, build_task_spec  # noqa: E402
from pac.v4.eval.runner import run_v4_policy_episode  # noqa: E402

METHODS = ["smc", "mpc", "constant_alpha", "v3_transformer", "residual_rl"]

KEEP = ["method", "block", "family", "environment_seed", "episode_rmse", "time",
        "x", "y", "z", "target_x", "target_y", "target_z", "error", "z_error",
        "roll", "pitch", "yaw", "desired_roll", "desired_pitch", "desired_yaw",
        "authority_alpha"]


def main() -> int:
    os.chdir(ROOT)
    config = load_v4_config(ROOT / "config" / "pac_v4.yaml")
    factory = MethodFactory(
        config,
        rl_dir=ROOT / "runs/predictive_authority_v4/rl-round1",
        sspo_schedule={"startup_0_3s": 0.1, "pre_step_3_10s": 0.3,
                       "step_recovery_10_13s": 0.3, "post_step_13_20p9s": 0.3},
    )
    episodes = [
        ("seen", "structured", 2, 42000),
        ("unseen", "actuator_delay_noise",
         int(config.disturbances.base_scenario["actuator_delay_noise"]), 43000),
    ]
    frames = []
    for block, family, scenario, seed in episodes:
        task = EpisodeTask(block, family, scenario, seed,
                           "seen" if block == "seen" else "test")
        spec = build_task_spec(config, task)
        for method in METHODS:
            model_seed = int(config.seeds.model[0]) if method in (
                "v3_transformer", "residual_rl") else None
            policy = factory.policy_for(method, model_seed)
            metrics = run_v4_policy_episode(policy, spec, config)
            ts = metrics["ts"].copy()
            ts["method"] = method
            ts["block"] = block
            ts["family"] = family
            ts["environment_seed"] = seed
            ts["episode_rmse"] = float(metrics["rmse_3d"])
            frames.append(ts[KEEP])
            print(f"{block}/{family} seed={seed} {method}: "
                  f"rmse={metrics['rmse_3d']:.4f}", flush=True)
    out = Path(__file__).resolve().parent / "data"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out / "trajectory_cases.csv",
                                                index=False)
    print("trajectory cases written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
