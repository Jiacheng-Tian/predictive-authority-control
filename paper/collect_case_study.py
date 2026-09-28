"""Collect case-study timeseries on the unseen actuator-delay family.

Runs MPC, the frozen v3 Transformer, and v4 residual RL on two unseen
actuator-delay episodes with full timeseries so the mechanism figure can
show how RL suppresses MPC authority when the actuators degrade.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from pac.v4.config import load_v4_config  # noqa: E402
from pac.v4.eval.formal import (  # noqa: E402
    EpisodeTask,
    MethodFactory,
    build_task_spec,
)
from pac.v4.eval.runner import run_v4_policy_episode  # noqa: E402

METHODS = ["mpc", "v3_transformer", "residual_rl"]


def main() -> int:
    os.chdir(ROOT)
    config = load_v4_config(ROOT / "config" / "pac_v4.yaml")
    factory = MethodFactory(
        config,
        rl_dir=ROOT / "runs/predictive_authority_v4/rl-round1",
        sspo_schedule={"startup_0_3s": 0.1, "pre_step_3_10s": 0.3,
                       "step_recovery_10_13s": 0.3, "post_step_13_20p9s": 0.3},
    )
    frames = []
    for seed in (43000, 43001):
        task = EpisodeTask("unseen", "actuator_delay_noise",
                           int(config.disturbances.base_scenario["actuator_delay_noise"]),
                           seed, "test")
        spec = build_task_spec(config, task)
        for method in METHODS:
            model_seed = int(config.seeds.model[0]) if method != "mpc" else None
            policy = factory.policy_for(method, model_seed)
            metrics = run_v4_policy_episode(policy, spec, config)
            timeseries = metrics["ts"].copy()
            timeseries["method"] = method
            timeseries["environment_seed"] = seed
            timeseries["episode_rmse"] = float(metrics["rmse_3d"])
            keep = ["method", "environment_seed", "episode_rmse", "time",
                    "error", "authority_alpha", "heading",
                    "desired_heading"] + [
                f"requested_action_{i}" for i in range(6)] + [
                f"applied_action_{i}" for i in range(6)] + [
                f"true_current_{i}" for i in range(3)]
            frames.append(timeseries[keep])
            print(f"{method} seed={seed}: rmse={metrics['rmse_3d']:.4f}", flush=True)
    out = Path(__file__).resolve().parent / "data"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out / "case_actuator_delay.csv",
                                                index=False)
    print("case study written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
