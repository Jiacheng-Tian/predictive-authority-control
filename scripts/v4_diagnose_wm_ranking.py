#!/usr/bin/env python3
"""Diagnose why the ensemble ranking underperforms physics persistence.

Compares four scorer variants on a small validation sample:
  1. ensemble mean (as in the gate report)
  2. member0 only (deterministic residual)
  3. member median (robust aggregation)
  4. ensemble mean with bounded current-forecast drift
plus a residual-ablation: zero state residual with the WM current forecast.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pac.v4.collector import load_transition_dataset
from pac.v4.config import load_v4_config
from pac.v4.worldmodel.model import load_world_model
from pac.v4.worldmodel.ranking import (
    CandidateRolloutEvaluator,
    RankingResult,
    physics_persistence_costs,
    ranking_row,
    select_ranking_windows,
)
from pac.v4.worldmodel.train import dataset_content_hash


class ClippedCurrentModel:
    """Wraps a member; zeros state residual unless enabled, clips current."""

    def __init__(self, member, *, state_scale: float = 1.0, current_limit: float | None = None):
        self.member = member
        self.state_scale = float(state_scale)
        self.current_limit = current_limit

    def predict_delta(self, windows):
        delta_state, delta_current = self.member.predict_delta(windows)
        delta_state = delta_state * self.state_scale
        if self.current_limit is not None:
            delta_current = torch.clamp(
                delta_current, -float(self.current_limit), float(self.current_limit)
            )
        return delta_state, delta_current


def cumulative_clipped(member, limit: float):
    """Return a model whose per-step current delta is clipped to +-limit."""
    return ClippedCurrentModel(member, state_scale=1.0, current_limit=limit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pac_v4.yaml")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--windows", type=int, default=50)
    args = parser.parse_args()

    config = load_v4_config(args.config)
    dataset = load_transition_dataset(args.dataset_dir)
    dataset_hash = dataset_content_hash(args.dataset_dir)
    ensemble, _metadata = load_world_model(args.checkpoint, expected_dataset_hash=dataset_hash)
    ensemble.eval()

    windows = select_ranking_windows(
        dataset,
        split=args.split,
        windows_per_episode=1,
        horizon=int(config.oracle.horizon),
        history_len=int(config.world_model.history_len),
    )
    if len(windows) > int(args.windows):
        step = max(1, len(windows) // int(args.windows))
        windows = windows[::step][: int(args.windows)]

    evaluator = CandidateRolloutEvaluator(config, device="cpu")
    scorers = {
        "persistence": lambda w: physics_persistence_costs(evaluator, w),
        "ensemble_mean": lambda w: np.mean([
            evaluator.single_model_costs(member, w) for member in ensemble.members
        ], axis=0),
        "member0": lambda w: evaluator.single_model_costs(ensemble.members[0], w),
        "member_median": lambda w: np.median([
            evaluator.single_model_costs(member, w) for member in ensemble.members
        ], axis=0),
        "ensemble_clip_current_0.05": lambda w: np.mean([
            evaluator.single_model_costs(cumulative_clipped(m, 0.05), w)
            for m in ensemble.members
        ], axis=0),
        "ensemble_no_state_residual": lambda w: np.mean([
            evaluator.single_model_costs(
                ClippedCurrentModel(m, state_scale=0.0, current_limit=0.05), w
            )
            for m in ensemble.members
        ], axis=0),
    }

    rows = []
    for index, window in enumerate(windows):
        true_costs = evaluator.true_oracle_costs(window)
        for name, scorer in scorers.items():
            result = RankingResult(true_costs, scorer(window), name)
            rows.append(ranking_row(
                result,
                family=window.family,
                scenario_id=window.scenario_id,
                step=window.step,
            ))
        if (index + 1) % 10 == 0:
            print(f"{index + 1}/{len(windows)}", flush=True)

    import pandas as pd

    frame = pd.DataFrame(rows)
    summary = frame.groupby("scorer").agg(
        spearman_mean=("spearman", "mean"),
        top1=("top1_correct", "mean"),
        regret_mean=("regret", "mean"),
        windows=("regret", "size"),
    )
    print(summary.to_string())
    print()
    print("=== per family (top1) ===")
    print(frame.pivot_table(index="scorer", columns="family", values="top1_correct").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
