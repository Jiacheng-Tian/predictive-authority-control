#!/usr/bin/env python3
"""Sweep the ensemble state-residual scale on the validation split.

The v4 plan mandates a conservative fallback for out-of-distribution states.
This sweep measures one-step prediction and candidate ranking as the residual
is shrunk (1.0 = full ensemble residual, 0.0 = physics + current forecast) so
the operating scale can be selected on validation data only.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from pac.v4.collector import load_transition_dataset
from pac.v4.config import load_v4_config
from pac.v4.worldmodel.data import build_transition_windows, windows_to_torch
from pac.v4.worldmodel.model import load_world_model
from pac.v4.worldmodel.ranking import (
    CandidateRolloutEvaluator,
    RankingResult,
    ranking_row,
    select_ranking_windows,
)
from pac.v4.worldmodel.train import dataset_content_hash, split_indices
from pac.v4.worldmodel.validate import one_step_evaluation


class ScaledEnsemble:
    """Deep copy of the ensemble with the state-residual scale reduced.

    Scaling the ``residual_scale`` buffer's state entries shrinks the
    denormalized state residual everywhere (forward, predict_delta, and the
    one-step evaluation's own denormalization) while leaving the current
    head untouched.
    """

    def __init__(self, members, scale: float):
        import copy

        self.members = []
        for member in members:
            clone = copy.deepcopy(member)
            with torch.no_grad():
                buffer = clone.residual_scale.clone()
                buffer[: clone.state_dim] = buffer[: clone.state_dim] * float(scale)
                clone.residual_scale.copy_(buffer)
            clone.eval()
            self.members.append(clone)
        self.size = len(self.members)

    def predict_delta(self, windows):
        states, currents = [], []
        with torch.no_grad():
            for member in self.members:
                delta_state, delta_current = member.predict_delta(windows)
                states.append(delta_state)
                currents.append(delta_current)
        return torch.stack(states).mean(dim=0), torch.stack(currents).mean(dim=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pac_v4.yaml")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--scales", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--ranking-windows", type=int, default=60)
    args = parser.parse_args()

    config = load_v4_config(args.config)
    dataset = load_transition_dataset(args.dataset_dir)
    dataset_hash = dataset_content_hash(args.dataset_dir)
    ensemble, _ = load_world_model(args.checkpoint, expected_dataset_hash=dataset_hash)
    ensemble.eval()
    scales = [float(value) for value in args.scales.split(",")]

    # --- one-step evaluation per scale -----------------------------------
    indices = split_indices(dataset.metadata, args.split)
    built = build_transition_windows(dataset, int(config.world_model.history_len))
    rows = windows_to_torch(built, indices)
    group_info = dataset.metadata.iloc[indices].reset_index(drop=True)
    physics_rmse = None
    one_step_rows = []
    for scale in scales:
        frame = one_step_evaluation(
            ScaledEnsemble(ensemble.members, scale), rows, group_info,
            device=torch.device("cpu"),
        )
        if physics_rmse is None:
            physics_rmse = float(np.sqrt(np.mean(frame["physics_position_error"] ** 2)))
        ensemble_rmse = float(np.sqrt(np.mean(frame["ensemble_position_error"] ** 2)))
        one_step_rows.append({
            "scale": scale,
            "ensemble_position_rmse": ensemble_rmse,
            "physics_position_rmse": physics_rmse,
            "relative_improvement": 1.0 - ensemble_rmse / physics_rmse,
        })
    print("=== one-step (validation) ===")
    print(pd.DataFrame(one_step_rows).to_string(index=False))

    # --- ranking per scale ------------------------------------------------
    windows = select_ranking_windows(
        dataset,
        split=args.split,
        windows_per_episode=1,
        horizon=int(config.oracle.horizon),
        history_len=int(config.world_model.history_len),
    )
    cap = int(args.ranking_windows)
    if len(windows) > cap:
        step = max(1, len(windows) // cap)
        windows = windows[::step][:cap]
    evaluator = CandidateRolloutEvaluator(config, device="cpu")
    true_costs_cache = [evaluator.true_oracle_costs(window) for window in windows]
    ranking_rows = []
    for scale in scales:
        scaled = ScaledEnsemble(ensemble.members, scale)
        costs_per_window = [
            np.mean([
                evaluator.single_model_costs(member, window)
                for member in scaled.members
            ], axis=0)
            for window in windows
        ]
        for window, true_costs, costs in zip(windows, true_costs_cache, costs_per_window):
            result = RankingResult(true_costs, costs, "wm")
            ranking_rows.append(ranking_row(result, scale=scale, family=window.family))
    frame = pd.DataFrame(ranking_rows)
    print("=== ranking (validation) ===")
    summary = frame.groupby("scale").agg(
        spearman_mean=("spearman", "mean"),
        top1=("top1_correct", "mean"),
        regret_mean=("regret", "mean"),
    )
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
