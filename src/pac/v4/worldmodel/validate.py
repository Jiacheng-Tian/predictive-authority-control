"""Stage-1 validation for the v4 world model.

Produces the acceptance evidence required by the v4 plan:

* one-step state prediction versus the physics-only baseline
  (physics / deterministic member / ensemble mean);
* 10- and 20-step rollout error on validation and unseen-disturbance test
  splits (teacher-forced actions, free-running states);
* ensemble uncertainty versus actual error (correlation, interval coverage,
  regime identification for stepchange recovery, stochastic currents, and
  parameter mismatch);
* candidate-alpha ranking agreement on the 11-point grid (true oracle under
  realized currents versus world-model and physics-persistence scorers);
* the five stage-1 acceptance gates as explicit booleans.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from pac.v4.worldmodel.data import (
    FEATURE_DIM,
    STATE_DIM,
    assemble_wm_feature,
    build_transition_windows,
    windows_to_torch,
)
from pac.v4.worldmodel.model import EnsembleDynamicsModel, load_world_model
from pac.v4.worldmodel.ranking import (
    CandidateRolloutEvaluator,
    RankingResult,
    physics_persistence_costs,
    ranking_row,
    select_ranking_windows,
)
from pac.v4.worldmodel.torch_dynamics import TorchAUVDynamics
from pac.v4.collector import load_transition_dataset
from pac.v4.worldmodel.train import dataset_content_hash, split_indices

_STEPCHANGE_RECOVERY = (1000, 1300)   # 10 s - 13 s
_STEPCHANGE_PRE = (300, 1000)         # 3 s - 10 s
_STOCHASTIC_FAMILIES = {"ou_current", "colored_noise"}
_EVAL_BATCH = 4096
_ONE_STEP_IMPROVEMENT_THRESHOLD = 0.10
_MULTISTEP_STABILITY_FACTOR = 2.0
_UNCERTAINTY_CORRELATION_THRESHOLD = 0.2


class _ZeroDelta:
    """Physics-only predictor: zero residual, zero current increment."""

    def predict_delta(self, windows: torch.Tensor):
        batch = windows.shape[0]
        return torch.zeros(batch, STATE_DIM), torch.zeros(batch, 2)


def _position_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(predicted[:, :3] - target[:, :3], axis=1)


def one_step_evaluation(
        ensemble: EnsembleDynamicsModel,
        rows: dict[str, torch.Tensor],
        group_info: pd.DataFrame,
        *,
        device: torch.device) -> pd.DataFrame:
    """One-step prediction errors for physics / member0 / ensemble mean."""
    windows = rows["windows"].to(device)
    physics_next = rows["physics_next_state"].numpy()
    next_state = rows["next_state"].numpy()

    physics_pos = _position_error(physics_next, next_state)
    physics_vel = np.linalg.norm(physics_next[:, 3:] - next_state[:, 3:], axis=1)

    def denormalized_state_delta(member, normalized: np.ndarray) -> np.ndarray:
        return (
            normalized[:, :STATE_DIM] * member.residual_scale[:STATE_DIM].numpy()
            + member.residual_mean[:STATE_DIM].numpy()
        )

    with torch.no_grad():
        chunks = [
            ensemble.members[0](
                windows[start:start + _EVAL_BATCH]
            ).cpu().numpy()
            for start in range(0, windows.shape[0], _EVAL_BATCH)
        ]
    member0_delta = denormalized_state_delta(
        ensemble.members[0], np.concatenate(chunks, axis=0)
    )
    member0_next = physics_next + member0_delta

    ensemble_mean_delta = np.zeros_like(next_state)
    ensemble_sq_delta = np.zeros_like(next_state)
    with torch.no_grad():
        for member in ensemble.members:
            chunks = [
                member(windows[start:start + _EVAL_BATCH]).cpu().numpy()
                for start in range(0, windows.shape[0], _EVAL_BATCH)
            ]
            delta = denormalized_state_delta(member, np.concatenate(chunks, axis=0))
            ensemble_mean_delta += delta
            ensemble_sq_delta += delta ** 2
    ensemble_mean_delta /= ensemble.size
    ensemble_var = np.maximum(
        ensemble_sq_delta / ensemble.size - ensemble_mean_delta ** 2, 0.0
    )
    ensemble_next = physics_next + ensemble_mean_delta
    ensemble_std = np.sqrt(ensemble_var)

    frame = pd.DataFrame({
        "family": group_info["family"].to_numpy(),
        "scenario_id": group_info["scenario_id"].to_numpy(),
        "behavior": group_info["behavior"].to_numpy(),
        "step": group_info["step"].to_numpy(),
        "physics_position_error": physics_pos,
        "member0_position_error": _position_error(member0_next, next_state),
        "ensemble_position_error": _position_error(ensemble_next, next_state),
        "physics_velocity_error": physics_vel,
        "member0_velocity_error": np.linalg.norm(
            member0_next[:, 3:] - next_state[:, 3:], axis=1
        ),
        "ensemble_velocity_error": np.linalg.norm(
            ensemble_next[:, 3:] - next_state[:, 3:], axis=1
        ),
        "ensemble_position_std": ensemble_std[:, :3].mean(axis=1),
        "ensemble_position_abs_error": np.abs(
            ensemble_next[:, :3] - next_state[:, :3]
        ).mean(axis=1),
    })
    for component in range(3):
        frame[f"ensemble_pos{component}_std"] = ensemble_std[:, component]
        frame[f"ensemble_pos{component}_abs_error"] = np.abs(
            ensemble_next[:, component] - next_state[:, component]
        )
    return frame


def _summarize_one_step(frame: pd.DataFrame, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def summarize(subset: pd.DataFrame, group: str, label: str) -> None:
        if subset.empty:
            return
        rows.append({
            "split": split,
            "group": group,
            "label": label,
            "rows": int(len(subset)),
            "physics_position_rmse": float(
                np.sqrt(np.mean(subset["physics_position_error"] ** 2))
            ),
            "member0_position_rmse": float(
                np.sqrt(np.mean(subset["member0_position_error"] ** 2))
            ),
            "ensemble_position_rmse": float(
                np.sqrt(np.mean(subset["ensemble_position_error"] ** 2))
            ),
            "physics_velocity_rmse": float(
                np.sqrt(np.mean(subset["physics_velocity_error"] ** 2))
            ),
            "ensemble_velocity_rmse": float(
                np.sqrt(np.mean(subset["ensemble_velocity_error"] ** 2))
            ),
            "ensemble_position_std_mean": float(subset["ensemble_position_std"].mean()),
        })

    summarize(frame, "all", "all")
    for family, subset in frame.groupby("family"):
        summarize(subset, "family", str(family))
    for scenario, subset in frame.groupby("scenario_id"):
        summarize(subset, "scenario", str(int(scenario)))
    return rows


def _select_rollout_starts(
        metadata: pd.DataFrame,
        *,
        split: str,
        horizon: int,
        windows_per_episode: int) -> list[int]:
    uids = metadata["episode_uid"].tolist()
    splits = metadata["split"].astype(str).to_numpy()
    starts: list[int] = []
    index = 0
    while index < len(uids):
        end = index
        while end < len(uids) and uids[end] == uids[index]:
            end += 1
        if splits[index] == str(split) and end - index > horizon:
            candidates = np.arange(index + 1, end - horizon)
            if candidates.size > 0:
                if candidates.size > int(windows_per_episode):
                    positions = np.linspace(
                        0, candidates.size - 1, int(windows_per_episode)
                    )
                    candidates = candidates[np.unique(np.round(positions).astype(int))]
                starts.extend(int(value) for value in candidates)
        index = end
    return starts


class _MeanEnsemblePredictor:
    """Batched ensemble-mean delta predictor."""

    def __init__(self, members: list[Any]):
        self.members = members

    def predict_delta(self, windows: torch.Tensor):
        if len(self.members) == 1:
            return self.members[0].predict_delta(windows)
        states, currents = [], []
        with torch.no_grad():
            for member in self.members:
                delta_state, delta_current = member.predict_delta(windows)
                states.append(delta_state)
                currents.append(delta_current)
        return (
            torch.stack(states).mean(dim=0),
            torch.stack(currents).mean(dim=0),
        )


def multistep_evaluation(
        config,
        ensemble: EnsembleDynamicsModel,
        dataset,
        torch_dynamics: TorchAUVDynamics,
        wrench_converter,
        *,
        split: str,
        device: torch.device,
        windows_per_episode: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Free-running rollout errors with recorded (teacher-forced) actions.

    Models roll out with the recorded applied actions; the ``physics_realized``
    reference additionally uses the realized current sequence (perfect current
    preview), while every other model must carry its own current forecast.
    """
    horizon = max(int(value) for value in config.world_model.horizons)
    metadata = dataset.metadata
    starts = _select_rollout_starts(
        metadata, split=split, horizon=horizon, windows_per_episode=windows_per_episode
    )
    if not starts:
        raise ValueError(f"no rollout starts available for split {split}")

    features_all = np.asarray(dataset.features, dtype=np.float32)
    state_all = np.asarray(dataset.state, dtype=np.float32)
    next_state_all = np.asarray(dataset.next_state, dtype=np.float32)
    applied_all = np.asarray(dataset.applied, dtype=np.float32)
    est_all = np.asarray(dataset.est_current, dtype=np.float32)
    true_all = np.asarray(dataset.true_current, dtype=np.float32)
    context_all = np.asarray(dataset.context, dtype=np.float32)

    history_len = int(config.world_model.history_len)
    starts_array = np.asarray(starts, dtype=np.int64)
    batch_states = state_all[starts_array]
    batch_windows = np.zeros((len(starts), history_len, FEATURE_DIM), dtype=np.float32)
    for index, row in enumerate(starts):
        window_start = row - history_len
        if window_start >= 0:
            batch_windows[index] = features_all[window_start:row]
        else:
            batch_windows[index][-row:] = features_all[0:row]
    batch_est = est_all[starts_array]
    batch_context = context_all[starts_array]
    batch_scenarios = metadata["scenario_id"].to_numpy()[starts_array].astype(int)
    applied_seq = np.stack([applied_all[starts_array + s] for s in range(horizon)])
    realized_seq = np.stack([true_all[starts_array + s] for s in range(horizon)])
    target_seq = np.stack([next_state_all[starts_array + s] for s in range(horizon)])

    torch_windows = torch.from_numpy(batch_windows).to(device)
    torch_states = torch.from_numpy(batch_states).to(device)
    applied_torch = torch.from_numpy(applied_seq).to(device)
    realized_torch = torch.from_numpy(realized_seq).to(device)
    targets = target_seq[:, :, :3]

    models = [
        (_ZeroDelta(), "physics_realized", True),
        (_ZeroDelta(), "physics_persistence", False),
        (_MeanEnsemblePredictor([ensemble.members[0]]), "member0", False),
        (_MeanEnsemblePredictor(list(ensemble.members)), "ensemble", False),
    ]
    error_rows: list[dict[str, Any]] = []
    for model, name, uses_realized_current in models:
        state = torch_states.clone()
        windows = torch_windows.clone()
        current_forecast = torch.from_numpy(batch_est.copy()).to(device)
        errors = np.zeros((horizon, len(starts)), dtype=np.float64)
        with torch.no_grad():
            for step in range(horizon):
                current_input = realized_torch[step] if uses_realized_current else current_forecast
                wrench = torch.stack([
                    torch.from_numpy(
                        np.asarray(wrench_converter(applied_torch[step, index].cpu().numpy()))
                    )
                    for index in range(len(starts))
                ]).to(device)
                eta_next, nu_next = torch_dynamics.predict_step(
                    state[:, :6], state[:, 6:], wrench, current_input
                )
                features = np.stack([
                    assemble_wm_feature(
                        state[index].cpu().numpy(),
                        applied_torch[step, index].cpu().numpy(),
                        applied_torch[step, index].cpu().numpy(),
                        current_input[index].cpu().numpy(),
                        batch_context[index],
                        batch_scenarios[index],
                    )
                    for index in range(len(starts))
                ])
                windows = torch.cat([
                    windows[:, 1:, :],
                    torch.from_numpy(features[:, None, :]).to(device),
                ], dim=1)
                delta_state, delta_current = model.predict_delta(windows)
                eta_next = eta_next + delta_state[:, :6].to(device)
                nu_next = nu_next + delta_state[:, 6:].to(device)
                state = torch.cat([eta_next, nu_next], dim=1)
                if not uses_realized_current:
                    current_forecast = current_forecast.clone()
                    current_forecast[:, :2] = current_forecast[:, :2] + delta_current.to(device)
                errors[step] = np.linalg.norm(
                    state[:, :3].cpu().numpy() - targets[step], axis=1
                )
        for step_index, horizon_step in enumerate(range(1, horizon + 1)):
            error_rows.append({
                "split": split,
                "model": name,
                "horizon_step": horizon_step,
                "position_rmse": float(np.sqrt(np.mean(errors[step_index] ** 2))),
                "position_mean_error": float(np.mean(errors[step_index])),
            })
    error_frame = pd.DataFrame(error_rows)
    summary_rows = []
    for model_name, subset in error_frame.groupby("model"):
        row: dict[str, Any] = {"split": split, "model": str(model_name)}
        for horizon_value in config.world_model.horizons:
            selected = subset[subset["horizon_step"] == int(horizon_value)]
            row[f"position_rmse_{int(horizon_value)}step"] = float(
                selected["position_rmse"].iloc[0]
            )
        row["position_rmse_mean"] = float(subset["position_rmse"].mean())
        summary_rows.append(row)
    return error_frame, pd.DataFrame(summary_rows)


def uncertainty_evaluation(one_step_frame: pd.DataFrame) -> dict[str, Any]:
    """Uncertainty calibration and regime identification statistics."""
    error = one_step_frame["ensemble_position_abs_error"].to_numpy(dtype=float)
    std = one_step_frame["ensemble_position_std"].to_numpy(dtype=float)
    if np.std(error) <= 1.0e-15 or np.std(std) <= 1.0e-15:
        correlation = float("nan")
    else:
        rank_error = pd.Series(error).rank().to_numpy(dtype=float)
        rank_std = pd.Series(std).rank().to_numpy(dtype=float)
        correlation = float(np.corrcoef(rank_error, rank_std)[0, 1])
    coverage: dict[str, float] = {}
    for z, name in ((1.0, "coverage_68"), (1.96, "coverage_95")):
        covered = np.all(
            one_step_frame[
                [f"ensemble_pos{component}_abs_error" for component in range(3)]
            ].to_numpy()
            <= z * one_step_frame[
                [f"ensemble_pos{component}_std" for component in range(3)]
            ].to_numpy(),
            axis=1,
        )
        coverage[name] = float(np.mean(covered))
    return {
        "uncertainty_error_spearman": correlation,
        **coverage,
        "std_mean": float(std.mean()),
    }


def regime_uncertainty(
        one_step_frame: pd.DataFrame,
        *,
        scenario_filter: int | None = None) -> dict[str, float]:
    frame = one_step_frame
    if scenario_filter is not None:
        frame = frame[frame["scenario_id"] == scenario_filter]
    stochastic = frame[frame["family"].isin(_STOCHASTIC_FAMILIES)]
    structured = frame[~frame["family"].isin(_STOCHASTIC_FAMILIES)]
    result = {
        "stochastic_std_mean": float(stochastic["ensemble_position_std"].mean())
        if not stochastic.empty else float("nan"),
        "structured_std_mean": float(structured["ensemble_position_std"].mean())
        if not structured.empty else float("nan"),
    }
    if scenario_filter == 3:
        step = frame["step"].to_numpy()
        recovery = frame[
            (step >= _STEPCHANGE_RECOVERY[0]) & (step < _STEPCHANGE_RECOVERY[1])
        ]
        pre = frame[(step >= _STEPCHANGE_PRE[0]) & (step < _STEPCHANGE_PRE[1])]
        result["stepchange_recovery_std_mean"] = (
            float(recovery["ensemble_position_std"].mean())
            if not recovery.empty else float("nan")
        )
        result["stepchange_prestep_std_mean"] = (
            float(pre["ensemble_position_std"].mean())
            if not pre.empty else float("nan")
        )
    return result


def ranking_evaluation(
        config,
        ensemble: EnsembleDynamicsModel,
        dataset,
        *,
        split: str,
        device: torch.device,
        max_windows: int | None = None,
        log=lambda message: None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Candidate-alpha ranking agreement on the 11-point grid."""
    wm_config = config.world_model
    windows = select_ranking_windows(
        dataset,
        split=split,
        windows_per_episode=int(wm_config.ranking_windows_per_episode),
        horizon=int(config.oracle.horizon),
        history_len=int(wm_config.history_len),
    )
    if max_windows is not None and len(windows) > int(max_windows):
        step = max(1, len(windows) // int(max_windows))
        windows = windows[::step][: int(max_windows)]
    evaluator = CandidateRolloutEvaluator(config, device=str(device))
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, window in enumerate(windows):
        true_costs = evaluator.true_oracle_costs(window)
        persistence_costs = physics_persistence_costs(evaluator, window)
        member_costs = np.stack([
            evaluator.single_model_costs(member, window)
            for member in ensemble.members
        ])
        wm_costs = member_costs.mean(axis=0)
        for scorer, costs in (
            ("physics_persistence", persistence_costs),
            ("wm", wm_costs),
        ):
            result = RankingResult(
                true_costs=true_costs, predicted_costs=costs, scorer=scorer
            )
            rows.append(ranking_row(
                result,
                split=split,
                episode_uid=window.episode_uid,
                family=window.family,
                behavior=window.behavior,
                scenario_id=window.scenario_id,
                step=window.step,
                ensemble_cost_std_mean=float(member_costs.std(axis=0).mean()),
            ))
        if log and (index == 0 or (index + 1) % 25 == 0):
            log(
                f"[wm-ranking {split}] {index + 1}/{len(windows)} "
                f"elapsed={time.perf_counter() - started:.0f}s"
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"ranking evaluation produced no windows for split {split}")
    summary_rows = []
    for scorer, subset in frame.groupby("scorer"):
        entry: dict[str, Any] = {"split": split, "scorer": str(scorer)}
        for label, family_subset in (
            ("all", subset),
            *[
                (f"family:{family}", family_frame)
                for family, family_frame in subset.groupby("family")
            ],
        ):
            spearman = family_subset["spearman"].to_numpy(dtype=float)
            spearman = spearman[np.isfinite(spearman)]
            entry[f"{label}_spearman_mean"] = (
                float(spearman.mean()) if spearman.size else float("nan")
            )
            entry[f"{label}_top1"] = float(family_subset["top1_correct"].mean())
            entry[f"{label}_regret_mean"] = float(family_subset["regret"].mean())
            entry[f"{label}_windows"] = int(len(family_subset))
        summary_rows.append(entry)
    return frame, pd.DataFrame(summary_rows)


def evaluate_world_model(
        config,
        dataset_dir: str | Path,
        checkpoint_path: str | Path,
        output_dir: str | Path,
        *,
        device: str = "cpu",
        ranking_splits: tuple[str, ...] = ("val", "test"),
        max_ranking_windows: int | None = None,
        log=lambda message: None) -> dict[str, Any]:
    """Run every stage-1 metric and write the acceptance report."""
    resolved_device = torch.device(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset = load_transition_dataset(dataset_dir)
    dataset_hash = dataset_content_hash(dataset_dir)
    ensemble, checkpoint_metadata = load_world_model(
        checkpoint_path, expected_dataset_hash=dataset_hash
    )
    ensemble.eval()

    torch_dynamics = TorchAUVDynamics(
        vehicle_profile=config.environment.vehicle_profile,
        dt=float(config.environment.dt),
        device=resolved_device,
    )
    layout = None
    from pac.simulation.thrusters import build_thruster_layout

    layout = build_thruster_layout(
        config.environment.thruster_layout,
        max_force=float(config.actuator.max_force_n),
    )

    def wrench_converter(action):
        forces = layout.normalized_action_to_forces(np.asarray(action, dtype=float))
        return layout.forces_to_wrench(forces)

    report: dict[str, Any] = {
        "protocol_version": "predictive_authority_v4",
        "dataset_hash": dataset_hash,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_members": int(checkpoint_metadata["members"]),
        "member_seeds": list(checkpoint_metadata["member_seeds"]),
    }

    built = build_transition_windows(dataset, int(config.world_model.history_len))
    one_step_summary: list[dict[str, Any]] = []
    uncertainty_summary: dict[str, Any] = {}
    for split in ("val", "test"):
        indices = split_indices(dataset.metadata, split)
        if indices.size == 0:
            raise ValueError(f"transition dataset has no rows for split '{split}'")
        rows = windows_to_torch(built, indices)
        group_info = dataset.metadata.iloc[indices].reset_index(drop=True)
        frame = one_step_evaluation(ensemble, rows, group_info, device=resolved_device)
        frame.to_csv(output_path / f"one_step_{split}.csv", index=False)
        one_step_summary.extend(_summarize_one_step(frame, split))
        uncertainty_summary[split] = uncertainty_evaluation(frame)
        uncertainty_summary[split]["regimes"] = {
            "all": regime_uncertainty(frame),
            "scenario3": regime_uncertainty(frame, scenario_filter=3),
        }
    pd.DataFrame(one_step_summary).to_csv(output_path / "one_step_summary.csv", index=False)

    multistep_frames: list[pd.DataFrame] = []
    multistep_summaries: list[pd.DataFrame] = []
    for split in ("val", "test"):
        error_frame, summary_frame = multistep_evaluation(
            config, ensemble, dataset, torch_dynamics, wrench_converter,
            split=split, device=resolved_device,
        )
        multistep_frames.append(error_frame)
        multistep_summaries.append(summary_frame)
    pd.concat(multistep_frames, ignore_index=True).to_csv(
        output_path / "multistep_errors.csv", index=False
    )
    multistep_summary = pd.concat(multistep_summaries, ignore_index=True)
    multistep_summary.to_csv(output_path / "multistep_summary.csv", index=False)

    ranking_frames: list[pd.DataFrame] = []
    ranking_summaries: list[pd.DataFrame] = []
    for split in ranking_splits:
        frame, summary = ranking_evaluation(
            config, ensemble, dataset,
            split=split, device=resolved_device,
            max_windows=max_ranking_windows, log=log,
        )
        ranking_frames.append(frame)
        ranking_summaries.append(summary)
    pd.concat(ranking_frames, ignore_index=True).to_csv(
        output_path / "ranking_windows.csv", index=False
    )
    ranking_summary = pd.concat(ranking_summaries, ignore_index=True)
    ranking_summary.to_csv(output_path / "ranking_summary.csv", index=False)

    report["one_step"] = one_step_summary
    report["uncertainty"] = uncertainty_summary
    report["multistep"] = multistep_summary.to_dict(orient="records")
    report["ranking_summary"] = ranking_summary.to_dict(orient="records")
    report["gates"] = _evaluate_gates(config, report)

    (output_path / "gate_report.json").write_text(
        json.dumps(nan_to_none(report), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return report


def nan_to_none(value: Any) -> Any:
    """Replace non-finite floats with null so the report is strict JSON."""
    if isinstance(value, dict):
        return {key: nan_to_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [nan_to_none(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _one_step_all_row(report: dict[str, Any], split: str) -> dict[str, Any]:
    for row in report["one_step"]:
        if row["split"] == split and row["group"] == "all":
            return row
    raise KeyError(f"missing one-step summary row for split {split}")


def _multistep_row(report: dict[str, Any], split: str, model: str) -> dict[str, Any]:
    for row in report["multistep"]:
        if row["split"] == split and row["model"] == model:
            return row
    raise KeyError(f"missing multistep summary row for split={split} model={model}")


def _ranking_row(report: dict[str, Any], split: str, scorer: str) -> dict[str, Any]:
    for row in report["ranking_summary"]:
        if row["split"] == split and row["scorer"] == scorer:
            return row
    raise KeyError(f"missing ranking summary row for split={split} scorer={scorer}")


def _evaluate_gates(config, report: dict[str, Any]) -> dict[str, Any]:
    """Compute the five stage-1 acceptance gates from the v4 plan."""
    gates: dict[str, Any] = {}

    improvements: dict[str, float] = {}
    passes: list[bool] = []
    for split in ("val", "test"):
        row = _one_step_all_row(report, split)
        improvement = 1.0 - (
            row["ensemble_position_rmse"] / max(row["physics_position_rmse"], 1.0e-12)
        )
        improvements[split] = float(improvement)
        passes.append(bool(improvement >= _ONE_STEP_IMPROVEMENT_THRESHOLD))
    gates["one_step_beats_physics"] = {
        "description": (
            "ensemble one-step position RMSE improves on physics-only by >= "
            f"{_ONE_STEP_IMPROVEMENT_THRESHOLD:.0%} on val and test"
        ),
        "relative_improvement": improvements,
        "pass": all(passes),
    }

    val_ensemble = _multistep_row(report, "val", "ensemble")
    test_ensemble = _multistep_row(report, "test", "ensemble")
    test_persistence = _multistep_row(report, "test", "physics_persistence")
    val_rmse_20 = float(val_ensemble["position_rmse_20step"])
    test_rmse_20 = float(test_ensemble["position_rmse_20step"])
    persist_rmse_20 = float(test_persistence["position_rmse_20step"])
    gates["multistep_stable_on_unseen"] = {
        "description": (
            "ensemble 20-step rollout error stays within "
            f"{_MULTISTEP_STABILITY_FACTOR:.0f}x of validation on the unseen "
            "test split and beats physics+persistence"
        ),
        "val_rmse_20step": val_rmse_20,
        "test_rmse_20step": test_rmse_20,
        "test_physics_persistence_rmse_20step": persist_rmse_20,
        "pass": bool(
            test_rmse_20 <= _MULTISTEP_STABILITY_FACTOR * max(val_rmse_20, 1.0e-12)
            and test_rmse_20 <= persist_rmse_20
        ),
    }

    test_uncertainty = report["uncertainty"]["test"]
    correlation = float(test_uncertainty["uncertainty_error_spearman"])
    gates["uncertainty_correlates_with_error"] = {
        "description": (
            "ensemble std correlates with actual one-step error on test "
            f"(Spearman >= {_UNCERTAINTY_CORRELATION_THRESHOLD})"
        ),
        "spearman": correlation,
        "coverage_68": float(test_uncertainty["coverage_68"]),
        "coverage_95": float(test_uncertainty["coverage_95"]),
        "pass": bool(correlation >= _UNCERTAINTY_CORRELATION_THRESHOLD),
    }

    regimes = test_uncertainty["regimes"]["all"]
    scenario3 = test_uncertainty["regimes"]["scenario3"]
    stochastic = float(regimes["stochastic_std_mean"])
    structured = float(regimes["structured_std_mean"])
    recovery = float(scenario3["stepchange_recovery_std_mean"])
    prestep = float(scenario3["stepchange_prestep_std_mean"])
    gates["uncertainty_identifies_regimes"] = {
        "description": (
            "uncertainty rises for stepchange recovery and stochastic currents "
            "versus their baseline windows on test"
        ),
        "stepchange_recovery_std_mean": recovery,
        "stepchange_prestep_std_mean": prestep,
        "stochastic_std_mean": stochastic,
        "structured_std_mean": structured,
        "pass": bool(recovery > prestep and stochastic > structured),
    }

    threshold_spearman = float(config.world_model.ranking_spearman_threshold)
    threshold_top1 = float(config.world_model.ranking_top1_threshold)
    wm_row = _ranking_row(report, "test", "wm")
    wm_spearman = float(wm_row["all_spearman_mean"])
    wm_top1 = float(wm_row["all_top1"])
    persistence_row = _ranking_row(report, "test", "physics_persistence")
    gates["ranking_consistent_with_true_costs"] = {
        "description": (
            "world-model candidate ranking agrees with the true rollout "
            "ranking (mean Spearman >= threshold and top-1 >= threshold on test)"
        ),
        "wm_spearman_mean": wm_spearman,
        "wm_top1": wm_top1,
        "wm_regret_mean": float(wm_row["all_regret_mean"]),
        "physics_persistence_spearman_mean": float(
            persistence_row["all_spearman_mean"]
        ),
        "physics_persistence_top1": float(persistence_row["all_top1"]),
        "physics_persistence_regret_mean": float(persistence_row["all_regret_mean"]),
        "spearman_threshold": threshold_spearman,
        "top1_threshold": threshold_top1,
        "pass": bool(wm_spearman >= threshold_spearman and wm_top1 >= threshold_top1),
    }
    return gates


__all__ = [
    "evaluate_world_model",
    "nan_to_none",
    "multistep_evaluation",
    "one_step_evaluation",
    "ranking_evaluation",
    "regime_uncertainty",
    "uncertainty_evaluation",
]
