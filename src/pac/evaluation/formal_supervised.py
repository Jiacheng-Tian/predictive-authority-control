"""Paired, simulation-only formal evaluation for PAC true-MPC."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable

import numpy as np
import pandas as pd

from pac.authority.dataset import episode_fingerprint, load_oracle_dataset
from pac.authority.evaluation import run_predictive_alpha_episode, scalar_metrics
from pac.authority.model import (
    _default_git_provenance,
    _semantic_config_hash,
    load_supervised_checkpoint,
)
from pac.evaluation.diagnostics import compute_axis_window_metrics
from pac.evaluation.episode_spec import EpisodeSpec, build_episode_spec
from pac.evaluation.episodes import run_fixed_controller_episode
from pac.experiment_config import SupervisedExperimentConfig, load_supervised_config


ROOT = Path(__file__).resolve().parents[3]
FORMAL_RUN_ROOT_NAME = "formal_supervised"
FORMAL_METHODS = (
    "real10kg_smc_steady",
    "real10kg_mpc_ltv_v3",
    "predictive_alpha",
)
PAC_METHOD = "predictive_alpha"
BASELINE_METHODS = FORMAL_METHODS[:2]
METRIC_COLUMNS = (
    "rmse_3d",
    "heading_rmse_deg",
    "applied_control_cost",
    "action_saturation_step_fraction",
    "solver_fallback_step_fraction",
    "solver_deadline_miss_step_fraction",
    "actuator_rate_limit_episode_mean",
    "final_error",
    "max_error",
    "success_1m",
)
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class ProfilePlan:
    """Resolved dimensions for one formal runner profile."""

    profile: str
    model_seeds: tuple[int, ...]
    environment_seeds: tuple[int, ...]
    scenarios: tuple[int, ...]
    steps: int

    @property
    def episode_count(self) -> int:
        return len(self.environment_seeds) * len(self.scenarios)

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "model_seeds": list(self.model_seeds),
            "environment_seeds": list(self.environment_seeds),
            "scenarios": list(self.scenarios),
            "steps": int(self.steps),
            "episode_count": int(self.episode_count),
            "rollout_count": int(
                self.episode_count
                * (len(BASELINE_METHODS) + len(self.model_seeds))
            ),
        }


@dataclass(frozen=True, slots=True)
class MethodDefinition:
    """Stable method registry entry used by the paired evaluator."""

    name: str
    kind: str


METHOD_REGISTRY = {
    name: MethodDefinition(
        name=name,
        kind="pac" if name == PAC_METHOD else "fixed",
    )
    for name in FORMAL_METHODS
}


def build_profile_plan(config: SupervisedExperimentConfig, profile: str) -> ProfilePlan:
    """Resolve the exact dry, short, and formal protocol grids."""
    profile = str(profile).strip().lower()
    if profile not in {"dry", "short", "formal"}:
        raise ValueError("profile must be one of dry, short, or formal")
    expected_models = tuple(range(31000, 31005))
    expected_environment = tuple(range(41000, 41020))
    if tuple(config.training.model_seeds) != expected_models:
        raise ValueError(
            "formal v3 training.model_seeds must be exactly "
            "[31000, 31001, 31002, 31003, 31004]"
        )
    if tuple(config.evaluation.episode_seeds) != expected_environment:
        raise ValueError(
            "formal supervised evaluation.episode_seeds must be exactly "
            "[41000, ..., 41019]"
        )
    if tuple(config.environment.scenarios) != (1, 2, 3):
        raise ValueError("formal v3 scenarios must be exactly [1, 2, 3]")
    if profile == "short":
        model_seeds = (31000,)
        environment_seeds = (41000,)
        scenarios = (1,)
        steps = 20
    else:
        model_seeds = expected_models
        environment_seeds = expected_environment
        scenarios = (1, 2, 3)
        steps = int(config.environment.steps)
    return ProfilePlan(
        profile=profile,
        model_seeds=model_seeds,
        environment_seeds=environment_seeds,
        scenarios=scenarios,
        steps=steps,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def _read_dataset_manifest(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid oracle dataset manifest: {manifest_path}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("dataset_hash"), str):
        raise ValueError("oracle dataset manifest must contain dataset_hash")
    value["_manifest_path"] = manifest_path
    return value


def _reject_results_path(path: Path) -> None:
    if any(part.lower() == "results" for part in path.resolve().parts):
        raise ValueError("formal v3 output must not be under a results path")


def _reject_path_overlap(first: Path, second: Path, *, label: str) -> None:
    first = first.resolve()
    second = second.resolve()
    if first == second:
        raise ValueError(f"formal output must not equal {label}")
    try:
        second.relative_to(first)
    except ValueError:
        pass
    else:
        raise ValueError(f"formal output must not be inside {label}")
    try:
        first.relative_to(second)
    except ValueError:
        pass
    else:
        raise ValueError(f"{label} must not be inside formal output")


def _validate_output_root(output_root: Path) -> None:
    _reject_results_path(output_root)
    if output_root.name != FORMAL_RUN_ROOT_NAME:
        raise ValueError(
            "formal v3 output root must end in runs/formal_supervised"
        )


def _resolve_run_id(output_root: Path, run_id: str | None) -> tuple[str, Path]:
    if run_id is None or not str(run_id).strip():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = str(os.getpid())
        run_id = f"{stamp}-{suffix}"
    run_id = str(run_id).strip()
    if not _RUN_ID_RE.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError("run_id must contain only safe filename characters")
    target = (output_root / run_id).resolve()
    if target.parent != output_root.resolve():
        raise ValueError("run_id must name a direct child of output root")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"formal v3 run directory already exists: {target}")
    return run_id, target


def _checkpoint_path(checkpoints_dir: Path, seed: int) -> Path:
    candidates = (
        checkpoints_dir / f"pac_train_seed_{seed}" / "checkpoint.pt",
        checkpoints_dir / f"{seed}.pt",
        checkpoints_dir / f"checkpoint_{seed}.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"missing PAC checkpoint for model seed {seed} under {checkpoints_dir}"
    )


def _load_checkpoints(
        checkpoints_dir: Path,
        plan: ProfilePlan,
        config: SupervisedExperimentConfig,
        dataset_hash: str) -> tuple[dict[int, Any], dict[int, dict[str, Any]]]:
    models: dict[int, Any] = {}
    records: dict[int, dict[str, Any]] = {}
    semantic_hash = _semantic_config_hash(config.authority_model)
    for seed in plan.model_seeds:
        path = _checkpoint_path(checkpoints_dir, seed)
        model, metadata = load_supervised_checkpoint(
            path,
            config=config.authority_model,
            expected_dataset_hash=dataset_hash,
            expected_model_seed=seed,
            expected_config_semantic_sha256=semantic_hash,
        )
        models[seed] = model
        records[seed] = {
            "path": str(path),
            "sha256": _sha256(path),
            "model_seed": int(seed),
            "dataset_hash": str(metadata["dataset_hash"]),
            "input_dim": int(metadata["input_dim"]),
            "history_len": int(metadata["history_len"]),
            "embed_dim": int(metadata["embed_dim"]),
            "heads": int(metadata["heads"]),
            "layers": int(metadata["layers"]),
            "dropout": float(metadata["dropout"]),
        }
    return models, records


def _episode_specs(config: SupervisedExperimentConfig, plan: ProfilePlan) -> dict[tuple[int, int], EpisodeSpec]:
    return {
        (int(scenario), int(seed)): build_episode_spec(
            int(scenario),
            int(seed),
            plan.steps,
            config.environment.dt,
        )
        for scenario in plan.scenarios
        for seed in plan.environment_seeds
    }


def _run_one_method(
        method: str,
        spec: EpisodeSpec,
        config: SupervisedExperimentConfig,
        model: Any | None = None,
        model_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    env = config.environment
    if method in BASELINE_METHODS:
        return run_fixed_controller_episode(
            scenario=spec.scenario_id,
            seed=spec.episode_seed,
            steps=spec.steps,
            mass_scale_xy=env.mass_scale_xy,
            damping_scale_xy=env.damping_scale_xy,
            base_controller=method,
            current_amplitude_scale=env.current_amplitude_scale,
            current_frequency_scale=env.current_frequency_scale,
            initial_position_std=env.eval_initial_position_std,
            initial_velocity_std=env.eval_initial_velocity_std,
            vertical_current=env.vertical_current,
            vehicle_profile=env.vehicle_profile,
            thruster_layout=env.thruster_layout,
            save_ts=True,
            episode_spec=spec,
            dt=env.dt,
            actuator_max_delta_per_step=config.actuator.max_delta_per_step,
            aligned_metrics=True,
        )
    if method != PAC_METHOD:
        raise ValueError(f"unknown formal v3 method: {method}")
    if model is None:
        raise ValueError("predictive_alpha requires a checkpoint model")
    metadata = model_metadata or {}
    return run_predictive_alpha_episode(
        model=model,
        scenario=spec.scenario_id,
        seed=spec.episode_seed,
        steps=spec.steps,
        mass_scale_xy=env.mass_scale_xy,
        damping_scale_xy=env.damping_scale_xy,
        current_amplitude_scale=env.current_amplitude_scale,
        current_frequency_scale=env.current_frequency_scale,
        vertical_current=env.vertical_current,
        primary_controller=config.controller.primary,
        authority_controller=config.controller.authority,
        feature_mode=config.authority_model.feature_mode,
        initial_position_std=env.eval_initial_position_std,
        initial_velocity_std=env.eval_initial_velocity_std,
        alpha_gain=config.authority_model.alpha_gain,
        alpha_threshold=0.0,
        vehicle_profile=env.vehicle_profile,
        thruster_layout=env.thruster_layout,
        alpha_smoothing=config.authority_model.alpha_smoothing,
        alpha_rate_limit=config.authority_model.alpha_rate_limit,
        alpha_deadband=config.authority_model.alpha_deadband,
        policy_architecture=config.authority_model.architecture,
        history_len=int(metadata.get("history_len", config.authority_model.history_len)),
        save_ts=True,
        episode_spec=spec,
        dt=env.dt,
        actuator_max_delta_per_step=config.actuator.max_delta_per_step,
        aligned_metrics=True,
    )


def _normalize_episode_metrics(
        metrics: dict[str, Any],
        *,
        method: str,
        model_seed: int | None,
        spec: EpisodeSpec,
        config: SupervisedExperimentConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    timeseries = metrics.get("ts")
    if not isinstance(timeseries, pd.DataFrame):
        raise ValueError(f"{method} did not return a timeseries")
    timeseries = timeseries.copy()
    action_columns = sorted(
        (column for column in timeseries.columns if re.fullmatch(r"action_[0-9]+", str(column))),
        key=lambda column: int(str(column).split("_", 1)[1]),
    )
    applied_columns = sorted(
        (column for column in timeseries.columns if re.fullmatch(r"applied_action_[0-9]+", str(column))),
        key=lambda column: int(str(column).split("_", 2)[2]),
    )
    if action_columns and applied_columns:
        if len(action_columns) != len(applied_columns):
            raise ValueError("timeseries action and applied_action dimensions differ")
        for action_column, applied_column in zip(action_columns, applied_columns):
            if not np.allclose(
                    timeseries[action_column].to_numpy(dtype=float),
                    timeseries[applied_column].to_numpy(dtype=float),
                    rtol=0.0,
                    atol=1.0e-12):
                raise ValueError("formal v3 action_i must equal applied_action_i")
    scalar = scalar_metrics(metrics)
    aliases = {
        "rmse_3d": scalar.get("rmse_3d", scalar.get("rmse")),
        "heading_rmse_deg": scalar.get("heading_rmse_deg"),
        "applied_control_cost": scalar.get("energy"),
        "action_saturation_step_fraction": scalar.get("action_saturation_step_fraction"),
        "solver_fallback_step_fraction": scalar.get("solver_fallback_step_fraction", 0.0),
        "solver_deadline_miss_step_fraction": scalar.get("solver_deadline_miss_step_fraction", 0.0),
        "actuator_rate_limit_episode_mean": scalar.get(
            "actuator_rate_limit_episode_mean",
            scalar.get("actuator_rate_limited_fraction_mean", 0.0),
        ),
        "final_error": scalar.get("final_error"),
        "max_error": scalar.get("max_error"),
        "success_1m": scalar.get("success_1m", scalar.get("success_1.0m")),
    }
    missing = [name for name, value in aliases.items() if value is None]
    if missing:
        raise ValueError(f"episode metrics missing required fields: {', '.join(missing)}")
    row: dict[str, Any] = {
        "method": method,
        "model_seed": model_seed,
        "scenario_id": int(spec.scenario_id),
        "environment_seed": int(spec.episode_seed),
        "episode_uid": str(spec.episode_uid),
        "episode_fingerprint": episode_fingerprint(spec),
        "steps": int(spec.steps),
        "dt": float(spec.dt),
    }
    row.update({key: float(value) for key, value in aliases.items()})
    row.update({
        key: value
        for key, value in scalar.items()
        if key not in row and key != "ts" and np.isscalar(value)
    })
    row["scenario"] = {1: "constant", 2: "sinusoidal", 3: "step_change"}[spec.scenario_id]
    timeseries["method"] = method
    timeseries["model_seed"] = "" if model_seed is None else int(model_seed)
    timeseries["scenario_id"] = int(spec.scenario_id)
    timeseries["scenario"] = row["scenario"]
    timeseries["environment_seed"] = int(spec.episode_seed)
    timeseries["episode_uid"] = str(spec.episode_uid)
    timeseries["episode_fingerprint"] = row["episode_fingerprint"]
    if "time" not in timeseries.columns and "step" in timeseries.columns:
        timeseries["time"] = timeseries["step"].to_numpy(dtype=float) * float(spec.dt)
    return row, timeseries


def _window_metrics(timeseries: pd.DataFrame, row: dict[str, Any], dt: float) -> pd.DataFrame:
    windows = compute_axis_window_metrics(timeseries, dt=float(dt))
    if windows.empty:
        return windows
    for key in (
        "method",
        "model_seed",
        "scenario",
        "scenario_id",
        "environment_seed",
        "episode_uid",
        "episode_fingerprint",
    ):
        windows[key] = row.get(key)
    return windows


def _aggregate(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_columns = [column for column in METRIC_COLUMNS if column in raw.columns]
    overall = raw.groupby(["method", "model_seed"], dropna=False, as_index=False)[
        aggregate_columns
    ].mean()
    by_scenario = raw.groupby(
        ["method", "model_seed", "scenario_id", "scenario"],
        dropna=False,
        as_index=False,
    )[aggregate_columns].mean()
    return overall, by_scenario


def _percent_difference(difference: float, reference: float) -> float:
    denominator = abs(float(reference))
    if denominator <= 1.0e-15:
        return float("nan")
    return float(100.0 * difference / denominator)


def _paired_effects(
        raw: pd.DataFrame,
        model_seeds: tuple[int, ...],
        *,
        scenario_count: int,
        environment_count: int) -> pd.DataFrame:
    """Compute seed-level paired effects and CI over five model-seed effects."""
    key_columns = ["scenario_id", "environment_seed", "episode_uid"]
    rows: list[dict[str, Any]] = []
    for metric in METRIC_COLUMNS:
        for baseline in BASELINE_METHODS:
            seed_effects: list[float] = []
            baseline_means: list[float] = []
            pac_means: list[float] = []
            comparison = f"{PAC_METHOD}_vs_{baseline}"
            for seed in model_seeds:
                base = raw.loc[raw["method"].eq(baseline), key_columns + [metric]]
                pac = raw.loc[
                    raw["method"].eq(PAC_METHOD) & raw["model_seed"].eq(seed),
                    key_columns + [metric],
                ]
                merged = pac.merge(
                    base,
                    on=key_columns,
                    how="outer",
                    suffixes=("_pac", "_baseline"),
                    validate="one_to_one",
                )
                expected_pairs = int(scenario_count) * int(environment_count)
                if len(merged) != expected_pairs or merged[[f"{metric}_pac", f"{metric}_baseline"]].isna().any().any():
                    raise ValueError(
                        f"paired grid is incomplete for {comparison}, metric {metric}, seed {seed}"
                    )
                differences = (
                    merged[f"{metric}_pac"].to_numpy(dtype=float)
                    - merged[f"{metric}_baseline"].to_numpy(dtype=float)
                )
                mean_difference = float(np.mean(differences))
                baseline_mean = float(np.mean(merged[f"{metric}_baseline"]))
                pac_mean = float(np.mean(merged[f"{metric}_pac"]))
                seed_effects.append(mean_difference)
                baseline_means.append(baseline_mean)
                pac_means.append(pac_mean)
                rows.append({
                    "effect_level": "model_seed",
                    "metric": metric,
                    "comparison": comparison,
                    "model_seed": int(seed),
                    "mean_difference": mean_difference,
                    "sample_sd": float(np.std(differences, ddof=1)) if len(differences) > 1 else float("nan"),
                    "ci95_low": float("nan"),
                    "ci95_high": float("nan"),
                    "percent_difference": _percent_difference(mean_difference, baseline_mean),
                    "ci_df": float("nan"),
                    "ci_basis": "model_seed_aggregate",
                    "n_model_seeds": len(model_seeds),
                    "n_env_episodes": int(environment_count),
                    "n_scenarios": int(scenario_count),
                    "n_rollouts": expected_pairs,
                })
            effect_array = np.asarray(seed_effects, dtype=float)
            mean_difference = float(np.mean(effect_array))
            sample_sd = float(np.std(effect_array, ddof=1)) if len(effect_array) > 1 else float("nan")
            critical = 2.7764451051977987
            half_width = (
                critical * sample_sd / math.sqrt(len(effect_array))
                if len(effect_array) > 1 else float("nan")
            )
            reference = float(np.mean(baseline_means))
            rows.append({
                "effect_level": "model_seed_aggregate",
                "metric": metric,
                "comparison": comparison,
                "model_seed": "all",
                "mean_difference": mean_difference,
                "sample_sd": sample_sd,
                "ci95_low": mean_difference - half_width,
                "ci95_high": mean_difference + half_width,
                "percent_difference": _percent_difference(mean_difference, reference),
                "ci_df": len(effect_array) - 1 if len(effect_array) > 1 else float("nan"),
                "ci_basis": "five_model_seed_effects" if len(effect_array) > 1 else "insufficient_model_seeds",
                "n_model_seeds": len(model_seeds),
                "n_env_episodes": int(environment_count),
                "n_scenarios": int(scenario_count),
                "n_rollouts": expected_pairs * len(model_seeds),
            })
    return pd.DataFrame(rows)


def _artifact_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS.txt"}:
            result[path.relative_to(root).as_posix()] = _sha256(path)
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, default=_json_default),
        encoding="utf-8",
    )


def _write_seed_map(root: Path, specs: dict[tuple[int, int], EpisodeSpec]) -> list[dict[str, Any]]:
    rows = []
    for (scenario, seed), spec in sorted(specs.items()):
        rows.append({
            "scenario_id": int(scenario),
            "environment_seed": int(seed),
            "episode_uid": str(spec.episode_uid),
            "episode_fingerprint": episode_fingerprint(spec),
            "steps": int(spec.steps),
            "dt": float(spec.dt),
        })
    return rows


def run_formal_supervised(
        *,
        config_path: str | Path,
        dataset_dir: str | Path,
        checkpoints_dir: str | Path,
        output_root: str | Path,
        profile: str,
        run_id: str | None = None,
        allow_dirty: bool | None = None,
        progress_every: int = 0) -> Path | dict[str, Any]:
    """Run the paired formal grid or return a no-write dry-run plan."""
    config_path = Path(config_path).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    checkpoints_dir = Path(checkpoints_dir).resolve()
    output_root = Path(output_root).resolve()
    config = load_supervised_config(config_path)
    plan = build_profile_plan(config, profile)
    _validate_output_root(output_root)
    if plan.profile == "dry":
        return {
            **plan.as_dict(),
            "config": str(config_path),
            "dataset_dir": str(dataset_dir),
            "checkpoints_dir": str(checkpoints_dir),
            "output_root": str(output_root),
            "methods": list(FORMAL_METHODS),
            "writes": False,
        }

    _validate_output_root(output_root)
    _reject_path_overlap(output_root, dataset_dir, label="dataset directory")
    _reject_path_overlap(output_root, checkpoints_dir, label="checkpoint directory")
    run_id, target = _resolve_run_id(output_root, run_id)
    provenance = _default_git_provenance()
    dirty_allowed = allow_dirty if allow_dirty is not None else os.environ.get(
        "PAC_ALLOW_DIRTY_FORMAL"
    ) == "1"
    if plan.profile == "formal" and provenance["git_dirty"] and not dirty_allowed:
        raise ValueError(
            "formal supervised evaluation requires a clean git worktree; "
            "set PAC_ALLOW_DIRTY_FORMAL=1 only for debugging"
        )
    dataset_manifest = _read_dataset_manifest(dataset_dir)
    dataset = load_oracle_dataset(dataset_dir)
    del dataset
    models, checkpoint_records = _load_checkpoints(
        checkpoints_dir,
        plan,
        config,
        str(dataset_manifest["dataset_hash"]),
    )
    specs = _episode_specs(config, plan)
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=str(output_root)))
    try:
        (temporary / "timeseries").mkdir(parents=True, exist_ok=True)
        raw_rows: list[dict[str, Any]] = []
        window_frames: list[pd.DataFrame] = []
        completed = 0
        total = plan.episode_count * (len(BASELINE_METHODS) + len(plan.model_seeds))
        for (scenario, seed), spec in sorted(specs.items()):
            for method in BASELINE_METHODS:
                metrics = _run_one_method(method, spec, config)
                row, timeseries = _normalize_episode_metrics(
                    metrics,
                    method=method,
                    model_seed=None,
                    spec=spec,
                    config=config,
                )
                raw_rows.append(row)
                window_frames.append(_window_metrics(timeseries, row, config.environment.dt))
                timeseries.to_csv(
                    temporary / "timeseries" / (
                        f"{method}__model_none__{spec.episode_uid}.csv"
                    ),
                    index=False,
                )
                completed += 1
                if progress_every and (completed == 1 or completed % progress_every == 0):
                    print(f"[formal-v3] {completed}/{total}", flush=True)
            for model_seed in plan.model_seeds:
                metrics = _run_one_method(
                    PAC_METHOD,
                    spec,
                    config,
                    model=models[model_seed],
                    model_metadata=checkpoint_records[model_seed],
                )
                row, timeseries = _normalize_episode_metrics(
                    metrics,
                    method=PAC_METHOD,
                    model_seed=model_seed,
                    spec=spec,
                    config=config,
                )
                raw_rows.append(row)
                window_frames.append(_window_metrics(timeseries, row, config.environment.dt))
                timeseries.to_csv(
                    temporary / "timeseries" / (
                        f"{PAC_METHOD}__model_{model_seed}__{spec.episode_uid}.csv"
                    ),
                    index=False,
                )
                completed += 1
                if progress_every and (completed == 1 or completed % progress_every == 0):
                    print(f"[formal-v3] {completed}/{total}", flush=True)
        raw = pd.DataFrame(raw_rows)
        window_metrics = pd.concat(window_frames, ignore_index=True)
        overall, by_scenario = _aggregate(raw)
        paired = _paired_effects(
            raw,
            plan.model_seeds,
            scenario_count=len(plan.scenarios),
            environment_count=len(plan.environment_seeds),
        )
        raw.to_csv(temporary / "raw_metrics.csv", index=False)
        window_metrics.to_csv(temporary / "window_metrics.csv", index=False)
        overall.to_csv(temporary / "overall_summary.csv", index=False)
        by_scenario.to_csv(temporary / "by_scenario.csv", index=False)
        paired.to_csv(temporary / "paired_effects.csv", index=False)
        summary = {
            "protocol_version": config.protocol.version,
            "profile": plan.profile,
            "run_id": run_id,
            "methods": list(FORMAL_METHODS),
            "metrics": list(METRIC_COLUMNS),
            "plan": plan.as_dict(),
            "raw_rows": int(len(raw)),
            "window_rows": int(len(window_metrics)),
            "timeseries_files": int(len(list((temporary / "timeseries").glob("*.csv")))),
            "paired_effect_rows": int(len(paired)),
            "ci_basis": "five model-seed effects with t(4), not pooled rollout SD",
            "status": "complete",
        }
        _write_json(temporary / "formal_summary.json", summary)
        artifact_hashes = _artifact_hashes(temporary)
        (temporary / "SHA256SUMS.txt").write_text(
            "".join(f"{digest}  {path}\n" for path, digest in artifact_hashes.items()),
            encoding="utf-8",
        )
        manifest = {
            "protocol_version": config.protocol.version,
            "profile": plan.profile,
            "run_id": run_id,
            "output_dir": str(target),
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "config_semantic_sha256": _semantic_config_hash(config),
            "dataset_dir": str(dataset_dir),
            "dataset_hash": str(dataset_manifest["dataset_hash"]),
            "dataset_manifest_sha256": _sha256(dataset_dir / "manifest.json"),
            "dataset_file_sha256": dataset_manifest.get("file_sha256", {}),
            "checkpoints_dir": str(checkpoints_dir),
            "checkpoint_hashes": {
                str(seed): record["sha256"]
                for seed, record in sorted(checkpoint_records.items())
            },
            "checkpoints": {
                str(seed): record for seed, record in sorted(checkpoint_records.items())
            },
            "seed_map": _write_seed_map(temporary, specs),
            "model_seeds": list(plan.model_seeds),
            "environment_seeds": list(plan.environment_seeds),
            "scenarios": list(plan.scenarios),
            "methods": list(FORMAL_METHODS),
            "method_registry": {
                name: {"kind": definition.kind}
                for name, definition in METHOD_REGISTRY.items()
            },
            "model_dimensions": {
                str(seed): {
                    key: record[key]
                    for key in ("input_dim", "history_len", "embed_dim", "heads", "layers", "dropout")
                }
                for seed, record in sorted(checkpoint_records.items())
            },
            "environment_dimensions": {
                "dt": config.environment.dt,
                "steps": plan.steps,
                "mass_scale_xy": config.environment.mass_scale_xy,
                "damping_scale_xy": config.environment.damping_scale_xy,
                "current_amplitude_scale": config.environment.current_amplitude_scale,
                "current_frequency_scale": config.environment.current_frequency_scale,
                "vertical_current": config.environment.vertical_current,
                "vehicle_profile": config.environment.vehicle_profile,
                "thruster_layout": config.environment.thruster_layout,
                "actuator_max_delta_per_step": config.actuator.max_delta_per_step,
            },
            "git_commit": provenance.get("git_commit"),
            "git_dirty": bool(provenance.get("git_dirty")),
            "git_diff_sha256": provenance.get("git_diff_sha256"),
            "artifact_sha256": artifact_hashes,
        }
        _write_json(temporary / "manifest.json", manifest)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"formal v3 run directory appeared during evaluation: {target}")
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    return target


__all__ = [
    "BASELINE_METHODS",
    "FORMAL_METHODS",
    "METHOD_REGISTRY",
    "METRIC_COLUMNS",
    "PAC_METHOD",
    "ProfilePlan",
    "build_profile_plan",
    "run_formal_supervised",
]
