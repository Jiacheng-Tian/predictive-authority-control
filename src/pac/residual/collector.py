"""Closed-loop transition collection for the v4 world model.

One collector episode runs the frozen supervised controller pair (SMC primary, MPC
authority) under a fixed behavior policy (constant alpha or the v3
Transformer) and records every transition needed to train and validate the
physics + residual world model:

* full state ``[eta, nu]`` before and after the step,
* requested (blended) and actuator-applied actions,
* estimated (causal) and true currents,
* the MPC plan active at that step plus solver telemetry flags,
* the physics-only predicted next state under the applied (post-actuator)
  action and the causal estimated current — the deployable baseline whose
  mismatch with the realized next state defines the residual target.

Collection never writes inside supervised output trees and refuses non-empty output
directories.  ``_collect_single_plan`` is a pure per-episode function so a
process pool can parallelize the formal profile.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from pac.authority.dataset import episode_fingerprint
from pac.authority.evaluation import filter_authority_alpha
from pac.authority.features import build_alpha_feature
from pac.authority.model import build_temporal_feature_window, load_alpha_model_checkpoint
from pac.evaluation.episodes import build_controller, compute_controller_action
from pac.controllers.presets import build_controller_pair
from pac.simulation.observations import CausalCurrentEstimator
from pac.residual.disturbances import (
    STRUCTURED_FAMILY,
    PairedEpisodeSpec,
    build_paired_episode_spec,
    draw_disturbance_params,
    make_disturbance_simulator,
)

FEATURE_DIM = 32
STATE_DIM = 12
PLAN_HORIZON = 20

COLLECTION_SPLITS = ("train", "val", "test")

_COLUMNS = (
    "episode_uid",
    "family",
    "scenario_id",
    "environment_seed",
    "split",
    "behavior",
    "behavior_alpha",
    "transformer_model_seed",
    "step",
    "sample_time",
    "plan_accepted",
    "params_json",
    "episode_fingerprint",
)

_BACKBONE_CACHE: dict[int, tuple] = {}

_FEATURE_DIM = 32


def assemble_transition_feature(
        state, requested, applied, est_current, context, scenario_id: int):
    """Assemble the 32-dim per-step transition feature (dataset-compatible)."""
    import numpy as np
    state_array = np.asarray(state, dtype=np.float64).reshape(12)
    requested_array = np.asarray(requested, dtype=np.float64).reshape(6)
    applied_array = np.asarray(applied, dtype=np.float64).reshape(6)
    current_array = np.asarray(est_current, dtype=np.float64).reshape(3)
    context_array = np.asarray(context, dtype=np.float64).reshape(4)
    values = np.concatenate([
        state_array, requested_array, applied_array,
        current_array, context_array,
        np.array([int(scenario_id) / 3.0]),
    ])
    if values.shape != (_FEATURE_DIM,):
        raise RuntimeError(
            f"transition feature dim {values.shape[0]}, expected {_FEATURE_DIM}")
    return values.astype(np.float32)


# Validation episodes use the seen (train) disturbance-parameter ranges; the
# test split is the only one drawn from the held-out unseen ranges.
_PROFILE_BY_SPLIT = {"train": "train", "val": "train", "test": "test"}


@dataclass(frozen=True)
class TransitionArrays:
    """Row-aligned transition tensors for one collection run."""

    features: np.ndarray          # (N, 32) world-model per-step inputs
    state: np.ndarray             # (N, 12) eta||nu at step start
    next_state: np.ndarray        # (N, 12) realized next state
    physics_next_state: np.ndarray  # (N, 12) physics rollout under realized current
    requested: np.ndarray         # (N, 6)
    applied: np.ndarray           # (N, 6)
    est_current: np.ndarray       # (N, 3)
    true_current: np.ndarray      # (N, 3)
    next_true_current: np.ndarray  # (N, 3)
    context: np.ndarray           # (N, 4) nominal mass/damping/amp/freq scales
    solver: np.ndarray            # (N, 3) wall_time_s, fallback flag, deadline flag
    plan: np.ndarray              # (N, 20, 6) float16
    alpha: np.ndarray             # (N,) blended behavior alpha
    metadata: pd.DataFrame

    def __post_init__(self) -> None:
        counts = {
            name: np.asarray(getattr(self, name)).shape[0]
            for name in (
                "features", "state", "next_state", "physics_next_state",
                "requested", "applied", "est_current", "true_current",
                "next_true_current", "context", "solver", "plan", "alpha",
            )
        }
        if len(set(counts.values())) != 1:
            raise ValueError(f"transition arrays have mismatched lengths: {counts}")
        if len(self.metadata) != counts["features"]:
            raise ValueError("metadata length must match transition arrays")


def build_behavior_policy_list(collection_config) -> list[tuple[str, float]]:
    """Return the behavior list [(behavior_name, alpha)] for collection."""
    behaviors = [
        ("constant_alpha", float(alpha))
        for alpha in collection_config.behavior_constant_alphas
    ]
    if collection_config.include_transformer_policy:
        behaviors.append(("transformer", float("nan")))
    return behaviors


def resolve_collection_plans(
        config,
        profile: str) -> list[dict]:
    """Enumerate every (split, seed, family, scenario, behavior) collection task."""
    profile = str(profile).strip().lower()
    if profile not in ("short", "formal"):
        raise ValueError("profile must be 'short' or 'formal'")
    if profile == "short":
        split_seeds = {
            "train": (int(config.seeds.wm_train[0]),),
            "val": (int(config.seeds.wm_val[0]),),
            "test": (int(config.seeds.wm_test[0]),),
        }
        families: tuple[str, ...] = (STRUCTURED_FAMILY, "ou_current")
        steps = 60
    else:
        split_seeds = {
            "train": tuple(int(seed) for seed in config.seeds.wm_train),
            "val": tuple(int(seed) for seed in config.seeds.wm_val),
            "test": tuple(int(seed) for seed in config.seeds.wm_test),
        }
        families = (STRUCTURED_FAMILY, *config.disturbances.families)
        steps = int(config.environment.steps)

    plans: list[dict] = []
    for split in COLLECTION_SPLITS:
        for seed in split_seeds[split]:
            for family in families:
                scenario_ids: tuple[int, ...] = (
                    (1, 2, 3) if family == STRUCTURED_FAMILY
                    else (int(config.disturbances.base_scenario[family]),)
                )
                for scenario_id in scenario_ids:
                    for behavior, alpha in build_behavior_policy_list(config.collection):
                        plans.append({
                            "split": split,
                            "seed": seed,
                            "family": family,
                            "scenario_id": scenario_id,
                            "behavior": behavior,
                            "behavior_alpha": alpha,
                            "steps": steps,
                        })
    return plans


def build_paired_spec_for_plan(config, plan: dict) -> PairedEpisodeSpec:
    """Build the deterministic episode spec for one collection task."""
    family = plan["family"]
    seed = int(plan["seed"])
    steps = int(plan["steps"])
    dt = float(config.environment.dt)
    if family == STRUCTURED_FAMILY:
        return build_paired_episode_spec(
            STRUCTURED_FAMILY,
            int(plan["scenario_id"]),
            seed,
            steps,
            dt,
        )
    ranges = {
        name: (bound.low, bound.high)
        for name, bound in config.disturbances.profile_ranges(
            family, _PROFILE_BY_SPLIT[plan["split"]]
        ).items()
    }
    params = draw_disturbance_params(family, seed, ranges)
    return build_paired_episode_spec(
        family,
        int(plan["scenario_id"]),
        seed,
        steps,
        dt,
        disturbance_params=params,
    )


def _build_simulator(config, spec: PairedEpisodeSpec):
    environment = make_disturbance_simulator(
        spec,
        mass_scale_xy=float(config.environment.mass_scale_xy),
        damping_scale_xy=float(config.environment.damping_scale_xy),
        current_amplitude_scale=float(config.environment.current_amplitude_scale),
        current_frequency_scale=float(config.environment.current_frequency_scale),
        vertical_current=float(config.environment.vertical_current),
        vehicle_profile=str(config.environment.vehicle_profile),
        thruster_layout=str(config.environment.thruster_layout),
        dt=float(config.environment.dt),
        actuator_command_min=float(config.actuator.command_min),
        actuator_command_max=float(config.actuator.command_max),
        actuator_max_delta_per_step=float(config.actuator.max_delta_per_step),
        max_force=float(config.actuator.max_force_n),
    )
    environment.reset(episode_spec=spec)
    return environment


def _load_backbone(config, model_seed: int):
    if model_seed not in _BACKBONE_CACHE:
        checkpoint_path = (
            Path(config.backbone.checkpoint_dir)
            / f"pac_train_seed_{model_seed}"
            / "checkpoint.pt"
        )
        model, metadata = load_alpha_model_checkpoint(
            checkpoint_path,
            expected_feature_mode=config.authority_model.feature_mode,
        )
        model.eval()
        _BACKBONE_CACHE[model_seed] = (model, metadata)
    return _BACKBONE_CACHE[model_seed]


def _physics_next_state(env_dynamics, action_to_wrench, state, requested, current):
    eta = np.asarray(state[:6], dtype=float)
    nu = np.asarray(state[6:], dtype=float)
    tau = np.asarray(action_to_wrench(np.asarray(requested, dtype=float)), dtype=float)
    next_eta, next_nu = env_dynamics.predict_step(eta, nu, tau, np.asarray(current, dtype=float))
    return np.concatenate([next_eta, next_nu])


def _collect_single_plan(config, plan: dict) -> dict:
    """Run one collection episode and return its transition chunks."""
    spec = build_paired_spec_for_plan(config, plan)
    environment = _build_simulator(config, spec)
    primary_name, primary = build_controller(config.controller.primary)
    primary.reset()
    primary.set_trajectory3d(True)
    authority_name, authority = build_controller_pair(config)
    authority.reset()
    authority.set_trajectory3d(True)
    estimator = CausalCurrentEstimator(
        spec.current_delay_steps,
        spec.current_estimation_noise,
    )
    estimator.reset(environment.privileged_state[:3])

    behavior = plan["behavior"]
    transformer_model = None
    transformer_metadata = None
    transformer_history: list[np.ndarray] = []
    model_seed = -1
    if behavior == "transformer":
        model_seed = int(config.collection.transformer_model_seed)
        transformer_model, transformer_metadata = _load_backbone(config, model_seed)

    context_base = np.array([
        float(config.environment.mass_scale_xy),
        float(config.environment.damping_scale_xy),
        float(config.environment.current_amplitude_scale),
        float(config.environment.current_frequency_scale),
    ], dtype=np.float64)

    steps = int(plan["steps"])
    episode_end = float(steps) * environment.dynamics.dt
    behavior_alpha = float(plan["behavior_alpha"])
    previous_alpha = 0.0
    episode_features: list[np.ndarray] = []
    episode_state: list[np.ndarray] = []
    episode_next_state: list[np.ndarray] = []
    episode_physics: list[np.ndarray] = []
    episode_requested: list[np.ndarray] = []
    episode_applied: list[np.ndarray] = []
    episode_est: list[np.ndarray] = []
    episode_true: list[np.ndarray] = []
    episode_next_true: list[np.ndarray] = []
    episode_solver: list[np.ndarray] = []
    episode_plan: list[np.ndarray] = []
    episode_alpha: list[float] = []
    episode_plan_accepted: list[bool] = []

    fingerprint = episode_fingerprint(spec)
    params_json = json.dumps(dict(spec.disturbance_params), sort_keys=True)
    action_to_wrench = (
        lambda act, env=environment: env._action_to_wrench(act)[0]
    )
    done = False
    while not done:
        dynamics = environment.dynamics
        step = int(environment.current_step)
        stamp = float(step * dynamics.dt)
        target = environment._get_target(stamp)
        true_current = np.asarray(
            environment._current_for_dynamics(environment._generate_current(stamp)),
            dtype=float,
        )
        estimated_current = estimator.estimate(true_current, step)
        state = np.concatenate([
            np.asarray(dynamics.eta, dtype=float),
            np.asarray(dynamics.nu, dtype=float),
        ])

        primary_action = compute_controller_action(
            primary, primary_name, target, dynamics.eta, dynamics.nu,
            stamp, dynamics.dt, estimated_current,
        )
        authority_action = compute_controller_action(
            authority, authority_name, target, dynamics.eta, dynamics.nu,
            stamp, dynamics.dt, estimated_current,
        )
        telemetry = getattr(authority, "last_telemetry", {})
        if not isinstance(telemetry, dict):
            telemetry = {}
        fallback_flag = float(str(telemetry.get("fallback_mode", "none")) != "none")
        deadline_flag = float(bool(telemetry.get("deadline_missed", False)))
        wall_time = float(telemetry.get("compute_wall_time_s", 0.0))
        plan_obj = authority.last_plan
        if plan_obj is not None:
            plan_array = np.asarray(plan_obj, dtype=float)
            plan_accepted = bool(telemetry.get("accepted", False)) and (
                plan_array.ndim == 2 and plan_array.shape[1] == 6
            )
        else:
            plan_array = np.zeros((PLAN_HORIZON, 6), dtype=float)
            plan_accepted = False
        if plan_array.shape[0] < PLAN_HORIZON:
            padded = np.zeros((PLAN_HORIZON, 6), dtype=float)
            padded[:plan_array.shape[0]] = plan_array
            plan_array = padded

        if behavior == "constant_alpha":
            alpha = float(np.clip(behavior_alpha, 0.0, 1.0))
            alpha = filter_authority_alpha(
                alpha,
                previous_alpha,
                smoothing=float(config.authority_model.alpha_smoothing),
                rate_limit=float(config.authority_model.alpha_rate_limit),
                deadband=float(config.authority_model.alpha_deadband),
            )
        else:
            feature = build_alpha_feature(
                target,
                dynamics.eta,
                dynamics.nu,
                primary_action,
                authority_action,
                estimated_current[:2],
                stamp,
                episode_end,
                config.authority_model.feature_mode,
            )
            transformer_history.append(feature)
            model_input = torch.from_numpy(
                build_temporal_feature_window(
                    transformer_history,
                    int(transformer_metadata["history_len"]),
                )
            ).unsqueeze(0)
            with torch.no_grad():
                raw_alpha = float(transformer_model(model_input).item())
            alpha = float(np.clip(
                raw_alpha * float(config.authority_model.alpha_gain), 0.0, 1.0
            ))
            alpha = filter_authority_alpha(
                alpha,
                previous_alpha,
                smoothing=float(config.authority_model.alpha_smoothing),
                rate_limit=float(config.authority_model.alpha_rate_limit),
                deadband=float(config.authority_model.alpha_deadband),
            )
        if bool(getattr(authority, "force_primary_authority", False)
                or getattr(authority, "force_primary", False)):
            alpha = 0.0

        requested = (1.0 - alpha) * np.asarray(primary_action, dtype=float) \
            + alpha * np.asarray(authority_action, dtype=float)
        requested = np.clip(requested, -1.0, 1.0)

        done, step_info = environment.step(requested)
        applied = np.asarray(step_info["applied_action"], dtype=float)
        next_state = np.concatenate([
            np.asarray(dynamics.eta, dtype=float),
            np.asarray(dynamics.nu, dtype=float),
        ])
        # The physics baseline must reproduce the realized transition's own
        # command path: the applied (post-actuator) action and the causal
        # estimated current.  Using the pre-actuator request would make the
        # "residual" absorb the actuator slew difference and poison rollouts.
        physics_prediction = _physics_next_state(
            dynamics, action_to_wrench, state, applied, estimated_current
        )

        episode_features.append(assemble_transition_feature(
            state, requested, applied, estimated_current, context_base,
            environment.scenario,
        ))
        episode_state.append(state.astype(np.float32))
        episode_next_state.append(next_state.astype(np.float32))
        episode_physics.append(physics_prediction.astype(np.float32))
        episode_requested.append(requested.astype(np.float32))
        episode_applied.append(applied.astype(np.float32))
        episode_est.append(np.asarray(estimated_current, dtype=float).astype(np.float32))
        episode_true.append(true_current.astype(np.float32))
        episode_solver.append(
            np.array([wall_time, fallback_flag, deadline_flag], dtype=np.float32)
        )
        episode_plan.append(plan_array.astype(np.float16))
        episode_alpha.append(float(alpha))
        episode_plan_accepted.append(plan_accepted)
        previous_alpha = float(alpha)

    realized = environment.realized_current_trajectory
    for row in range(steps):
        next_index = row + 1
        if next_index < len(realized):
            episode_next_true.append(realized[next_index].astype(np.float32))
        else:
            episode_next_true.append(realized[-1].astype(np.float32))

    metadata_rows = []
    for row in range(steps):
        metadata_rows.append({
            "episode_uid": spec.episode_uid,
            "family": spec.family,
            "scenario_id": int(spec.scenario_id),
            "environment_seed": int(spec.episode_seed),
            "split": plan["split"],
            "behavior": behavior,
            "behavior_alpha": float("nan") if behavior == "transformer" else behavior_alpha,
            "transformer_model_seed": model_seed,
            "step": row,
            "sample_time": float((row + 1) * environment.dynamics.dt),
            "plan_accepted": bool(episode_plan_accepted[row]),
            "params_json": params_json,
            "episode_fingerprint": fingerprint,
        })
    return {
        "episode_uid": spec.episode_uid,
        "features": np.stack(episode_features),
        "state": np.stack(episode_state),
        "next_state": np.stack(episode_next_state),
        "physics_next_state": np.stack(episode_physics),
        "requested": np.stack(episode_requested),
        "applied": np.stack(episode_applied),
        "est_current": np.stack(episode_est),
        "true_current": np.stack(episode_true),
        "next_true_current": np.stack(episode_next_true),
        "context": np.tile(context_base.astype(np.float32), (steps, 1)),
        "solver": np.stack(episode_solver),
        "plan": np.stack(episode_plan),
        "alpha": np.asarray(episode_alpha, dtype=np.float32),
        "metadata": metadata_rows,
    }


def collect_transition_dataset(
        config,
        profile: str,
        *,
        progress_every: int = 1,
        log=lambda message: None) -> TransitionArrays:
    """Sequentially run every collection task and return aligned arrays."""
    plans = resolve_collection_plans(config, profile)
    features, states, nexts, physics = [], [], [], []
    requesteds, applieds, ests, trues, next_trues = [], [], [], [], []
    contexts, solvers, plans_store, alphas = [], [], [], []
    metadata_rows: list[dict] = []
    total = len(plans)
    for index, plan in enumerate(plans, start=1):
        started = time.perf_counter()
        result = _collect_single_plan(config, plan)
        features.append(result["features"])
        states.append(result["state"])
        nexts.append(result["next_state"])
        physics.append(result["physics_next_state"])
        requesteds.append(result["requested"])
        applieds.append(result["applied"])
        ests.append(result["est_current"])
        trues.append(result["true_current"])
        next_trues.append(result["next_true_current"])
        contexts.append(result["context"])
        solvers.append(result["solver"])
        plans_store.append(result["plan"])
        alphas.append(result["alpha"])
        metadata_rows.extend(result["metadata"])
        if progress_every and (index == 1 or index % progress_every == 0 or index == total):
            log(
                f"[wm-collect] {index}/{total} episode={result['episode_uid']} "
                f"elapsed={time.perf_counter() - started:.1f}s"
            )
    if not features:
        raise ValueError("collection plan produced no episodes")
    return TransitionArrays(
        features=np.concatenate(features),
        state=np.concatenate(states),
        next_state=np.concatenate(nexts),
        physics_next_state=np.concatenate(physics),
        requested=np.concatenate(requesteds),
        applied=np.concatenate(applieds),
        est_current=np.concatenate(ests),
        true_current=np.concatenate(trues),
        next_true_current=np.concatenate(next_trues),
        context=np.concatenate(contexts),
        solver=np.concatenate(solvers),
        plan=np.concatenate(plans_store),
        alpha=np.concatenate(alphas),
        metadata=pd.DataFrame(metadata_rows, columns=_COLUMNS),
    )


def save_transition_dataset(
        dataset: TransitionArrays,
        out_dir: str | Path,
        provenance: dict | None = None) -> dict:
    """Atomically persist the transition dataset with full hashes."""
    target = Path(out_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"output directory is not empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent)))
    try:
        arrays = {
            name: np.ascontiguousarray(np.asarray(getattr(dataset, name)))
            for name in (
                "features", "state", "next_state", "physics_next_state",
                "requested", "applied", "est_current", "true_current",
                "next_true_current", "context", "solver", "plan", "alpha",
            )
        }
        np.savez(temp_dir / "transitions.npz", **arrays)
        dataset.metadata.to_csv(temp_dir / "metadata.csv", index=False)
        manifest = {
            "schema_version": 1,
            "protocol_version": (provenance or {}).get("protocol_version", ""),
            "sample_count": int(arrays["features"].shape[0]),
            "feature_dim": int(arrays["features"].shape[1]),
            "plan_horizon": int(arrays["plan"].shape[1]),
            "git_commit": _git_commit(),
            "dependency_versions": _dependency_versions(),
            "seed_partitions": (provenance or {}).get("seed_partitions", {}),
            "profile": (provenance or {}).get("profile", ""),
            "provenance": _jsonable(provenance or {}),
        }
        file_hashes = {}
        for name in ("transitions.npz", "metadata.csv"):
            digest = hashlib.sha256()
            with (temp_dir / name).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            file_hashes[name] = digest.hexdigest()
        manifest["file_sha256"] = file_hashes
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
        )
        if target.exists():
            target.rmdir()
        temp_dir.replace(target)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return manifest


def load_transition_dataset(in_dir: str | Path) -> TransitionArrays:
    """Load and hash-verify a persisted transition dataset."""
    root = Path(in_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["file_sha256"].items():
        digest = hashlib.sha256()
        with (root / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"transition dataset file hash mismatch: {name}")
    payload = np.load(root / "transitions.npz", allow_pickle=False)
    metadata = pd.read_csv(root / "metadata.csv", keep_default_na=False)
    metadata["params_json"] = metadata["params_json"].astype(str)
    metadata["episode_fingerprint"] = metadata["episode_fingerprint"].astype(str)
    metadata["family"] = metadata["family"].astype(str)
    metadata["behavior"] = metadata["behavior"].astype(str)
    metadata["episode_uid"] = metadata["episode_uid"].astype(str)
    metadata["plan_accepted"] = (
        metadata["plan_accepted"].astype(str).str.lower().isin({"true", "1"})
    )
    for column in ("scenario_id", "environment_seed", "step", "transformer_model_seed"):
        metadata[column] = metadata[column].astype(np.int64)
    metadata["behavior_alpha"] = pd.to_numeric(
        metadata["behavior_alpha"], errors="coerce"
    )
    metadata["sample_time"] = pd.to_numeric(metadata["sample_time"], errors="raise")
    return TransitionArrays(
        features=payload["features"],
        state=payload["state"],
        next_state=payload["next_state"],
        physics_next_state=payload["physics_next_state"],
        requested=payload["requested"],
        applied=payload["applied"],
        est_current=payload["est_current"],
        true_current=payload["true_current"],
        next_true_current=payload["next_true_current"],
        context=payload["context"],
        solver=payload["solver"],
        plan=payload["plan"],
        alpha=payload["alpha"],
        metadata=metadata,
    )


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[3]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dependency_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "pandas", "scipy", "osqp", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("provenance values must be finite")
        return value
    if isinstance(value, Path):
        return str(value)
    return str(value)


__all__ = [
    "FEATURE_DIM",
    "STATE_DIM",
    "PLAN_HORIZON",
    "TransitionArrays",
    "_collect_single_plan",
    "build_paired_spec_for_plan",
    "collect_transition_dataset",
    "load_transition_dataset",
    "resolve_collection_plans",
    "save_transition_dataset",
]
