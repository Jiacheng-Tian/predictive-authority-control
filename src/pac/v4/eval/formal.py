"""Paired formal evaluation for predictive authority v4 (stage two)."""

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

from pac.authority.dataset import episode_fingerprint
from pac.v4.collector import load_transition_dataset
from pac.v4.config import load_v4_config
from pac.v4.disturbances import (
    STRUCTURED_FAMILY,
    build_v4_episode_spec,
    draw_disturbance_params,
)
from pac.v4.eval.runner import (
    ConstantAlphaPolicy,
    FixedControllerPolicy,
    TransformerAlphaPolicy,
    build_backbone_policy,
    run_v4_policy_episode,
)

FORMAL_METHODS_FIXED = ("smc", "mpc", "constant_alpha", "sspo")
LEARNED_METHODS = ("v3_transformer", "wm_hybrid", "residual_rl", "wm_residual_rl")
METRIC_COLUMNS = (
    "rmse_3d",
    "heading_rmse_deg",
    "applied_control_cost",
    "action_saturation_step_fraction",
    "solver_fallback_step_fraction",
    "solver_deadline_miss_step_fraction",
    "actuator_rate_limit_episode_mean",
    "authority_alpha_mean",
    "wm_gate_open_fraction",
    "final_error",
    "max_error",
    "success_1m",
)
METRIC_ALIASES = {
    "rmse_3d": "rmse_3d",
    "heading_rmse_deg": "heading_rmse_deg",
    "applied_control_cost": "energy",
    "action_saturation_step_fraction": "action_saturation_step_fraction",
    "solver_fallback_step_fraction": "solver_fallback_step_fraction",
    "solver_deadline_miss_step_fraction": "solver_deadline_miss_step_fraction",
    "actuator_rate_limit_episode_mean": "actuator_rate_limit_episode_mean",
    "authority_alpha_mean": "authority_alpha_mean",
    "wm_gate_open_fraction": "wm_gate_open_fraction",
    "final_error": "final_error",
    "max_error": "max_error",
    "success_1m": "success_1.0m",
}
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_T_CRIT_4 = 2.7764451051977987


@dataclass(frozen=True)
class EpisodeTask:
    block: str        # seen | unseen
    family: str
    scenario_id: int
    seed: int
    profile: str      # disturbance range profile ("none" for structured)


def build_episode_tasks(config, profile: str) -> list[EpisodeTask]:
    """Resolve the paired seen/unseen episode grid."""
    profile = str(profile).strip().lower()
    family_count = int(config.evaluation.family_seed_count)
    tasks: list[EpisodeTask] = []
    if profile == "short":
        tasks.append(EpisodeTask("seen", STRUCTURED_FAMILY, 1,
                                 int(config.seeds.eval_seen[0]), "none"))
        tasks.append(EpisodeTask(
            "unseen", "ou_current", 2, int(config.seeds.eval_unseen[0]), "test"
        ))
        return tasks
    if profile not in ("formal", "dry"):
        raise ValueError("profile must be dry, short, or formal")
    for scenario in (1, 2, 3):
        for seed in config.seeds.eval_seen:
            tasks.append(EpisodeTask(
                "seen", STRUCTURED_FAMILY, scenario, int(seed), "none"
            ))
    for family in config.disturbances.families:
        for seed in config.seeds.eval_seen[:family_count]:
            tasks.append(EpisodeTask(
                "seen", family, int(config.disturbances.base_scenario[family]),
                int(seed), "train",
            ))
    for family in config.disturbances.families:
        for seed in config.seeds.eval_unseen[:family_count]:
            tasks.append(EpisodeTask(
                "unseen", family, int(config.disturbances.base_scenario[family]),
                int(seed), "test",
            ))
    for family in sorted(config.disturbances.eval_only):
        for seed in config.seeds.eval_unseen[:family_count]:
            tasks.append(EpisodeTask(
                "unseen", family, int(config.disturbances.eval_only[family].base_scenario),
                int(seed), "test",
            ))
    return tasks


def build_task_spec(config, task: EpisodeTask):
    """Materialize the deterministic episode spec for one grid task."""
    steps = int(config.environment.steps)
    dt = float(config.environment.dt)
    if task.family == STRUCTURED_FAMILY:
        return build_v4_episode_spec(
            STRUCTURED_FAMILY, task.scenario_id, task.seed, steps, dt
        )
    ranges = {
        name: (bound.low, bound.high)
        for name, bound in config.disturbances.profile_ranges(
            task.family, task.profile
        ).items()
    }
    params = draw_disturbance_params(task.family, task.seed, ranges)
    return build_v4_episode_spec(
        task.family,
        task.scenario_id,
        task.seed,
        steps,
        dt,
        disturbance_params=params,
    )


def method_grid(config, *, include_wm_rl: bool) -> list[dict]:
    """Enumerate (method, model_seed) combos for the formal run."""
    combos: list[dict] = [
        {"method": "smc", "model_seed": None},
        {"method": "mpc", "model_seed": None},
        {"method": "constant_alpha", "model_seed": None},
        {"method": "sspo", "model_seed": None},
    ]
    for method in ("v3_transformer", "wm_hybrid", "residual_rl"):
        for seed in config.seeds.model:
            combos.append({"method": method, "model_seed": int(seed)})
    if include_wm_rl:
        for seed in config.seeds.model:
            combos.append({"method": "wm_residual_rl", "model_seed": int(seed)})
    return combos


class MethodFactory:
    """Lazily built, cached policy instances shared across episodes."""

    def __init__(self, config, *, rl_dir: Path | None, wm_rl_dir: Path | None,
                 wm_checkpoint: Path | None,
                 dataset_dir: Path | None, sspo_schedule: dict | None,
                 include_wm_rl: bool, wm_sigma: dict | None = None):
        self.wm_sigma = wm_sigma
        self.config = config
        self.rl_dir = rl_dir
        self.wm_rl_dir = wm_rl_dir or rl_dir
        self.wm_checkpoint = wm_checkpoint
        self.dataset_dir = dataset_dir
        self.sspo_schedule = sspo_schedule
        self.include_wm_rl = include_wm_rl
        self._cache: dict[tuple, Any] = {}
        self._wm_computer = None

    def wm_computer(self):
        if self._wm_computer is None and self.wm_checkpoint is not None:
            from pac.v4.wmauthority import build_wm_authority_computer

            self._wm_computer, _version = build_wm_authority_computer(
                self.config,
                self.wm_checkpoint,
                self.dataset_dir,
                sigma=self.wm_sigma,
                device="cpu",
            )
        return self._wm_computer

    def policy_for(self, method: str, model_seed: int | None):
        key = (method, model_seed)
        if key in self._cache:
            return self._cache[key]
        if method == "smc":
            policy = FixedControllerPolicy("primary")
        elif method == "mpc":
            policy = FixedControllerPolicy("authority")
        elif method == "constant_alpha":
            policy = ConstantAlphaPolicy(0.5)
        elif method == "sspo":
            if not self.sspo_schedule:
                raise ValueError("sspo method requires a bias schedule file")
            policy = build_backbone_policy(
                self.config, self.config.seeds.model[0],
                alpha_bias_schedule=dict(self.sspo_schedule),
            )
        elif method == "v3_transformer":
            policy = build_backbone_policy(self.config, int(model_seed))
        elif method == "wm_hybrid":
            from pac.v4.wmauthority import WMHybridPolicy

            backbone = build_backbone_policy(self.config, int(model_seed))
            policy = WMHybridPolicy(
                backbone.model, alpha_gain=float(self.config.authority_model.alpha_gain)
            )
        elif method in ("residual_rl", "wm_residual_rl"):
            from pac.authority.model import load_alpha_model_checkpoint
            from pac.v4.rl.policy import load_rl_checkpoint

            if method == "wm_residual_rl" and not self.include_wm_rl:
                raise ValueError("wm_residual_rl requested but not included")
            round_index = 1
            backbone_path = (
                Path(self.config.backbone.checkpoint_dir)
                / f"pac_train_seed_{int(model_seed)}"
                / "checkpoint.pt"
            )
            backbone, _meta = load_alpha_model_checkpoint(backbone_path)
            name = (
                f"residual_rl_seed_{int(model_seed)}_round{round_index}.pt"
            )
            source_dir = (
                self.wm_rl_dir if method == "wm_residual_rl" else self.rl_dir
            )
            checkpoint = source_dir / name
            policy, _metadata = load_rl_checkpoint(
                checkpoint, backbone=backbone, expected_model_seed=int(model_seed)
            )
            from pac.v4.rl.collect import LiveResidualPolicy

            policy = LiveResidualPolicy(policy, explore_std=0.0, seed=0)
        else:
            raise ValueError(f"unknown formal method: {method}")
        self._cache[key] = policy
        return policy


def run_task_methods(
        config,
        factory: MethodFactory,
        task: EpisodeTask,
        spec,
        combos: list[dict]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for combo in combos:
        policy = factory.policy_for(combo["method"], combo["model_seed"])
        wm_computer = factory.wm_computer() if combo["method"] == "wm_hybrid" else None
        metrics = run_v4_policy_episode(policy, spec, config, wm_computer=wm_computer)
        row: dict[str, Any] = {
            "method": combo["method"],
            "model_seed": combo["model_seed"],
            "block": task.block,
            "family": task.family,
            "scenario_id": task.scenario_id,
            "environment_seed": task.seed,
            "episode_uid": str(spec.episode_uid),
            "episode_fingerprint": episode_fingerprint(spec),
        }
        for column in METRIC_COLUMNS:
            source = METRIC_ALIASES.get(column, column)
            value = metrics.get(source)
            if value is None:
                raise ValueError(
                    f"method {combo['method']} missing metric {source}"
                )
            row[column] = float(value)
        rows.append(row)
    return rows


def _aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    grouping = ["block", "method", "model_seed"]
    columns = [column for column in METRIC_COLUMNS if column in raw.columns]
    overall = raw.groupby(grouping, dropna=False, as_index=False)[columns].mean()
    return overall


def _block_family_table(raw: pd.DataFrame) -> pd.DataFrame:
    columns = ["rmse_3d", "heading_rmse_deg", "applied_control_cost",
               "solver_deadline_miss_step_fraction"]
    return raw.groupby(
        ["block", "family", "method"], dropna=False, as_index=False
    )[columns].mean()


PREREGISTERED_COMPARISONS = (
    ("residual_rl", "v3_transformer"),
    ("wm_hybrid", "v3_transformer"),
    ("wm_residual_rl", "residual_rl"),
    ("sspo", "residual_rl"),
    ("wm_hybrid", "sspo"),
)


def _paired_effects(raw: pd.DataFrame) -> pd.DataFrame:
    """Model-seed level paired effects with t(4) CIs, per block."""
    key_columns = ["block", "family", "scenario_id", "environment_seed"]
    rows: list[dict[str, Any]] = []
    model_seeds = sorted({
        int(seed) for seed in raw["model_seed"].dropna().unique()
    })
    for metric in ("rmse_3d", "heading_rmse_deg", "applied_control_cost",
                   "solver_deadline_miss_step_fraction"):
        for learned, baseline in PREREGISTERED_COMPARISONS:
            if learned not in set(raw["method"]) or baseline not in set(raw["method"]):
                continue
            for block in ("seen", "unseen"):
                block_frame = raw[raw["block"] == block]
                seed_effects: list[float] = []
                for seed in model_seeds:
                    learned_frame = block_frame[
                        (block_frame["method"] == learned)
                        & (block_frame["model_seed"] == seed)
                    ]
                    if learned_frame.empty:
                        continue
                    if baseline in LEARNED_METHODS:
                        base_frame = block_frame[
                            (block_frame["method"] == baseline)
                            & (block_frame["model_seed"] == seed)
                        ]
                    else:
                        base_frame = block_frame[block_frame["method"] == baseline]
                    merged = learned_frame.merge(
                        base_frame[key_columns + [metric]],
                        on=key_columns,
                        how="inner",
                        suffixes=("_learned", "_baseline"),
                        validate="one_to_one",
                    )
                    if merged.empty:
                        continue
                    differences = (
                        merged[f"{metric}_learned"].to_numpy(dtype=float)
                        - merged[f"{metric}_baseline"].to_numpy(dtype=float)
                    )
                    seed_effects.append(float(np.mean(differences)))
                if len(seed_effects) < 2:
                    continue
                effects = np.asarray(seed_effects, dtype=float)
                mean_difference = float(np.mean(effects))
                sample_sd = float(np.std(effects, ddof=1))
                half_width = _T_CRIT_4 * sample_sd / math.sqrt(len(effects))
                rows.append({
                    "metric": metric,
                    "comparison": f"{learned}_vs_{baseline}",
                    "block": block,
                    "n_model_seeds": len(effects),
                    "mean_difference": mean_difference,
                    "ci95_low": mean_difference - half_width,
                    "ci95_high": mean_difference + half_width,
                    "sample_sd": sample_sd,
                    "ci_basis": "model_seed_effects_t4",
                })
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_WORKER_STATE: dict[str, Any] = {}


def _worker_initialize(arguments) -> None:
    (config_path, rl_dir, wm_rl_dir, wm_checkpoint, dataset_dir,
     sspo_schedule, include_wm_rl, wm_sigma) = arguments
    import torch

    torch.set_num_threads(1)
    config = load_v4_config(config_path)
    _WORKER_STATE["config"] = config
    _WORKER_STATE["factory"] = MethodFactory(
        config,
        rl_dir=Path(rl_dir),
        wm_rl_dir=Path(wm_rl_dir),
        wm_checkpoint=Path(wm_checkpoint),
        dataset_dir=Path(dataset_dir),
        sspo_schedule=dict(sspo_schedule),
        include_wm_rl=include_wm_rl,
        wm_sigma=wm_sigma,
    )
    _WORKER_STATE["combos"] = method_grid(config, include_wm_rl=include_wm_rl)


def _worker_run_task(task: EpisodeTask) -> list[dict[str, Any]]:
    config = _WORKER_STATE["config"]
    spec = build_task_spec(config, task)
    return run_task_methods(
        config, _WORKER_STATE["factory"], task, spec, _WORKER_STATE["combos"]
    )


def run_v4_formal(
        *,
        config_path: str | Path,
        dataset_dir: str | Path,
        rl_dir: str | Path,
        wm_checkpoint: str | Path,
        sspo_schedule_path: str | Path,
        output_root: str | Path,
        profile: str,
        run_id: str | None = None,
        include_wm_rl: bool = False,
        wm_rl_dir: str | Path | None = None,
        jobs: int = 1,
        progress_every: int = 0,
        resume_from: str | Path | None = None,
        log: Callable[[str], None] = lambda message: None) -> Path | dict[str, Any]:
    """Run the paired formal v4 grid; returns the run directory.

    ``resume_from`` points at a ``partial_rows.jsonl`` saved by an earlier
    interrupted run (crash, OOM kill, reboot): tasks whose episodes are
    already complete in that file are skipped and its rows are merged into
    the final tables.
    """
    config_path = Path(config_path).resolve()
    config = load_v4_config(config_path)
    dataset_dir = Path(dataset_dir).resolve()
    rl_dir = Path(rl_dir).resolve()
    wm_checkpoint = Path(wm_checkpoint).resolve()
    sspo_schedule_path = Path(sspo_schedule_path).resolve()
    output_root = Path(output_root).resolve()
    tasks = build_episode_tasks(config, profile)
    combos = method_grid(config, include_wm_rl=include_wm_rl)
    plan = {
        "profile": profile,
        "episodes": len(tasks),
        "method_seed_combos": len(combos),
        "rollouts": len(tasks) * len(combos),
        "blocks": {
            "seen": sum(1 for task in tasks if task.block == "seen"),
            "unseen": sum(1 for task in tasks if task.block == "unseen"),
        },
    }
    if profile == "dry":
        return {**plan, "writes": False,
                "config": str(config_path), "output_root": str(output_root)}

    sspo_schedule = json.loads(sspo_schedule_path.read_text(encoding="utf-8"))
    resolved_wm_rl_dir = Path(wm_rl_dir).resolve() if wm_rl_dir else rl_dir
    # Compute the conservative-fallback sigmas once in the parent: loading
    # the multi-GB transition dataset in every worker exhausted RAM.
    from pac.v4.collector import load_transition_dataset
    from pac.v4.worldmodel.validate import train_residual_sigma

    wm_sigma = train_residual_sigma(load_transition_dataset(dataset_dir))
    if run_id is None or not str(run_id).strip():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}-{os.getpid()}"
    if not _RUN_ID_RE.fullmatch(str(run_id)):
        raise ValueError("run_id must contain only safe filename characters")
    target = output_root / str(run_id)
    if target.exists():
        raise FileExistsError(f"formal v4 run directory already exists: {target}")
    resumed_rows: list[dict[str, Any]] = []
    if resume_from is not None:
        resume_path = Path(resume_from).resolve()
        with resume_path.open("r", encoding="utf-8") as stream:
            resumed_rows = [
                json.loads(line)
                for line in stream
                if line.strip()
            ]
        counts: dict[str, int] = {}
        for row in resumed_rows:
            counts[row["episode_uid"]] = counts.get(row["episode_uid"], 0) + 1
        combo_count = len(combos)
        complete_uids = {
            uid for uid, count in counts.items() if count >= combo_count
        }
        before = len(tasks)
        tasks = [task for task in tasks
                 if str(build_task_spec(config, task).episode_uid)
                 not in complete_uids]
        log(
            f"[formal-v4] resumed {len(resumed_rows)} rows; skipping "
            f"{before - len(tasks)} complete tasks, {len(tasks)} remaining"
        )
        plan = {
            **plan,
            "episodes": len(tasks),
            "rollouts": len(tasks) * combo_count,
            "resumed_rows": len(resumed_rows),
        }
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=str(output_root)))
    try:
        rows: list[dict[str, Any]] = list(resumed_rows)
        completed = 0
        total = plan["rollouts"]
        if jobs > 1:
            import multiprocessing as mp

            context = mp.get_context("spawn")
            initializer_arguments = (
                str(config_path), str(rl_dir), str(resolved_wm_rl_dir),
                str(wm_checkpoint), str(dataset_dir), sspo_schedule,
                include_wm_rl, wm_sigma,
            )
            partial_rows_path = temporary / "partial_rows.jsonl"
            if resumed_rows:
                with partial_rows_path.open("w", encoding="utf-8") as stream:
                    for row in resumed_rows:
                        stream.write(json.dumps(row, sort_keys=True) + chr(10))
            with context.Pool(
                    processes=int(jobs),
                    initializer=_worker_initialize,
                    initargs=[initializer_arguments],
            ) as pool:
                for task_rows in pool.imap_unordered(_worker_run_task, tasks):
                    rows.extend(task_rows)
                    # Incremental persistence: a crash or OOM kill costs at
                    # most the in-flight tasks, never the whole grid.
                    with partial_rows_path.open("a", encoding="utf-8") as stream:
                        for row in task_rows:
                            stream.write(json.dumps(row, sort_keys=True) + chr(10))
                    completed += len(combos)
                    if progress_every and (
                            completed <= progress_every * len(combos)
                            or completed % (progress_every * len(combos)) == 0
                    ):
                        log(f"[formal-v4] {completed}/{total} rollouts")
        else:
            factory = MethodFactory(
                config,
                rl_dir=rl_dir,
                wm_rl_dir=resolved_wm_rl_dir,
                wm_checkpoint=wm_checkpoint,
                dataset_dir=dataset_dir,
                sspo_schedule=sspo_schedule,
                include_wm_rl=include_wm_rl,
                wm_sigma=wm_sigma,
            )
            for task in tasks:
                spec = build_task_spec(config, task)
                task_rows = run_task_methods(config, factory, task, spec, combos)
                rows.extend(task_rows)
                completed += len(combos)
                if progress_every and (
                        completed <= progress_every * len(combos)
                        or completed % (progress_every * len(combos)) == 0
                ):
                    log(f"[formal-v4] {completed}/{total} episode={spec.episode_uid}")
        raw = pd.DataFrame(rows)
        raw.to_csv(temporary / "raw_metrics.csv", index=False)
        overall = _aggregate(raw)
        overall.to_csv(temporary / "overall_summary.csv", index=False)
        _block_family_table(raw).to_csv(temporary / "block_family_summary.csv", index=False)
        paired = _paired_effects(raw)
        paired.to_csv(temporary / "paired_effects.csv", index=False)
        summary = {
            "protocol_version": config.protocol.version,
            "profile": profile,
            "run_id": str(run_id),
            "plan": plan,
            "preregistered_comparisons": [
                f"{a}_vs_{b}" for a, b in PREREGISTERED_COMPARISONS
            ],
            "raw_rows": int(len(raw)),
            "paired_rows": int(len(paired)),
            "ci_basis": "model-seed level paired effects, t(4)",
            "status": "complete",
        }
        (temporary / "formal_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        checkpoint_hashes = {}
        for label, directory in (("rl", rl_dir), ("wm_rl", resolved_wm_rl_dir)):
            if directory.is_dir():
                for seed in config.seeds.model:
                    checkpoint_hashes[f"{label}/seed_{seed}"] = _sha256(
                        directory / f"residual_rl_seed_{seed}_round1.pt"
                    )
        manifest = {
            "protocol_version": config.protocol.version,
            "profile": profile,
            "run_id": str(run_id),
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "dataset_dir": str(dataset_dir),
            "wm_checkpoint": str(wm_checkpoint),
            "wm_checkpoint_sha256": _sha256(wm_checkpoint),
            "sspo_schedule_path": str(sspo_schedule_path),
            "sspo_schedule_sha256": _sha256(sspo_schedule_path),
            "rl_checkpoint_sha256": checkpoint_hashes,
            "model_seeds": list(config.seeds.model),
            "episode_grid": [
                {
                    "block": task.block,
                    "family": task.family,
                    "scenario_id": task.scenario_id,
                    "environment_seed": task.seed,
                    "profile": task.profile,
                }
                for task in tasks
            ],
            "methods": [combo["method"] for combo in combos],
            "git_commit": _git_commit(),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        artifact_hashes = {
            path.relative_to(temporary).as_posix(): _sha256(path)
            for path in sorted(temporary.glob("*"))
            if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS.txt"}
        }
        (temporary / "SHA256SUMS.txt").write_text(
            "".join(f"{digest}  {path}\n" for path, digest in artifact_hashes.items()),
            encoding="utf-8",
        )
        if target.exists():
            raise FileExistsError(f"formal v4 run directory appeared: {target}")
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            recovered = output_root / f".recovered-{run_id}.jsonl"
            partial = temporary / "partial_rows.jsonl"
            if partial.is_file() and partial.stat().st_size > 0:
                shutil.copy2(partial, recovered)
            shutil.rmtree(temporary, ignore_errors=True)
    return target


def _git_commit() -> str | None:
    import subprocess

    try:
        root = Path(__file__).resolve().parents[4]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


__all__ = [
    "METRIC_COLUMNS",
    "PREREGISTERED_COMPARISONS",
    "build_episode_tasks",
    "build_task_spec",
    "method_grid",
    "run_v4_formal",
]
