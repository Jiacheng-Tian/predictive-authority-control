"""Run the formal seeded PAC evaluation protocol.

The formal evaluation protocol separates model-training randomness from evaluation
episode randomness:

- five independent PAC training seeds;
- ten randomized evaluation episodes per training seed and scenario;
- fixed controllers evaluated once on the same ten episode seeds per scenario.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from pac.config import PACConfig, load_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "pac.yaml"
DEFAULT_CONFIG = load_config(DEFAULT_CONFIG_PATH)
DEFAULT_OUT_DIR = ROOT / DEFAULT_CONFIG.outputs.formal_run_dir
DEFAULT_FIGURE_DIR = DEFAULT_OUT_DIR / "figures"
DEFAULT_TRAIN_SEEDS = list(DEFAULT_CONFIG.training.model_seeds)
DEFAULT_EVAL_EPISODE_SEEDS = list(DEFAULT_CONFIG.evaluation.episode_seeds)
DEFAULT_SCENARIOS = list(DEFAULT_CONFIG.environment.scenarios)
PAC_RUN_PREFIX = "pac_train_seed"


def _csv(values) -> str:
    return ",".join(str(value) for value in values)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def _portable_path(path: str | Path) -> str:
    """Return a package-relative path when the target is inside the release."""
    resolved = _resolve(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_formal_protocol_manifest(
        *,
        out_dir: Path | str = DEFAULT_OUT_DIR,
        figure_dir: Path | str = DEFAULT_FIGURE_DIR,
        train_seeds: list[int] | None = None,
        eval_episode_seeds: list[int] | None = None,
        scenarios: list[int] | None = None,
        steps: int = 2100,
        epochs: int = 180,
        config: PACConfig = DEFAULT_CONFIG) -> dict:
    """Return a formal manifest for the seeded evaluation protocol."""
    train_seeds = list(DEFAULT_TRAIN_SEEDS if train_seeds is None else train_seeds)
    eval_episode_seeds = list(
        DEFAULT_EVAL_EPISODE_SEEDS
        if eval_episode_seeds is None else eval_episode_seeds
    )
    scenarios = list(DEFAULT_SCENARIOS if scenarios is None else scenarios)
    return {
        "version": "formal_seeded_v2_5trainx10eval",
        "main_methods": ["SMC", "MPC", "PAC"],
        "pac_training_seeds": train_seeds,
        "evaluation_episode_seeds": eval_episode_seeds,
        "evaluation_scenarios": scenarios,
        "evaluation_episodes_per_train_seed": len(eval_episode_seeds),
        "total_pac_rollouts_per_scenario": len(train_seeds) * len(eval_episode_seeds),
        "fixed_controller_rollouts_per_scenario": len(eval_episode_seeds),
        "output_directory": _portable_path(out_dir),
        "figure_directory": _portable_path(figure_dir),
        "shared_environment": {
            "steps": int(steps),
            "mass_scale_xy": config.environment.mass_scale_xy,
            "damping_scale_xy": config.environment.damping_scale_xy,
            "current_amplitude_scale": config.environment.current_amplitude_scale,
            "current_frequency_scale": config.environment.current_frequency_scale,
            "eval_initial_position_std": config.environment.eval_initial_position_std,
            "eval_initial_velocity_std": config.environment.eval_initial_velocity_std,
            "vertical_current": config.environment.vertical_current,
            "vehicle_profile": config.environment.vehicle_profile,
            "action_mode": "thruster",
            "thruster_layout": config.environment.thruster_layout,
        },
        "pac": {
            "policy_architecture": "transformer",
            "history_len": config.authority.history_len,
            "transformer_embed_dim": config.authority.embed_dim,
            "transformer_heads": config.authority.heads,
            "transformer_layers": config.authority.layers,
            "model_dropout": config.authority.dropout,
            "epochs": int(epochs),
            "teacher_train_scenarios": scenarios,
            "teacher_train_seeds": list(config.training.data_seeds),
            "teacher_mode": "oracle",
            "primary_controller": config.controllers.primary,
            "authority_controller": config.controllers.authority,
            "alpha_gain": config.authority.alpha_gain,
            "alpha_smoothing": config.authority.alpha_smoothing,
            "alpha_rate_limit": config.authority.alpha_rate_limit,
        },
    }


def pac_seed_out_dir(root: Path | str, train_seed: int) -> Path:
    """Return the run directory for one PAC training seed."""
    return _resolve(root) / f"{PAC_RUN_PREFIX}_{int(train_seed)}"


def formal_eval_episode_rows(
        *,
        train_seeds: list[int],
        eval_episode_seeds: list[int]) -> list[dict[str, int]]:
    """Map formal train/eval dimensions onto unique simulator seeds."""
    rows = []
    for train_seed in train_seeds:
        for eval_episode in eval_episode_seeds:
            rows.append({
                "train_seed": int(train_seed),
                "eval_episode": int(eval_episode),
                "sim_seed": int(train_seed) * 1000 + int(eval_episode),
            })
    return rows


def annotate_formal_eval_dimensions(
        data: pd.DataFrame,
        *,
        train_seeds: list[int],
        eval_episode_seeds: list[int]) -> pd.DataFrame:
    """Recover formal train/eval episode labels from simulator seed values."""
    if "seed" not in data.columns:
        raise ValueError("formal evaluation table must contain seed column")
    mapping = pd.DataFrame(
        formal_eval_episode_rows(
            train_seeds=train_seeds,
            eval_episode_seeds=eval_episode_seeds,
        )
    )
    out = data.copy()
    out = out.drop(
        columns=["train_seed", "eval_episode", "episode_uid", "sim_seed"],
        errors="ignore",
    )
    out["seed"] = pd.to_numeric(out["seed"], errors="raise").astype(int)
    out = out.merge(mapping, left_on="seed", right_on="sim_seed", how="left")
    if out["train_seed"].isna().any() or out["eval_episode"].isna().any():
        missing = sorted(out.loc[out["train_seed"].isna(), "seed"].unique().tolist())
        raise ValueError(f"Encountered seeds outside formal protocol: {missing}")
    if "scenario_id" not in out.columns:
        raise ValueError("formal evaluation table must contain scenario_id column")
    out["train_seed"] = out["train_seed"].astype(int)
    out["eval_episode"] = out["eval_episode"].astype(int)
    out["episode_uid"] = (
        "train"
        + out["train_seed"].astype(str)
        + "_eval"
        + out["eval_episode"].astype(str)
        + "_scn"
        + out["scenario_id"].astype(str)
    )
    return out.drop(columns=["sim_seed"], errors="ignore")


def annotate_fixed_controller_outputs(
        fixed_dir: Path | str,
        *,
        train_seeds: list[int],
        eval_episode_seeds: list[int]) -> list[Path]:
    """Annotate fixed-controller CSV outputs with formal episode dimensions."""
    fixed_dir = _resolve(fixed_dir)
    written = []
    required = ["raw_metrics.csv", "window_metrics.csv"]
    optional = ["timeseries/timeseries_3d.csv"]
    for filename in required + optional:
        path = fixed_dir / filename
        if not path.exists():
            if filename in required:
                raise FileNotFoundError(f"Missing fixed-controller output: {path}")
            continue
        data = annotate_formal_eval_dimensions(
            pd.read_csv(path),
            train_seeds=train_seeds,
            eval_episode_seeds=eval_episode_seeds,
        )
        data.to_csv(path, index=False)
        written.append(path)
    return written


def build_pac_command(
        *,
        out_dir: Path | str,
        figure_dir: Path | str,
        train_seed: int,
        eval_episode_seeds: list[int],
        scenarios: list[int],
        steps: int,
        epochs: int,
        no_timeseries: bool = False,
        progress_every: int = 10,
        config: PACConfig = DEFAULT_CONFIG,
        config_path: str | Path = DEFAULT_CONFIG_PATH) -> list[str]:
    """Build one PAC training/evaluation command for a single training seed."""
    eval_sim_seeds = [
        row["sim_seed"]
        for row in formal_eval_episode_rows(
            train_seeds=[int(train_seed)],
            eval_episode_seeds=eval_episode_seeds,
        )
    ]
    command = [
        sys.executable,
        "-m", "pac.authority.pipeline",
        "--config", str(config_path),
        "--out-dir", str(out_dir),
        "--figure-dir", str(figure_dir),
        "--steps", str(int(steps)),
        "--train-scenarios", _csv(scenarios),
        "--train-seeds", _csv(config.training.data_seeds),
        "--eval-scenarios", _csv(scenarios),
        "--eval-seeds", _csv(eval_sim_seeds),
        "--mass-scale-xy", str(config.environment.mass_scale_xy),
        "--damping-scale-xy", str(config.environment.damping_scale_xy),
        "--current-amplitude-scale", str(config.environment.current_amplitude_scale),
        "--current-frequency-scale", str(config.environment.current_frequency_scale),
        "--eval-initial-position-std", str(config.environment.eval_initial_position_std),
        "--eval-initial-velocity-std", str(config.environment.eval_initial_velocity_std),
        "--vertical-current", str(config.environment.vertical_current),
        "--vehicle-profile", config.environment.vehicle_profile,
        "--thruster-layout", config.environment.thruster_layout,
        "--primary-controller", config.controllers.primary,
        "--authority-controller", config.controllers.authority,
        "--teacher-mode", "oracle",
        "--feature-mode", "state_phase",
        "--policy-architecture", "transformer",
        "--history-len", str(config.authority.history_len),
        "--transformer-embed-dim", str(config.authority.embed_dim),
        "--transformer-heads", str(config.authority.heads),
        "--transformer-layers", str(config.authority.layers),
        "--model-dropout", str(config.authority.dropout),
        "--epochs", str(int(epochs)),
        "--batch-size", str(config.training.batch_size),
        "--lr", str(config.training.learning_rate),
        "--train-seed", str(int(train_seed)),
        "--alpha-gain", str(config.authority.alpha_gain),
        "--alpha-threshold", "0.0",
        "--oracle-alpha-grid", _csv(config.authority.oracle_alpha_grid),
        "--oracle-horizon-steps", str(config.authority.oracle_horizon_steps),
        "--oracle-action-saturation-weight", str(config.authority.oracle_action_saturation_weight),
        "--oracle-action-delta-weight", str(config.authority.oracle_action_delta_weight),
        "--oracle-alpha-delta-weight", str(config.authority.oracle_alpha_delta_weight),
        "--alpha-smoothing", str(config.authority.alpha_smoothing),
        "--alpha-rate-limit", str(config.authority.alpha_rate_limit),
        "--alpha-deadband", str(config.authority.alpha_deadband),
        "--progress-every", str(int(progress_every)),
    ]
    command.append("--no-save-timeseries" if no_timeseries else "--save-timeseries")
    return command


def build_fixed_controller_command(
        *,
        out_dir: Path | str,
        figure_dir: Path | str,
        train_seeds: list[int],
        eval_episode_seeds: list[int],
        scenarios: list[int],
        steps: int,
        no_figures: bool = False,
        config: PACConfig = DEFAULT_CONFIG) -> list[str]:
    """Build the fixed-controller command for one shared set of eval episodes."""
    if not train_seeds:
        raise ValueError("train_seeds must contain a reference seed")
    sim_seeds = [
        row["sim_seed"]
        for row in formal_eval_episode_rows(
            train_seeds=[train_seeds[0]],
            eval_episode_seeds=eval_episode_seeds,
        )
    ]
    command = [
        sys.executable,
        "-m", "pac.evaluation.diagnostics",
        "--mode", "baseline",
        "--out-dir", str(out_dir),
        "--figure-dir", str(figure_dir),
        "--scenarios", _csv(scenarios),
        "--seeds", _csv(sim_seeds),
        "--steps", str(int(steps)),
        "--mass-scale-xy", str(config.environment.mass_scale_xy),
        "--damping-scale-xy", str(config.environment.damping_scale_xy),
        "--current-amplitude-scales", str(config.environment.current_amplitude_scale),
        "--current-frequency-scales", str(config.environment.current_frequency_scale),
        "--initial-position-std", str(config.environment.eval_initial_position_std),
        "--initial-velocity-std", str(config.environment.eval_initial_velocity_std),
        "--vertical-current", str(config.environment.vertical_current),
        "--vehicle-profile", config.environment.vehicle_profile,
        "--thruster-layout", config.environment.thruster_layout,
        "--base-controllers", _csv([config.controllers.primary, config.controllers.authority]),
        "--reference-controller", config.controllers.primary,
    ]
    if no_figures:
        command.append("--no-figures")
    return command


def _read_seed_table(
        run_dir: Path,
        filename: str,
        train_seed: int,
        eval_episode_seeds: list[int]) -> pd.DataFrame:
    path = run_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing PAC seed output: {path}")
    data = pd.read_csv(path)
    if "seed" not in data.columns:
        raise ValueError(f"{path} must contain seed column")
    return annotate_formal_eval_dimensions(
        data,
        train_seeds=[int(train_seed)],
        eval_episode_seeds=eval_episode_seeds,
    )


def _read_optional_seed_timeseries(
        run_dir: Path,
        train_seed: int,
        eval_episode_seeds: list[int]) -> pd.DataFrame:
    path = run_dir / "timeseries" / "timeseries_3d.csv"
    if not path.exists():
        return pd.DataFrame()
    return annotate_formal_eval_dimensions(
        pd.read_csv(path),
        train_seeds=[int(train_seed)],
        eval_episode_seeds=eval_episode_seeds,
    )


def merge_pac_seed_outputs(
        root_dir: Path | str,
        *,
        train_seeds: list[int] | None = None,
        eval_episode_seeds: list[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge per-training-seed PAC outputs and expose both seed dimensions."""
    root = _resolve(root_dir)
    train_seeds = list(DEFAULT_TRAIN_SEEDS if train_seeds is None else train_seeds)
    eval_episode_seeds = list(
        DEFAULT_EVAL_EPISODE_SEEDS
        if eval_episode_seeds is None else eval_episode_seeds
    )
    raw_frames = []
    window_frames = []
    timeseries_frames = []
    for train_seed in train_seeds:
        run_dir = pac_seed_out_dir(root, train_seed)
        raw_frames.append(_read_seed_table(
            run_dir,
            "raw_metrics.csv",
            train_seed,
            eval_episode_seeds,
        ))
        window_frames.append(_read_seed_table(
            run_dir,
            "window_metrics.csv",
            train_seed,
            eval_episode_seeds,
        ))
        timeseries = _read_optional_seed_timeseries(
            run_dir,
            train_seed,
            eval_episode_seeds,
        )
        if not timeseries.empty:
            timeseries_frames.append(timeseries)
    raw = pd.concat(raw_frames, ignore_index=True, sort=False)
    window = pd.concat(window_frames, ignore_index=True, sort=False)
    timeseries = (
        pd.concat(timeseries_frames, ignore_index=True, sort=False)
        if timeseries_frames else pd.DataFrame()
    )
    return raw, window, timeseries


def write_merged_pac_outputs(
        root_dir: Path | str,
        *,
        train_seeds: list[int] | None = None,
        eval_episode_seeds: list[int] | None = None) -> tuple[Path, Path]:
    """Write merged PAC raw/window tables for formal aggregation."""
    root = _resolve(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    raw, window, timeseries = merge_pac_seed_outputs(
        root,
        train_seeds=train_seeds,
        eval_episode_seeds=eval_episode_seeds,
    )
    raw_path = root / "raw_metrics.csv"
    window_path = root / "window_metrics.csv"
    raw.to_csv(raw_path, index=False)
    window.to_csv(window_path, index=False)
    # Per-seed timeseries are the source artifacts. Do not duplicate them into
    # a large merged file; callers can use ``merge_pac_seed_outputs`` in memory.
    return raw_path, window_path


def run_command(command: list[str], *, dry_run: bool = False) -> None:
    """Run or print a formal protocol command."""
    printable = " ".join(str(part) for part in command)
    if dry_run:
        print(printable)
        return
    print(printable, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def require_empty_output_directory(path: Path | str) -> None:
    """Refuse to overwrite an existing run directory."""
    target = Path(path)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"output directory is not empty: {target}")


def run_formal_protocol(args) -> None:
    """Run fixed controllers and PAC training seeds, then merge PAC outputs."""
    out_root = _resolve(args.out_dir)
    figure_root = _resolve(args.figure_dir)
    train_seeds = _parse_int_csv(args.train_seeds)
    eval_episode_seeds = _parse_int_csv(args.eval_episode_seeds)
    scenarios = _parse_int_csv(args.scenarios)
    config = getattr(args, "runtime_config", DEFAULT_CONFIG)
    manifest = build_formal_protocol_manifest(
        out_dir=out_root,
        figure_dir=figure_root,
        train_seeds=train_seeds,
        eval_episode_seeds=eval_episode_seeds,
        scenarios=scenarios,
        steps=args.steps,
        epochs=args.epochs,
        config=config,
    )
    if not args.dry_run:
        require_empty_output_directory(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "formal_protocol_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
    if not args.skip_fixed:
        fixed_out = out_root / "fixed_controllers"
        run_command(
            build_fixed_controller_command(
                out_dir=fixed_out,
                figure_dir=figure_root / "fixed_controllers",
                train_seeds=train_seeds,
                eval_episode_seeds=eval_episode_seeds,
                scenarios=scenarios,
                steps=args.steps,
                no_figures=args.no_figures,
                config=config,
            ),
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            annotate_fixed_controller_outputs(
                fixed_out,
                train_seeds=train_seeds,
                eval_episode_seeds=eval_episode_seeds,
            )
    if not args.skip_pac:
        for train_seed in train_seeds:
            run_command(
                build_pac_command(
                    out_dir=pac_seed_out_dir(out_root, train_seed),
                    figure_dir=figure_root / f"{PAC_RUN_PREFIX}_{train_seed}",
                    train_seed=train_seed,
                    eval_episode_seeds=eval_episode_seeds,
                    scenarios=scenarios,
                    steps=args.steps,
                    epochs=args.epochs,
                    no_timeseries=args.no_timeseries,
                    progress_every=args.progress_every,
                    config=config,
                    config_path=args.config,
                ),
                dry_run=args.dry_run,
            )
    if not args.dry_run and not args.skip_merge:
        raw_path, window_path = write_merged_pac_outputs(
            out_root,
            train_seeds=train_seeds,
            eval_episode_seeds=eval_episode_seeds,
        )
        print(f"Wrote merged PAC raw metrics: {raw_path}")
        print(f"Wrote merged PAC window metrics: {window_path}")


def _parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def build_parser(config: PACConfig = DEFAULT_CONFIG) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=_portable_path(DEFAULT_CONFIG_PATH))
    out_dir = ROOT / config.outputs.formal_run_dir
    parser.add_argument("--out-dir", default=str(out_dir))
    parser.add_argument("--figure-dir", default=str(out_dir / "figures"))
    parser.add_argument("--train-seeds", default=_csv(config.training.model_seeds))
    parser.add_argument("--eval-episode-seeds", default=_csv(config.evaluation.episode_seeds))
    parser.add_argument("--scenarios", default=_csv(config.environment.scenarios))
    parser.add_argument("--steps", type=int, default=config.environment.steps)
    parser.add_argument("--epochs", type=int, default=config.training.epochs)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--no-timeseries", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--skip-fixed", action="store_true")
    parser.add_argument("--skip-pac", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=_portable_path(DEFAULT_CONFIG_PATH))
    known, _unknown = config_parser.parse_known_args(argv)
    config = load_config(_resolve(known.config))
    args = build_parser(config).parse_args(argv)
    args.runtime_config = config
    run_formal_protocol(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
