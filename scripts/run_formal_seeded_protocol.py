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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "results" / "formal_seeded_v2"
DEFAULT_FIGURE_DIR = ROOT / "results" / "figures" / "formal_seeded_v2"
DEFAULT_TRAIN_SEEDS = [20, 21, 22, 23, 24]
DEFAULT_EVAL_EPISODE_SEEDS = list(range(10))
DEFAULT_SCENARIOS = [1, 2, 3]
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
        epochs: int = 180) -> dict:
    """Return a formal manifest for the seeded evaluation protocol."""
    train_seeds = list(DEFAULT_TRAIN_SEEDS if train_seeds is None else train_seeds)
    eval_episode_seeds = list(
        DEFAULT_EVAL_EPISODE_SEEDS
        if eval_episode_seeds is None else eval_episode_seeds
    )
    scenarios = list(DEFAULT_SCENARIOS if scenarios is None else scenarios)
    return {
        "version": "formal_seeded_v2_5trainx10eval",
        "main_methods": ["SMC", "Authority controller", "PAC"],
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
            "mass_scale_xy": 1.0,
            "damping_scale_xy": 1.0,
            "current_amplitude_scale": 2.5,
            "current_frequency_scale": 1.0,
            "eval_initial_position_std": 0.03,
            "eval_initial_velocity_std": 0.01,
            "vertical_current": 0.75,
            "vehicle_profile": "real_10kg_v1",
            "action_mode": "thruster",
            "thruster_layout": "real_10kg_x",
        },
        "pac": {
            "policy_architecture": "transformer",
            "history_len": 16,
            "transformer_embed_dim": 32,
            "transformer_heads": 4,
            "transformer_layers": 1,
            "model_dropout": 0.1,
            "epochs": int(epochs),
            "teacher_train_scenarios": scenarios,
            "teacher_train_seeds": [0, 1, 2],
            "teacher_mode": "oracle",
            "primary_controller": "real10kg_smc_steady",
            "authority_controller": "real10kg_mpc_event",
            "alpha_gain": 1.2,
            "alpha_smoothing": 0.5,
            "alpha_rate_limit": 0.0125,
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
        progress_every: int = 10) -> list[str]:
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
        "code/evaluation/predictive_3d_authority_alpha.py",
        "--out-dir", str(out_dir),
        "--figure-dir", str(figure_dir),
        "--steps", str(int(steps)),
        "--train-scenarios", _csv(scenarios),
        "--train-seeds", "0,1,2",
        "--eval-scenarios", _csv(scenarios),
        "--eval-seeds", _csv(eval_sim_seeds),
        "--mass-scale-xy", "1.0",
        "--damping-scale-xy", "1.0",
        "--current-amplitude-scale", "2.5",
        "--current-frequency-scale", "1.0",
        "--eval-initial-position-std", "0.03",
        "--eval-initial-velocity-std", "0.01",
        "--vertical-current", "0.75",
        "--vehicle-profile", "real_10kg_v1",
        "--action-mode", "thruster",
        "--thruster-layout", "real_10kg_x",
        "--primary-controller", "real10kg_smc_steady",
        "--authority-controller", "real10kg_mpc_event",
        "--teacher-authority-until", "2.0",
        "--teacher-blend-duration", "0.5",
        "--teacher-blend-curve", "smoothstep",
        "--teacher-mode", "oracle",
        "--feature-mode", "state_phase",
        "--hidden-dim", "64",
        "--policy-architecture", "transformer",
        "--history-len", "16",
        "--transformer-embed-dim", "32",
        "--transformer-heads", "4",
        "--transformer-layers", "1",
        "--model-dropout", "0.1",
        "--epochs", str(int(epochs)),
        "--batch-size", "512",
        "--lr", "0.001",
        "--train-seed", str(int(train_seed)),
        "--alpha-gain", "1.2",
        "--alpha-threshold", "0.0",
        "--eval-alpha-mode", "model",
        "--eval-methods", "predictive_alpha",
        "--oracle-alpha-grid", "0,0.25,0.5,0.75,1",
        "--oracle-horizon-steps", "10",
        "--oracle-action-saturation-weight", "0.02",
        "--oracle-action-delta-weight", "0.01",
        "--oracle-alpha-delta-weight", "0.0",
        "--alpha-smoothing", "0.5",
        "--alpha-rate-limit", "0.0125",
        "--alpha-deadband", "0.0",
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
        no_figures: bool = False) -> list[str]:
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
        "code/evaluation/diagnose_3d_authority.py",
        "--mode", "baseline",
        "--out-dir", str(out_dir),
        "--figure-dir", str(figure_dir),
        "--scenarios", _csv(scenarios),
        "--seeds", _csv(sim_seeds),
        "--steps", str(int(steps)),
        "--mass-scale-xy", "1.0",
        "--damping-scale-xy", "1.0",
        "--current-amplitude-scales", "2.5",
        "--current-frequency-scales", "1.0",
        "--initial-position-std", "0.03",
        "--initial-velocity-std", "0.01",
        "--vertical-current", "0.75",
        "--vehicle-profile", "real_10kg_v1",
        "--action-mode", "thruster",
        "--thruster-layout", "real_10kg_x",
        "--base-controllers", "real10kg_smc_steady,real10kg_mpc_event",
        "--reference-controller", "real10kg_smc_steady",
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
    if not timeseries.empty:
        ts_dir = root / "timeseries"
        ts_dir.mkdir(parents=True, exist_ok=True)
        timeseries.to_csv(ts_dir / "timeseries_3d.csv", index=False)
    return raw_path, window_path


def run_command(command: list[str], *, dry_run: bool = False) -> None:
    """Run or print a formal protocol command."""
    printable = " ".join(str(part) for part in command)
    if dry_run:
        print(printable)
        return
    print(printable, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def run_formal_protocol(args) -> None:
    """Run fixed controllers and PAC training seeds, then merge PAC outputs."""
    out_root = _resolve(args.out_dir)
    figure_root = _resolve(args.figure_dir)
    train_seeds = _parse_int_csv(args.train_seeds)
    eval_episode_seeds = _parse_int_csv(args.eval_episode_seeds)
    scenarios = _parse_int_csv(args.scenarios)
    manifest = build_formal_protocol_manifest(
        out_dir=out_root,
        figure_dir=figure_root,
        train_seeds=train_seeds,
        eval_episode_seeds=eval_episode_seeds,
        scenarios=scenarios,
        steps=args.steps,
        epochs=args.epochs,
    )
    if not args.dry_run:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--train-seeds", default=_csv(DEFAULT_TRAIN_SEEDS))
    parser.add_argument("--eval-episode-seeds", default=_csv(DEFAULT_EVAL_EPISODE_SEEDS))
    parser.add_argument("--scenarios", default=_csv(DEFAULT_SCENARIOS))
    parser.add_argument("--steps", type=int, default=2100)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--no-timeseries", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--skip-fixed", action="store_true")
    parser.add_argument("--skip-pac", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run_formal_protocol(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
