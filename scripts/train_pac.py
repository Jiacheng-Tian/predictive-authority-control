#!/usr/bin/env python3
"""Train the immutable V3 predictive-authority checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

from pac.authority.dataset import load_oracle_dataset
from pac.authority.model import (
    _default_dependency_versions,
    _default_git_provenance,
    _semantic_config_hash,
    save_v3_checkpoint,
    train_alpha_model_v3,
)
from pac.experiment_config import load_v3_config


def _config_path(value: str) -> Path:
    if str(value).strip().lower() in {"pac_v3", "pac_v3.yaml"}:
        return ROOT / "config" / "pac_v3.yaml"
    path = Path(value)
    return path if path.is_absolute() else (Path.cwd() / path)


def _reject_output(path: Path) -> None:
    resolved = path.resolve()
    if any(part.lower() == "results" for part in resolved.parts):
        raise ValueError("V3 training output must not be under a results path")
    if resolved.exists():
        raise FileExistsError(f"output directory already exists: {resolved}")


def _reject_path_relationships(dataset_dir: Path, output_dir: Path) -> None:
    if dataset_dir == output_dir:
        raise ValueError("dataset and output directories must be different")
    try:
        output_dir.relative_to(dataset_dir)
    except ValueError:
        pass
    else:
        raise ValueError("output directory must not be inside dataset directory")
    try:
        dataset_dir.relative_to(output_dir)
    except ValueError:
        pass
    else:
        raise ValueError("dataset directory must not be inside output directory")


def _publish_output_directory(temporary_output: Path, output_dir: Path) -> None:
    """Atomically publish training output, tolerating transient Windows locks."""
    retry_delays = (0.05, 0.1, 0.2)
    for attempt, delay in enumerate(retry_delays):
        if output_dir.exists():
            raise FileExistsError(f"output directory appeared during training: {output_dir}")
        try:
            os.replace(temporary_output, output_dir)
            return
        except PermissionError:
            if attempt == len(retry_delays) - 1:
                raise
            time.sleep(delay)


def _read_dataset_manifest(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid dataset manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or not manifest.get("dataset_hash"):
        raise ValueError("dataset manifest must contain dataset_hash")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_hash(versions: dict[str, str]) -> str:
    encoded = json.dumps(versions, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def _profile_seeds(config, profile: str) -> list[int]:
    configured = [int(seed) for seed in config.training.model_seeds]
    required = [31000, 31001, 31002, 31003, 31004]
    if configured != required:
        raise ValueError("V3 training.model_seeds must be exactly [31000, 31001, 31002, 31003, 31004]")
    if profile == "short":
        return [31000]
    if profile == "formal":
        return required
    return required


def _dry_manifest(
        config,
        dataset_manifest: dict[str, Any],
        config_path: Path,
        git_provenance: dict[str, Any],
) -> dict[str, Any]:
    versions = _default_dependency_versions()
    dataset_manifest_path = Path(dataset_manifest.get("_manifest_path", ""))
    return {
        "profile": "dry",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "config_semantic_sha256": _semantic_config_hash(config.authority_model),
        "dataset_hash": dataset_manifest["dataset_hash"],
        "dataset_manifest_sha256": (
            _sha256(dataset_manifest_path) if dataset_manifest_path.is_file() else None
        ),
        "model_seeds": [int(seed) for seed in config.training.model_seeds],
        "dependency_versions": versions,
        "dependency_hash": _dependency_hash(versions),
        "protocol_version": config.protocol.version,
        **git_provenance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pac_v3")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--profile", choices=("dry", "short", "formal"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = _config_path(args.config).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.out_dir).resolve()
    _reject_path_relationships(dataset_dir, output_dir)
    _reject_output(output_dir)
    config = load_v3_config(config_path)
    git_provenance = _default_git_provenance()
    if (
        args.profile == "formal"
        and git_provenance["git_dirty"]
        and os.environ.get("PAC_ALLOW_DIRTY_FORMAL") != "1"
    ):
        raise ValueError(
            "formal V3 training requires a clean git worktree; "
            "set PAC_ALLOW_DIRTY_FORMAL=1 only for debugging"
        )
    if args.profile == "dry":
        _profile_seeds(config, "formal")
    dataset_manifest = _read_dataset_manifest(dataset_dir)
    dataset_manifest["_manifest_path"] = str(dataset_dir / "manifest.json")
    if args.profile == "dry":
        print(json.dumps(_dry_manifest(config, dataset_manifest, config_path, git_provenance), sort_keys=True))
        return 0

    seeds = _profile_seeds(config, args.profile)
    dataset = load_oracle_dataset(dataset_dir)
    dataset_hash = str(dataset_manifest["dataset_hash"])
    config_hash = _semantic_config_hash(config.authority_model)
    config_file_hash = _sha256(config_path)
    dataset_manifest_hash = _sha256(dataset_dir / "manifest.json")
    dependency_versions = _default_dependency_versions()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.tmp-",
        dir=output_dir.parent,
    ))
    checkpoint_records: dict[str, dict[str, Any]] = {}
    try:
        for seed in seeds:
            seed_dir = temporary_output / f"pac_train_seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            if args.profile == "short":
                max_epochs = min(3, int(config.training.max_epochs))
                run_patience = min(int(config.training.patience), max_epochs)
            else:
                max_epochs = int(config.training.max_epochs)
                run_patience = int(config.training.patience)
            model, history, metrics = train_alpha_model_v3(
                dataset,
                config.authority_model,
                model_seed=seed,
                max_epochs=max_epochs,
                patience=run_patience,
            )
            history.to_csv(seed_dir / "training_history.csv", index=False)
            summary = {
                **metrics,
                "dataset_hash": dataset_hash,
                "config_semantic_sha256": config_hash,
                "profile": args.profile,
                "checkpoint": "checkpoint.pt",
            }
            (seed_dir / "training_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True, default=_json_default),
                encoding="utf-8",
            )
            save_v3_checkpoint(
                seed_dir / "checkpoint.pt",
                model,
                config.authority_model,
                model_seed=seed,
                dataset_hash=dataset_hash,
                config_semantic_sha256=config_hash,
                git_commit=git_provenance["git_commit"],
                git_dirty=git_provenance["git_dirty"],
                git_diff_sha256=git_provenance["git_diff_sha256"],
                dependency_versions=dependency_versions,
                training_metrics=metrics,
            )
            checkpoint_records[str(seed)] = {
                "path": (seed_dir / "checkpoint.pt").relative_to(temporary_output).as_posix(),
                "sha256": _sha256(seed_dir / "checkpoint.pt"),
                "model_seed": seed,
                "dataset_hash": dataset_hash,
            }

        command = [sys.executable, "scripts/train_pac.py", *(sys.argv[1:] if argv is None else argv)]
        manifest = {
        "protocol_version": config.protocol.version,
        "profile": args.profile,
        "config_path": config_path.relative_to(ROOT).as_posix() if config_path.is_relative_to(ROOT) else str(config_path),
        "config_sha256": config_file_hash,
        "config_semantic_sha256": config_hash,
        "dataset_dir": str(dataset_dir),
        "dataset_hash": dataset_hash,
        "dataset_manifest_sha256": dataset_manifest_hash,
        "git_commit": git_provenance["git_commit"],
        "git_dirty": git_provenance["git_dirty"],
        "git_diff_sha256": git_provenance["git_diff_sha256"],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependency_versions": dependency_versions,
            "dependency_hash": _dependency_hash(dependency_versions),
        },
        "commands": [command],
        "model_seeds": seeds,
        "checkpoints": checkpoint_records,
        "checkpoint_hashes": {
            record["path"]: record["sha256"] for record in checkpoint_records.values()
        },
        }
        (temporary_output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8",
        )
        _publish_output_directory(temporary_output, output_dir)
        temporary_output = None
    finally:
        if temporary_output is not None:
            shutil.rmtree(temporary_output, ignore_errors=True)
    print(json.dumps(manifest, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
