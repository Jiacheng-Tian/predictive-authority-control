"""Verify PAC repository contents without rewriting integrity baselines."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import re
from collections import Counter
from pathlib import Path

from pac.authority.model import load_alpha_model_checkpoint
from pac.config import load_config
from pac.evaluation.summary import summarize_formal_results
from pac.repository import file_sha256, repository_files

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    ".gitignore",
    ".gitattributes",
    "pyproject.toml",
    "requirements-lock.txt",
    "config/pac.yaml",
    "src/pac/config.py",
    "src/pac/repository.py",
    "src/pac/simulation/core.py",
    "src/pac/simulation/environment.py",
    "src/pac/simulation/dynamics.py",
    "src/pac/simulation/thrusters.py",
    "src/pac/simulation/vehicle_profiles.py",
    "src/pac/controllers/smc.py",
    "src/pac/controllers/predictive.py",
    "src/pac/controllers/presets.py",
    "src/pac/authority/features.py",
    "src/pac/authority/model.py",
    "src/pac/authority/training.py",
    "src/pac/authority/evaluation.py",
    "src/pac/authority/pipeline.py",
    "src/pac/evaluation/episodes.py",
    "src/pac/evaluation/diagnostics.py",
    "src/pac/evaluation/metrics.py",
    "src/pac/evaluation/summary.py",
    "scripts/run_formal_seeded_protocol.py",
    "scripts/sspo_alpha_calibration.py",
    "scripts/summarize_formal_results.py",
    "scripts/update_release_manifest.py",
    "scripts/verify_repository.py",
    "results/formal_seeded_v2/formal_protocol_manifest.json",
    "results/formal_seeded_v2/raw_metrics.csv",
    "results/formal_seeded_v2/window_metrics.csv",
    "results/formal_seeded_v2/fixed_controllers/raw_metrics.csv",
    "results/formal_seeded_v2/fixed_controllers/timeseries/timeseries_3d.csv",
    "results/3d_authority_diagnosis/sspo_alpha_calibration_seed20_v2_regret/sspo_summary.json",
    "results/3d_authority_diagnosis/sspo_alpha_calibration_seed20_v2_regret/window_metrics.csv",
    "results/3d_authority_diagnosis/sspo_alpha_calibration_seed20_v2_regret/timeseries/timeseries_3d.csv",
]
FORBIDDEN_PATH_FRAGMENTS = {
    "release_packages" + "0608",
    "release_packages" + "0609",
    "pre_" + "transformer",
    "nn_" + "direct_control",
    "mlp_" + "pac",
    "formal_seeded_" + "smoke",
    "manual_fig3_" + "tuner",
    "pro" + "be_",
    "pac_epoch_" + "sweep",
    "tuning_" + "mlp",
    "ren" + "dered",
    "qa_" + "render",
    "__" + "pycache__",
    "ic" + "ra",
}
TEXT_SUFFIXES = {".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
ABSOLUTE_PATH = re.compile(r"[A-Za-z]:[\\/]")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def tested_environment() -> dict[str, object]:
    packages = {}
    for name in ["torch", "gymnasium", "numpy", "matplotlib", "pandas", "PyYAML"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def verify_evidence(failures: list[str]) -> None:
    def read_rows(relative: str) -> list[dict[str, str]]:
        with (ROOT / relative).open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    pac = read_rows("results/formal_seeded_v2/raw_metrics.csv")
    fixed = read_rows("results/formal_seeded_v2/fixed_controllers/raw_metrics.csv")
    pac_counts = Counter(row["scenario"] for row in pac)
    fixed_counts = Counter((row["controller"], row["scenario"]) for row in fixed)
    pac_keys = {(row["train_seed"], row["eval_episode"], row["scenario"]) for row in pac}
    fixed_keys = {(row["controller"], row["seed"], row["scenario"]) for row in fixed}
    conditions = {
        "five PAC training seeds": {int(row["train_seed"]) for row in pac} == {20, 21, 22, 23, 24},
        "ten PAC evaluation episodes": {int(row["eval_episode"]) for row in pac} == set(range(10)),
        "50 PAC rollouts per scenario": set(pac_counts.values()) == {50} and len(pac_counts) == 3,
        "10 fixed rollouts per controller/scenario": set(fixed_counts.values()) == {10} and len(fixed_counts) == 6,
        "five model checkpoints": len(list((ROOT / "results/formal_seeded_v2").glob("pac_train_seed_*/predictive_alpha_model.pt"))) == 5,
        "unique PAC episode keys": len(pac_keys) == len(pac),
        "unique fixed episode keys": len(fixed_keys) == len(fixed),
        "PAC simulator seed mapping": all(
            int(row["seed"]) == int(row["train_seed"]) * 1000 + int(row["eval_episode"])
            for row in pac
        ),
    }
    failures.extend(f"evidence check failed: {name}" for name, ok in conditions.items() if not ok)

    try:
        summary = summarize_formal_results(ROOT / "results" / "formal_seeded_v2")
        expected = {
            ("pac", "mean_rmse_3d"): 0.0902230450212123,
            ("pac", "pooled_sample_sd"): 0.00979868270936307,
            ("pac", "training_seed_mean_sd"): 0.00128272608284843,
            ("smc", "mean_rmse_3d"): 0.14374800885215,
            ("predictive", "mean_rmse_3d"): 0.145712776770692,
        }
        for (method, metric), value in expected.items():
            if abs(float(summary[method][metric]) - value) > 1.0e-12:
                failures.append(f"headline mismatch: {method}.{metric}")
    except Exception as exc:
        failures.append(f"formal summary failed: {exc}")

    for checkpoint in sorted((ROOT / "results" / "formal_seeded_v2").glob(
            "pac_train_seed_*/predictive_alpha_model.pt")):
        try:
            model, metadata = load_alpha_model_checkpoint(
                checkpoint,
                expected_feature_mode="state_phase",
            )
            if metadata["input_dim"] != 24 or metadata["history_len"] != 16:
                failures.append(f"checkpoint shape metadata mismatch: {checkpoint.name}")
            if sum(parameter.numel() for parameter in model.parameters()) != 14113:
                failures.append(f"checkpoint parameter count mismatch: {checkpoint.name}")
        except Exception as exc:
            failures.append(f"checkpoint load failed: {checkpoint}: {exc}")

    config = load_config(ROOT / "config" / "pac.yaml")
    overlap = set(config.sspo.search_seeds) & set(config.sspo.eval_seeds)
    if overlap != {20000, 20001, 20002}:
        failures.append("SSPO search/evaluation overlap provenance mismatch")
    if (ROOT / "results/formal_seeded_v2/timeseries/timeseries_3d.csv").exists():
        failures.append("derived merged PAC timeseries must not be stored")


def main() -> int:
    failures: list[str] = []
    manifest_path = ROOT / "MANIFEST.csv"
    sums_path = ROOT / "SHA256SUMS.txt"
    if not manifest_path.is_file():
        failures.append("MANIFEST.csv is missing")
    if not sums_path.is_file():
        failures.append("SHA256SUMS.txt is missing")
    if failures:
        print(json.dumps({"status": "FAIL", "failures": failures}, indent=2))
        return 1

    manifest_rows = read_manifest(manifest_path)
    manifest_paths = [row.get("path", "") for row in manifest_rows]
    listed_paths = set(manifest_paths)
    disk_paths = {path.relative_to(ROOT).as_posix() for path in repository_files(ROOT)}
    if len(manifest_paths) != len(listed_paths):
        failures.append("manifest contains duplicate paths")
    if listed_paths - disk_paths:
        failures.append(f"manifest files missing from disk: {sorted(listed_paths - disk_paths)}")
    if disk_paths - listed_paths:
        failures.append(f"disk payload absent from manifest: {sorted(disk_paths - listed_paths)}")

    for row in manifest_rows:
        relative = row["path"]
        path = ROOT / relative
        if not path.is_file():
            continue
        if str(path.stat().st_size) != row["bytes"]:
            failures.append(f"manifest byte mismatch: {relative}")
        if file_sha256(path) != row["sha256"]:
            failures.append(f"manifest sha256 mismatch: {relative}")

    expected_sums = {f"{row['sha256']}  {row['path']}" for row in manifest_rows}
    actual_sums = set(sums_path.read_text(encoding="utf-8").splitlines())
    if expected_sums != actual_sums or len(actual_sums) != len(manifest_rows):
        failures.append("SHA256SUMS.txt does not exactly match MANIFEST.csv")

    for relative in REQUIRED_FILES:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0 or relative not in listed_paths:
            failures.append(f"missing, empty, or unlisted required file: {relative}")

    all_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(
            part in {".git", ".venv", "dist", "runs", "build", "pa" + "per", "__" + "pycache__", ".pytest_cache"}
            or part.endswith(".egg-info")
            for part in path.relative_to(ROOT).parts
        )
    }
    forbidden_paths = sorted(
        path for path in all_paths
        if any(fragment in path.lower() for fragment in FORBIDDEN_PATH_FRAGMENTS)
        or Path(path).name.startswith(("figS1_", "figS2_", "figS3_", "figS4_", "~$"))
        or Path(path).suffix.lower() in {".pyc", ".aux", ".bbl", ".blg", ".log", ".tmp", ".lock", ".zip"}
    )
    if forbidden_paths:
        failures.append(f"forbidden repository paths: {forbidden_paths}")

    text_offenders = []
    for relative in sorted(disk_paths):
        path = ROOT / relative
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        if ABSOLUTE_PATH.search(text) or any(term in text.lower() for term in FORBIDDEN_PATH_FRAGMENTS):
            text_offenders.append(relative)
    if text_offenders:
        failures.append(f"non-portable or disallowed text: {text_offenders}")

    verify_evidence(failures)
    summary = {
        "repository_root": ROOT.name,
        "manifest_rows": len(manifest_rows),
        "sha256_rows": len(actual_sums),
        "checked_requirements": len(REQUIRED_FILES),
        "tested_environment": tested_environment(),
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
