"""Verify PAC repository contents without rewriting integrity baselines."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.update_release_manifest import repository_files


REQUIRED_FILES = [
    "README.md",
    "REPRODUCIBILITY.md",
    ".gitignore",
    ".gitattributes",
    "code/requirements.txt",
    "code/configs/config.yaml",
    "code/controllers/smc.py",
    "code/controllers/mpc.py",
    "code/env/auv_env.py",
    "code/env/dynamics.py",
    "code/env/thrusters.py",
    "code/env/vehicle_profiles.py",
    "code/evaluation/baseline_presets.py",
    "code/evaluation/current_control.py",
    "code/evaluation/diagnose_3d_authority.py",
    "code/evaluation/engineering_metrics.py",
    "code/evaluation/predictive_3d_authority_alpha.py",
    "scripts/run_formal_seeded_protocol.py",
    "scripts/sspo_alpha_calibration.py",
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
    "pa" + "per",
    "manu" + "script",
    "re" + "viewer",
    "sub" + "mission",
    "ic" + "ra",
}
TEXT_SUFFIXES = {".json", ".md", ".py", ".txt", ".yaml", ".yml"}
ABSOLUTE_PATH = re.compile(r"[A-Za-z]:[\\/]")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def tested_environment() -> dict[str, object]:
    packages = {}
    for name in ["torch", "gymnasium", "numpy", "matplotlib", "pandas"]:
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
    conditions = {
        "five PAC training seeds": {int(row["train_seed"]) for row in pac} == {20, 21, 22, 23, 24},
        "ten PAC evaluation episodes": {int(row["eval_episode"]) for row in pac} == set(range(10)),
        "50 PAC rollouts per scenario": set(pac_counts.values()) == {50} and len(pac_counts) == 3,
        "10 fixed rollouts per controller/scenario": set(fixed_counts.values()) == {10} and len(fixed_counts) == 6,
        "five model checkpoints": len(list((ROOT / "results/formal_seeded_v2").glob("pac_train_seed_*/predictive_alpha_model.pt"))) == 5,
    }
    failures.extend(f"evidence check failed: {name}" for name, ok in conditions.items() if not ok)


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
        if path.is_file() and not any(part in {".git", ".venv", "dist"} for part in path.relative_to(ROOT).parts)
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
