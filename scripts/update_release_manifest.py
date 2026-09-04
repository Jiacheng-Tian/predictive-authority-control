"""Generate integrity baselines for the PAC reproducibility repository."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATED_NAMES = {
    "MANIFEST.csv",
    "SHA256SUMS.txt",
    "REPOSITORY_VERIFICATION.json",
    "REPOSITORY_VERIFICATION_CHECKS.csv",
}
RUNTIME_DIRS = {".git", ".venv", "__" + "pycache__", ".pytest_cache", "dist"}
RUNTIME_SUFFIXES = {".pyc", ".tmp", ".log", ".lock"}


def repository_files(root: Path = ROOT) -> list[Path]:
    """Return every tracked payload file, excluding local runtime artifacts."""
    root = Path(root)
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name in GENERATED_NAMES:
            continue
        relative = path.relative_to(root)
        if any(part in RUNTIME_DIRS for part in relative.parts):
            continue
        if path.suffix.lower() in RUNTIME_SUFFIXES:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix().lower())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_rows(root: Path = ROOT) -> list[dict[str, str]]:
    rows = []
    for path in repository_files(root):
        stat = path.stat()
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": str(stat.st_size),
            "sha256": file_sha256(path),
        })
    return rows


def write_manifests(root: Path = ROOT) -> tuple[Path, Path, int]:
    root = Path(root)
    rows = build_rows(root)
    manifest_path = root / "MANIFEST.csv"
    sums_path = root / "SHA256SUMS.txt"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)
    with sums_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(f"{row['sha256']}  {row['path']}\n")
    return manifest_path, sums_path, len(rows)


def main() -> int:
    manifest_path, sums_path, count = write_manifests()
    print(f"Wrote {manifest_path.name} with {count} payload rows")
    print(f"Wrote {sums_path.name} with {count} payload rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
