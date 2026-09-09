"""Repository payload enumeration and hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path


GENERATED_NAMES = {
    "MANIFEST.csv",
    "SHA256SUMS.txt",
    "REPOSITORY_VERIFICATION.json",
    "REPOSITORY_VERIFICATION_CHECKS.csv",
}
RUNTIME_DIRS = {
    ".git",
    ".venv",
    "__" + "pycache__",
    ".pytest_cache",
    "dist",
    "runs",
    "build",
    "pa" + "per",
}
RUNTIME_SUFFIXES = {".pyc", ".tmp", ".log", ".lock"}


def repository_files(root: Path) -> list[Path]:
    """Return public payload files while excluding ignored local artifacts."""
    root = Path(root)
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name in GENERATED_NAMES:
            continue
        relative = path.relative_to(root)
        if any(
                part in RUNTIME_DIRS or part.endswith(".egg-info")
                for part in relative.parts):
            continue
        if path.suffix.lower() in RUNTIME_SUFFIXES:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix().lower())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
