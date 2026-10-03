"""Portable promotion and verification helpers for formal v3 evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any


_ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\\\/]|/|\\\\)")
_METADATA_FILES = ("manifest.json", "promotion_manifest.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _inside(path: Path, root: Path, *, message: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(message) from exc


def _is_absolute(value: str) -> bool:
    return bool(_ABSOLUTE_PATH.match(value))


def _validate_path_string(value: str) -> None:
    if "legacy_evidence" in value.lower():
        raise ValueError("legacy_evidence references are forbidden in v3 evidence")
    if _is_absolute(value):
        raise ValueError(f"absolute path is forbidden in v3 evidence: {value}")
    if ".." in value.replace("\\", "/").split("/"):
        raise ValueError(f"parent traversal is forbidden in v3 evidence: {value}")


def _walk_strings(value: Any) -> None:
    if isinstance(value, str):
        _validate_path_string(value)
    elif isinstance(value, list):
        for item in value:
            _walk_strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                _validate_path_string(key)
            _walk_strings(item)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid evidence metadata: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"evidence metadata must be an object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _artifact_hashes(evidence_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(evidence_dir).as_posix(): _sha256(path)
        for path in sorted(evidence_dir.rglob("*"))
        if path.is_file() and path.name != "promotion_manifest.json"
    }


def _portable_value(value: Any, repo_root: Path) -> Any:
    if isinstance(value, str):
        if _is_absolute(value):
            resolved = _resolve(value)
            _inside(resolved, repo_root, message=f"path is outside repository root: {value}")
            return resolved.relative_to(repo_root).as_posix()
        _validate_path_string(value)
        return value.replace("\\", "/")
    if isinstance(value, list):
        return [_portable_value(item, repo_root) for item in value]
    if isinstance(value, dict):
        return {key: _portable_value(item, repo_root) for key, item in value.items()}
    return value


def _portable_artifact_hashes(value: Any, evidence_dir: Path) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("artifact_sha256 must be an object")
    normalized: dict[str, str] = {}
    for key, digest in value.items():
        if not isinstance(key, str) or not isinstance(digest, str):
            raise ValueError("artifact_sha256 entries must be string pairs")
        path = _resolve(key) if _is_absolute(key) else _resolve(evidence_dir / key)
        _inside(path, evidence_dir, message=f"artifact path is outside evidence directory: {key}")
        normalized[path.relative_to(evidence_dir).as_posix()] = digest
    return normalized


def verify_evidence(evidence_dir: Path | str, *, repo_root: Path | str) -> dict[str, Any]:
    """Validate portable metadata and promotion hashes without mutating evidence."""
    root = _resolve(repo_root)
    evidence = _resolve(evidence_dir)
    _inside(evidence, root, message=f"evidence directory is outside repository root: {evidence}")
    if "legacy_evidence" in evidence.as_posix().lower():
        raise ValueError("legacy_evidence cannot be validated as v3 evidence")

    for name in _METADATA_FILES:
        if not (evidence / name).is_file():
            raise ValueError(f"missing evidence metadata: {name}")
    manifest = _read_json(evidence / "manifest.json")
    promotion = _read_json(evidence / "promotion_manifest.json")
    _walk_strings(manifest)
    _walk_strings(promotion)

    expected = promotion.get("artifact_sha256")
    if not isinstance(expected, dict):
        raise ValueError("promotion manifest is missing artifact_sha256")
    actual = _artifact_hashes(evidence)
    expected_normalized = _portable_artifact_hashes(expected, evidence)
    if set(expected_normalized) != set(actual):
        raise ValueError("artifact_sha256 paths do not match evidence payload")
    for relative, digest in expected_normalized.items():
        if digest != actual[relative]:
            raise ValueError(f"sha256 mismatch for {relative}")

    return {
        "ok": True,
        "evidence_dir": evidence.relative_to(root).as_posix(),
        "artifact_count": len(actual),
        "protocol_version": promotion.get("protocol_version", manifest.get("protocol_version")),
    }


def repair_portability(evidence_dir: Path | str, *, repo_root: Path | str) -> dict[str, Any]:
    """Atomically repair only v3 metadata files and refresh promotion hashes."""
    root = _resolve(repo_root)
    evidence = _resolve(evidence_dir)
    _inside(evidence, root, message=f"evidence directory is outside repository root: {evidence}")
    if "legacy_evidence" in evidence.as_posix().lower():
        raise ValueError("legacy_evidence cannot be repaired as v3 evidence")

    manifest_path = evidence / "manifest.json"
    promotion_path = evidence / "promotion_manifest.json"
    manifest = _portable_value(_read_json(manifest_path), root)
    promotion = _portable_value(_read_json(promotion_path), root)
    promotion["artifact_sha256"] = _portable_artifact_hashes(
        promotion.get("artifact_sha256"), evidence
    )

    _atomic_json(manifest_path, manifest)
    promotion["artifact_sha256"] = _artifact_hashes(evidence)
    _atomic_json(promotion_path, promotion)
    return verify_evidence(evidence, repo_root=root)


def promote_evidence(
    source_dir: Path | str,
    target_dir: Path | str,
    *,
    repo_root: Path | str,
) -> dict[str, Any]:
    """Copy a candidate evidence tree into an empty repository-contained target."""
    root = _resolve(repo_root)
    source = _resolve(source_dir)
    target = _resolve(target_dir)
    _inside(source, root, message=f"source directory is outside repository root: {source}")
    _inside(target, root, message=f"target directory is outside repository root: {target}")
    if "legacy_evidence" in source.as_posix().lower() or "legacy_evidence" in target.as_posix().lower():
        raise ValueError("legacy_evidence cannot be used for v3 promotion")
    if not source.is_dir():
        raise ValueError(f"source evidence directory is missing: {source}")
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"target evidence directory is not empty: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.rmdir()
    shutil.copytree(source, target)
    return repair_portability(target, repo_root=root)
