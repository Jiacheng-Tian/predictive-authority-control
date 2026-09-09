"""Validated, content-addressed storage for oracle training samples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd


METADATA_COLUMNS = (
    "split",
    "scenario",
    "environment_seed",
    "episode_uid",
    "step",
    "sample_time",
    "teacher_alpha",
    "oracle_best",
    "oracle_worst",
    "episode_fingerprint",
)


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        return {
            "__ndarray__": True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("provenance values must be finite")
        return value
    if hasattr(value, "__dict__"):
        return _json_value(vars(value))
    return str(value)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _validate_metadata(metadata: pd.DataFrame, length: int) -> pd.DataFrame:
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame")
    if len(metadata) != length:
        raise ValueError("metadata length must match features and labels")
    if metadata.columns.has_duplicates:
        raise ValueError("metadata columns must be unique")
    missing = [column for column in METADATA_COLUMNS if column not in metadata.columns]
    if missing:
        raise ValueError(f"metadata is missing required columns: {', '.join(missing)}")
    if metadata.isna().any().any():
        raise ValueError("metadata must not contain null values")
    split_values = set(metadata["split"].tolist())
    if not split_values.issubset({"train", "val"}):
        raise ValueError("metadata split values must be 'train' or 'val'")
    for column in ("episode_uid", "episode_fingerprint"):
        values = metadata[column].tolist()
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"metadata {column} values must be non-empty strings")
    for column in metadata.columns:
        values = metadata[column]
        if pd.api.types.is_numeric_dtype(values):
            numeric = values.to_numpy(dtype=float)
            if not np.isfinite(numeric).all():
                raise ValueError(f"metadata column {column} must contain finite values")
    return metadata.copy(deep=True)


@dataclass(frozen=True, slots=True)
class OracleDataset:
    """A strict float32 feature/label table with episode provenance."""

    features: np.ndarray
    labels: np.ndarray
    metadata: pd.DataFrame

    def __post_init__(self) -> None:
        try:
            features = np.asarray(self.features, dtype=np.float32)
            labels = np.asarray(self.labels, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("features and labels must be numeric") from exc
        if features.ndim != 2 or features.shape[1] != 24:
            raise ValueError("features must have shape (N, 24)")
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError("labels must have shape (N,) matching features")
        if not np.isfinite(features).all() or not np.isfinite(labels).all():
            raise ValueError("features and labels must contain only finite values")
        features = np.array(features, dtype=np.float32, copy=True, order="C")
        labels = np.array(labels, dtype=np.float32, copy=True, order="C")
        features.setflags(write=False)
        labels.setflags(write=False)
        metadata = _validate_metadata(self.metadata, features.shape[0])
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "metadata", metadata)

    @property
    def sample_count(self) -> int:
        return int(self.labels.shape[0])

    def __iter__(self):
        """Compatibility unpacking for callers of the archived collector."""
        yield self.features
        yield self.labels
        yield self.metadata


def episode_fingerprint(spec: Any) -> str:
    """Return a stable SHA-256 fingerprint for one realized episode spec."""
    if isinstance(spec, dict):
        payload = spec
    elif is_dataclass(spec):
        payload = {field.name: getattr(spec, field.name) for field in fields(spec)}
    elif hasattr(spec, "__dict__"):
        payload = vars(spec)
    else:
        raise TypeError("episode spec must be a mapping, dataclass, or object")
    digest = hashlib.sha256()
    digest.update(b"pac-episode-fingerprint-v1\0")
    digest.update(_canonical_json(payload))
    return digest.hexdigest()


def assert_unique_episode_fingerprints(values: pd.DataFrame | Iterable[Any]) -> None:
    """Reject duplicate or missing episode fingerprints."""
    if isinstance(values, pd.DataFrame):
        if "episode_fingerprint" not in values.columns:
            raise ValueError("metadata must contain episode_fingerprint")
        fingerprints = values["episode_fingerprint"].tolist()
        if any(not isinstance(value, str) or not value for value in fingerprints):
            raise ValueError("episode fingerprints must be non-empty strings")
        if "episode_uid" in values.columns:
            mapping = values.groupby("episode_uid", sort=False)["episode_fingerprint"].nunique()
            if bool((mapping > 1).any()):
                raise ValueError("one episode_uid maps to multiple fingerprints")
            reverse = values.groupby("episode_fingerprint", sort=False)["episode_uid"].nunique()
            if bool((reverse > 1).any()):
                raise ValueError("one episode fingerprint maps to multiple episode_uids")
            return
        # Without an episode key, each row is treated as one episode record.
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("episode fingerprints must be unique")
        return
    else:
        fingerprints = list(values)
    if any(not isinstance(value, str) or not value for value in fingerprints):
        raise ValueError("episode fingerprints must be non-empty strings")
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("episode fingerprints must be unique")


def assert_episode_split_disjoint(metadata: pd.DataFrame) -> None:
    """Ensure every episode belongs to exactly one train/validation split."""
    _validate_metadata(metadata, len(metadata))
    for key in ("environment_seed", "episode_uid"):
        grouped = metadata.groupby(key, sort=False)["split"].nunique()
        if bool((grouped > 1).any()):
            raise ValueError(f"an episode {key} cannot occur in multiple splits")
    split_values = set(metadata["split"].tolist())
    if not {"train", "val"}.issubset(split_values):
        raise ValueError("metadata must contain train and val splits")


def split_indices_by_episode(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return train/validation row indices grouped by episode, never samples."""
    assert_episode_split_disjoint(metadata)
    result: dict[str, np.ndarray] = {}
    for split in ("train", "val"):
        result[split] = np.flatnonzero(metadata["split"].to_numpy() == split)
    return result


def _canonical_metadata(metadata: pd.DataFrame) -> bytes:
    columns = sorted(str(column) for column in metadata.columns)
    records = []
    for _, row in metadata.loc[:, columns].iterrows():
        records.append(_canonical_json({column: row[column] for column in columns}).decode("utf-8"))
    records.sort()
    return _canonical_json({"columns": columns, "rows": records})


def canonical_dataset_hash(dataset: OracleDataset, provenance: Any | None = None) -> str:
    """Hash arrays, sorted metadata content, and oracle/config provenance."""
    dataset = dataset if isinstance(dataset, OracleDataset) else OracleDataset(**dataset)
    digest = hashlib.sha256()
    digest.update(b"pac-oracle-dataset-v1\0")
    for array in (dataset.features, dataset.labels):
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(_canonical_json(list(array.shape)))
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(_canonical_metadata(dataset.metadata))
    digest.update(_canonical_json({"provenance": provenance or {}}))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[3]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _default_dependency_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "pandas", "scipy", "osqp", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def save_oracle_dataset(
    dataset: OracleDataset,
    out_dir: str | Path,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically save a dataset, refusing to overwrite non-empty output."""
    if not isinstance(dataset, OracleDataset):
        dataset = OracleDataset(**dataset)
    assert_episode_split_disjoint(dataset.metadata)
    assert_unique_episode_fingerprints(dataset.metadata)
    target = Path(out_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"output directory is not empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source_provenance = dict(provenance or {})
    dataset_hash = canonical_dataset_hash(dataset, source_provenance)
    config_hash = source_provenance.get("config_hash")
    if config_hash is None and "config" in source_provenance:
        config_hash = hashlib.sha256(_canonical_json(source_provenance["config"])).hexdigest()
    manifest = {
        "schema_version": 1,
        "feature_dim": 24,
        "sample_count": dataset.sample_count,
        "dataset_hash": dataset_hash,
        "config_hash": config_hash,
        "git_commit": source_provenance.get("git_commit", _default_git_commit()),
        "dependency_versions": source_provenance.get(
            "dependency_versions", _default_dependency_versions()
        ),
        "seed_partitions": source_provenance.get("seed_partitions", {}),
        "provenance": source_provenance,
    }
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent)))
    try:
        np.save(temp_dir / "features.npy", dataset.features, allow_pickle=False)
        np.save(temp_dir / "labels.npy", dataset.labels, allow_pickle=False)
        dataset.metadata.to_csv(
            temp_dir / "metadata.csv", index=False, float_format="%.17g"
        )
        # Hash exactly what a later load observes on disk, while retaining the
        # original NPY bytes and dtypes.
        disk_metadata = pd.read_csv(temp_dir / "metadata.csv")
        disk_dataset = OracleDataset(dataset.features, dataset.labels, disk_metadata)
        assert_episode_split_disjoint(disk_dataset.metadata)
        assert_unique_episode_fingerprints(disk_dataset.metadata)
        dataset_hash = canonical_dataset_hash(disk_dataset, source_provenance)
        manifest["dataset_hash"] = dataset_hash
        file_hashes = {
            name: _file_sha256(temp_dir / name)
            for name in ("features.npy", "labels.npy", "metadata.csv")
        }
        manifest["files"] = {
            name: {"sha256": digest} for name, digest in file_hashes.items()
        }
        manifest["file_sha256"] = file_hashes
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8"
        )
        if target.exists():
            target.rmdir()
        temp_dir.replace(target)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return manifest


def load_oracle_dataset(out_dir: str | Path) -> OracleDataset:
    """Load and verify all file and canonical content hashes."""
    root = Path(out_dir)
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid oracle dataset manifest: {manifest_path}") from exc
    files = manifest.get("files", {})
    for name in ("features.npy", "labels.npy", "metadata.csv"):
        path = root / name
        if not path.is_file():
            raise ValueError(f"missing oracle dataset file: {name}")
        expected = files.get(name, {}).get("sha256")
        if expected is None:
            expected = manifest.get("file_sha256", {}).get(name)
        if not expected or _file_sha256(path) != expected:
            raise ValueError(f"oracle dataset file hash mismatch: {name}")
    try:
        dataset = OracleDataset(
            np.load(root / "features.npy", allow_pickle=False),
            np.load(root / "labels.npy", allow_pickle=False),
            pd.read_csv(root / "metadata.csv"),
        )
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("invalid oracle dataset content") from exc
    assert_episode_split_disjoint(dataset.metadata)
    assert_unique_episode_fingerprints(dataset.metadata)
    expected_dataset_hash = manifest.get("dataset_hash")
    actual_dataset_hash = canonical_dataset_hash(dataset, manifest.get("provenance", {}))
    if expected_dataset_hash != actual_dataset_hash:
        raise ValueError("oracle dataset canonical hash mismatch")
    if int(manifest.get("sample_count", -1)) != dataset.sample_count:
        raise ValueError("oracle dataset sample count mismatch")
    return dataset


__all__ = [
    "METADATA_COLUMNS",
    "OracleDataset",
    "assert_episode_split_disjoint",
    "assert_unique_episode_fingerprints",
    "canonical_dataset_hash",
    "episode_fingerprint",
    "load_oracle_dataset",
    "save_oracle_dataset",
    "split_indices_by_episode",
]
