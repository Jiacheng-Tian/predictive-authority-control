from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path

from scripts.update_release_manifest import repository_files

ROOT = Path(__file__).resolve().parents[1]
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RepositoryIntegrityTest(unittest.TestCase):
    def test_manifest_uses_lf_line_endings(self):
        self.assertNotIn(b"\r\n", (ROOT / "MANIFEST.csv").read_bytes())

    def test_manifest_uses_only_clone_stable_fields(self):
        with (ROOT / "MANIFEST.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, ["path", "bytes", "sha256"])

    def test_manifest_covers_exact_repository_payload(self):
        with (ROOT / "MANIFEST.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        entries = {row["path"]: row for row in rows}
        payload = {path.relative_to(ROOT).as_posix() for path in repository_files(ROOT)}

        self.assertEqual(set(entries), payload)
        for relative, row in entries.items():
            path = ROOT / relative
            self.assertEqual(int(row["bytes"]), path.stat().st_size)
            self.assertEqual(row["sha256"], sha256(path))

    def test_read_only_verifier_preserves_integrity_baselines(self):
        manifest_before = sha256(ROOT / "MANIFEST.csv")
        sums_before = sha256(ROOT / "SHA256SUMS.txt")

        completed = subprocess.run(
            [sys.executable, "scripts/verify_repository.py"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(sha256(ROOT / "MANIFEST.csv"), manifest_before)
        self.assertEqual(sha256(ROOT / "SHA256SUMS.txt"), sums_before)

    def test_formal_evidence_dimensions_are_complete(self):
        with (ROOT / "results/formal_seeded_v2/raw_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
            pac = list(csv.DictReader(handle))
        with (ROOT / "results/formal_seeded_v2/fixed_controllers/raw_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
            fixed = list(csv.DictReader(handle))
        pac_counts = Counter(row["scenario"] for row in pac)
        fixed_counts = Counter((row["controller"], row["scenario"]) for row in fixed)

        self.assertEqual({int(row["train_seed"]) for row in pac}, {20, 21, 22, 23, 24})
        self.assertEqual({int(row["eval_episode"]) for row in pac}, set(range(10)))
        self.assertEqual(set(pac_counts.values()), {50})
        self.assertEqual(len(pac_counts), 3)
        self.assertEqual(set(fixed_counts.values()), {10})
        self.assertEqual(len(fixed_counts), 6)
        self.assertEqual(len(list((ROOT / "results/formal_seeded_v2").glob("pac_train_seed_*/predictive_alpha_model.pt"))), 5)

    def test_derived_merged_timeseries_is_not_stored(self):
        self.assertFalse(
            (ROOT / "results/formal_seeded_v2/timeseries/timeseries_3d.csv").exists()
        )


if __name__ == "__main__":
    unittest.main()
