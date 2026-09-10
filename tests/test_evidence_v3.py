from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from pac.evaluation.evidence_v3 import (
    promote_evidence,
    repair_portability,
    verify_evidence,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_formal_fixture(root: Path) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "fixed_controllers").mkdir()
    (root / "raw_metrics.csv").write_text(
        "method,scenario_id,seed,rmse_3d\n"
        "PAC,1,20,0.10\n",
        encoding="ascii",
    )
    (root / "window_metrics.csv").write_text(
        "method,scenario_id,seed,window,rmse_3d\n"
        "PAC,1,20,all,0.10\n",
        encoding="ascii",
    )
    (root / "fixed_controllers" / "raw_metrics.csv").write_text(
        "controller,scenario_id,seed,rmse_3d\n"
        "SMC,1,20,0.20\n",
        encoding="ascii",
    )
    _write_json(
        root / "manifest.json",
        {
            "schema_version": "formal_v3",
            "protocol": {
                "name": "formal_v3",
                "output_directory": "results/formal_v3",
            },
            "model": {"policy_architecture": "transformer"},
        },
    )
    hashes = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "promotion_manifest.json"
    }
    _write_json(
        root / "promotion_manifest.json",
        {
            "schema_version": "formal_v3_promotion_v1",
            "evidence_directory": "results/formal_v3",
            "artifact_sha256": hashes,
        },
    )
    return hashes


class EvidenceV3Test(unittest.TestCase):
    def test_verify_rejects_absolute_paths_in_evidence_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            evidence = repo / "results" / "formal_v3"
            _write_formal_fixture(evidence)
            manifest = json.loads((evidence / "manifest.json").read_text())
            manifest["protocol"]["output_directory"] = str(repo / "runs" / "formal_v3")
            _write_json(evidence / "manifest.json", manifest)

            with self.assertRaisesRegex(ValueError, "absolute path"):
                verify_evidence(evidence, repo_root=repo)

    def test_verify_rejects_parent_traversal_in_evidence_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            evidence = repo / "results" / "formal_v3"
            _write_formal_fixture(evidence)
            manifest = json.loads((evidence / "manifest.json").read_text())
            manifest["protocol"]["output_directory"] = "results/formal_v3/../outside"
            _write_json(evidence / "manifest.json", manifest)

            with self.assertRaisesRegex(ValueError, "parent traversal"):
                verify_evidence(evidence, repo_root=repo)

    def test_verify_rejects_legacy_formal_seeded_v2_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            evidence = repo / "results" / "formal_v3"
            _write_formal_fixture(evidence)
            promotion = json.loads(
                (evidence / "promotion_manifest.json").read_text()
            )
            promotion["evidence_directory"] = "results/formal_seeded_v2"
            _write_json(evidence / "promotion_manifest.json", promotion)

            with self.assertRaisesRegex(ValueError, "formal_seeded_v2"):
                verify_evidence(evidence, repo_root=repo)

    def test_verify_detects_payload_tampering_from_promotion_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            evidence = repo / "results" / "formal_v3"
            _write_formal_fixture(evidence)
            (evidence / "raw_metrics.csv").write_text(
                "method,scenario_id,seed,rmse_3d\nPAC,1,20,9.99\n",
                encoding="ascii",
            )

            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                verify_evidence(evidence, repo_root=repo)

    def test_repair_portability_rewrites_only_manifests_and_preserves_payload_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            evidence = repo / "results" / "formal_v3"
            payload_hashes = _write_formal_fixture(evidence)
            manifest = json.loads((evidence / "manifest.json").read_text())
            manifest["protocol"]["output_directory"] = str(evidence)
            _write_json(evidence / "manifest.json", manifest)
            promotion = json.loads(
                (evidence / "promotion_manifest.json").read_text()
            )
            promotion["evidence_directory"] = str(evidence)
            promotion["artifact_sha256"] = {
                str(evidence / relative): digest
                for relative, digest in payload_hashes.items()
            }
            _write_json(evidence / "promotion_manifest.json", promotion)
            before_payload = {
                relative: (evidence / relative).read_bytes()
                for relative in payload_hashes
            }

            result = repair_portability(evidence, repo_root=repo)

            self.assertTrue(result["ok"])
            repaired_manifest = json.loads((evidence / "manifest.json").read_text())
            repaired_promotion = json.loads(
                (evidence / "promotion_manifest.json").read_text()
            )
            self.assertEqual(
                repaired_manifest["protocol"]["output_directory"],
                "results/formal_v3",
            )
            self.assertEqual(
                set(repaired_promotion["artifact_sha256"]),
                set(payload_hashes),
            )
            self.assertEqual(
                repaired_promotion["artifact_sha256"]["manifest.json"],
                _sha256(evidence / "manifest.json"),
            )
            for relative, digest in payload_hashes.items():
                if relative == "manifest.json":
                    continue
                self.assertEqual(_sha256(evidence / relative), digest)
                self.assertEqual((evidence / relative).read_bytes(), before_payload[relative])
            self.assertTrue(verify_evidence(evidence, repo_root=repo)["ok"])
            self.assertEqual(
                {path.name for path in evidence.iterdir()},
                {"manifest.json", "promotion_manifest.json", "raw_metrics.csv", "window_metrics.csv", "fixed_controllers"},
            )

    def test_promote_rejects_nonempty_target_and_paths_outside_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            source = repo / "staging" / "formal_v3"
            _write_formal_fixture(source)

            nonempty = repo / "results" / "formal_v3"
            nonempty.mkdir(parents=True)
            (nonempty / "keep.txt").write_text("keep", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "not empty"):
                promote_evidence(source, nonempty, repo_root=repo)
            self.assertEqual((nonempty / "keep.txt").read_text(), "keep")

            outside = Path(tmp) / "outside" / "formal_v3"
            with self.assertRaisesRegex(ValueError, "outside repository root"):
                promote_evidence(source, outside, repo_root=repo)

    def test_cli_verify_emits_json(self):
        from scripts.formal_v3_evidence import main

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            evidence = repo / "results" / "formal_v3"
            _write_formal_fixture(evidence)
            output = io.StringIO()
            with redirect_stdout(output):
                code = main([
                    "verify",
                    "--evidence-dir",
                    str(evidence),
                    "--repo-root",
                    str(repo),
                ])

            self.assertEqual(code, 0)
            self.assertTrue(json.loads(output.getvalue())["ok"])

    def test_repository_verifier_validates_each_v3_evidence_directory(self):
        import scripts.verify_repository as repository_verifier

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_formal_fixture(root / "results" / "formal_v3" / "formal-a")
            failures: list[str] = []
            with patch.object(repository_verifier, "ROOT", root):
                repository_verifier.verify_v3_evidence(failures)

            self.assertEqual(failures, [])
            self.assertIn(
                "scripts/formal_v3_evidence.py",
                repository_verifier.REQUIRED_FILES,
            )


if __name__ == "__main__":
    unittest.main()
