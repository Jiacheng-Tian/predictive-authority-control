from __future__ import annotations

import unittest
import subprocess
from pathlib import Path

from scripts.update_release_manifest import repository_files


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".json", ".md", ".py", ".txt", ".yaml", ".yml"}
FORBIDDEN_TERMS = (
    "pa" + "per",
    "manu" + "script",
    "re" + "viewer",
    "sub" + "mission",
    "ic" + "ra",
)


class RepositoryIdentityTest(unittest.TestCase):
    def test_repository_is_named_and_documented_as_predictive_authority_control(self):
        readme = ROOT / "README.md"
        reproducibility = ROOT / "REPRODUCIBILITY.md"

        self.assertTrue(readme.is_file())
        self.assertFalse(reproducibility.exists())
        text = readme.read_text(encoding="utf-8")
        self.assertIn("Predictive Authority Control", text)
        self.assertIn("Git LFS", text)
        self.assertIn("git lfs pull", text)
        self.assertIn("0.0902 +/- 0.0098 m", text)
        self.assertIn("0.07417 m", text)
        self.assertIn("17.29%", text)
        for phrase in (
            "explor" + "atory",
            "mechanism" + " study",
            "not a strictly " + "held" + "-out",
        ):
            self.assertNotIn(phrase, text.lower())

    def test_runtime_uses_only_the_installable_pac_package(self):
        legacy_sources = []
        if (ROOT / "code").exists():
            legacy_sources = [
                path.relative_to(ROOT).as_posix()
                for path in (ROOT / "code").rglob("*")
                if path.suffix.lower() in {".py", ".yaml", ".yml"}
            ]
        self.assertEqual(legacy_sources, [])
        source_files = list((ROOT / "src" / "pac").rglob("*.py")) + list((ROOT / "scripts").glob("*.py"))
        forbidden = (
            "Alpha" + "MLP",
            "startup_" + "handover",
            "window_" + "oracle",
            "schedule_" + "oracle",
            "hybrid_" + "oracle",
            "mc_" + "dropout",
            "MPC" + "Controller",
            "sys.path" + ".insert",
        )
        offenders = []
        private_imports = []
        for path in source_files:
            text = path.read_text(encoding="utf-8")
            if any(term in text for term in forbidden):
                offenders.append(path.relative_to(ROOT).as_posix())
            if "from pac." in text and " import _" in text:
                private_imports.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(offenders, [])
        self.assertEqual(private_imports, [])

    def test_repository_has_no_publication_or_conference_assets(self):
        offenders = []
        for path in ROOT.rglob("*"):
            relative_path = path.relative_to(ROOT)
            if any(part in {".git", ".venv", "pa" + "per", "__" + "pycache__", ".pytest_cache"} for part in relative_path.parts):
                continue
            relative = relative_path.as_posix().lower()
            if any(term in relative for term in FORBIDDEN_TERMS):
                offenders.append(relative)
                continue
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8")
            if any(term in text.lower() for term in FORBIDDEN_TERMS):
                offenders.append(relative)

        self.assertEqual(sorted(offenders), [])

    def test_repository_has_no_archives_caches_or_latex_intermediates(self):
        prohibited_names = {"__" + "pycache__", ".pytest_cache"}
        prohibited_suffixes = {
            "." + "zip", ".pyc", ".aux", ".bbl", ".blg", ".log", ".tmp", ".lock",
        }
        offenders = []
        for path in ROOT.rglob("*"):
            relative_path = path.relative_to(ROOT)
            if any(part in {".git", ".venv", "pa" + "per", "__" + "pycache__", ".pytest_cache"} for part in relative_path.parts):
                continue
            if any(part in prohibited_names for part in relative_path.parts):
                offenders.append(relative_path.as_posix())
            elif path.is_file() and path.suffix.lower() in prohibited_suffixes:
                offenders.append(relative_path.as_posix())

        self.assertEqual(sorted(offenders), [])

    def test_local_authoring_workspace_is_ignored_by_release_inventory(self):
        relative = ("pa" + "per") + "/example.tex"
        completed = subprocess.run(
            ["git", "check-ignore", "-q", relative],
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertNotIn(ROOT / relative, set(repository_files(ROOT)))


if __name__ == "__main__":
    unittest.main()
