from __future__ import annotations

import unittest
from pathlib import Path


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
        self.assertTrue(reproducibility.is_file())
        self.assertIn("Predictive Authority Control", readme.read_text(encoding="utf-8"))

    def test_repository_has_no_publication_or_conference_assets(self):
        offenders = []
        for path in ROOT.rglob("*"):
            relative_path = path.relative_to(ROOT)
            if any(part in {".git", ".venv"} for part in relative_path.parts):
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
            if any(part in {".git", ".venv"} for part in relative_path.parts):
                continue
            if any(part in prohibited_names for part in relative_path.parts):
                offenders.append(relative_path.as_posix())
            elif path.is_file() and path.suffix.lower() in prohibited_suffixes:
                offenders.append(relative_path.as_posix())

        self.assertEqual(sorted(offenders), [])


if __name__ == "__main__":
    unittest.main()
