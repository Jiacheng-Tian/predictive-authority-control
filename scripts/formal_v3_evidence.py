#!/usr/bin/env python3
"""Promote, verify, and repair portable formal v3 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pac.evaluation.evidence_v3 import (
    promote_evidence,
    repair_portability,
    verify_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    promote = commands.add_parser("promote")
    promote.add_argument("--source-dir", required=True)
    promote.add_argument("--target-dir", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--evidence-dir", required=True)
    repair = commands.add_parser("repair-portability")
    repair.add_argument("--evidence-dir", required=True)
    for command in (promote, verify, repair):
        command.add_argument("--repo-root", default=str(ROOT))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "promote":
            result = promote_evidence(
                args.source_dir,
                args.target_dir,
                repo_root=args.repo_root,
            )
        elif args.command == "verify":
            result = verify_evidence(args.evidence_dir, repo_root=args.repo_root)
        else:
            result = repair_portability(args.evidence_dir, repo_root=args.repo_root)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
