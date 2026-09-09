"""Print a read-only JSON summary of archived formal PAC results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pac.config import load_config
from pac.evaluation.summary import summarize_formal_results


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "pac.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--results-dir", default="")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    results_dir = Path(args.results_dir) if args.results_dir else ROOT / config.outputs.formal_evidence_dir
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    print(json.dumps(summarize_formal_results(results_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
