#!/usr/bin/env python3
"""Build the v4 world-model transition dataset.

Outputs live under runs/predictive_authority_v4/ and never touch v3 runs or
the archived results/ evidence tree.  The output directory must not exist or
must be empty.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pac.v4.collector import (  # noqa: E402
    _COLUMNS,
    TransitionArrays,
    _collect_single_plan,
    resolve_collection_plans,
    save_transition_dataset,
)
from pac.v4.config import load_v4_config  # noqa: E402


def _collect_one_plan(arguments: tuple[dict, str]) -> dict:
    """Process-pool worker: collect one episode in a fresh interpreter."""
    plan, config_path = arguments
    import torch

    torch.set_num_threads(1)
    config = load_v4_config(config_path)
    return _collect_single_plan(config, plan)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "pac_v4.yaml"))
    parser.add_argument("--profile", default="dry", choices=["dry", "short", "formal"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--jobs", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config_path = str(Path(arguments.config).resolve())
    config = load_v4_config(config_path)
    plans = resolve_collection_plans(config, arguments.profile)
    plan_summary = {
        "profile": arguments.profile,
        "episodes": len(plans),
        "steps_per_episode": sorted({int(plan["steps"]) for plan in plans}),
        "families": sorted({plan["family"] for plan in plans}),
        "splits": sorted({plan["split"] for plan in plans}),
    }
    if arguments.profile == "dry":
        print(json.dumps(plan_summary, sort_keys=True))
        return 0

    output = Path(arguments.out_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    started = time.perf_counter()
    jobs = max(1, int(arguments.jobs))

    features_chunks, state_chunks, next_chunks, physics_chunks = [], [], [], []
    requested_chunks, applied_chunks, est_chunks, true_chunks = [], [], [], []
    next_true_chunks, context_chunks, solver_chunks = [], [], []
    plan_chunks, alpha_chunks = [], []
    metadata_rows: list[dict] = []

    def accumulate(result: dict) -> None:
        features_chunks.append(result["features"])
        state_chunks.append(result["state"])
        next_chunks.append(result["next_state"])
        physics_chunks.append(result["physics_next_state"])
        requested_chunks.append(result["requested"])
        applied_chunks.append(result["applied"])
        est_chunks.append(result["est_current"])
        true_chunks.append(result["true_current"])
        next_true_chunks.append(result["next_true_current"])
        context_chunks.append(result["context"])
        solver_chunks.append(result["solver"])
        plan_chunks.append(result["plan"])
        alpha_chunks.append(result["alpha"])
        metadata_rows.extend(result["metadata"])

    if jobs == 1:
        for index, plan in enumerate(plans, start=1):
            episode_started = time.perf_counter()
            result = _collect_single_plan(config, plan)
            accumulate(result)
            print(
                f"[wm-collect] {index}/{len(plans)} episode={result['episode_uid']} "
                f"elapsed={time.perf_counter() - episode_started:.1f}s",
                file=sys.stderr,
                flush=True,
            )
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=jobs) as pool:
            for index, result in enumerate(
                pool.imap_unordered(
                    _collect_one_plan,
                    ((plan, config_path) for plan in plans),
                    chunksize=1,
                ),
                start=1,
            ):
                accumulate(result)
                print(
                    f"[wm-collect] {index}/{len(plans)} episode={result['episode_uid']}",
                    file=sys.stderr,
                    flush=True,
                )

    dataset = TransitionArrays(
        features=np.concatenate(features_chunks),
        state=np.concatenate(state_chunks),
        next_state=np.concatenate(next_chunks),
        physics_next_state=np.concatenate(physics_chunks),
        requested=np.concatenate(requested_chunks),
        applied=np.concatenate(applied_chunks),
        est_current=np.concatenate(est_chunks),
        true_current=np.concatenate(true_chunks),
        next_true_current=np.concatenate(next_true_chunks),
        context=np.concatenate(context_chunks),
        solver=np.concatenate(solver_chunks),
        plan=np.concatenate(plan_chunks),
        alpha=np.concatenate(alpha_chunks),
        metadata=pd.DataFrame(metadata_rows, columns=_COLUMNS),
    )
    manifest = save_transition_dataset(
        dataset,
        output,
        provenance={
            "protocol_version": config.protocol.version,
            "profile": arguments.profile,
            "seed_partitions": {
                "wm_train": list(config.seeds.wm_train),
                "wm_val": list(config.seeds.wm_val),
                "wm_test": list(config.seeds.wm_test),
            },
            "episodes": len(plans),
        },
    )
    print(json.dumps({
        "profile": arguments.profile,
        "out_dir": str(output),
        "sample_count": manifest["sample_count"],
        "episodes": len(plans),
        "elapsed_s": time.perf_counter() - started,
        "file_sha256": manifest["file_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
