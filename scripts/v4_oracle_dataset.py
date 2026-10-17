#!/usr/bin/env python3
"""Generate an oracle-teacher dataset over v4 disturbance episodes (S0/S1)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import torch  # noqa: E402

torch.set_num_threads(1)

from pac.v4.config import load_v4_config  # noqa: E402
from pac.v4.teacher import (  # noqa: E402
    TeacherPlanItem,
    collect_v4_teacher_episode,
    save_teacher_dataset,
)

_WORKER: dict = {}


def _worker_init(config_path: str) -> None:
    _WORKER["config"] = load_v4_config(config_path)


def _worker_collect(item_dict: dict) -> dict:
    item = TeacherPlanItem(**item_dict)
    episode = collect_v4_teacher_episode(_WORKER["config"], item)
    # plan items are not picklable-through-dataclass issues; return plain data
    episode["plan_item"] = None
    return episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "pac_v4.yaml"))
    parser.add_argument("--plan", required=True,
                        help="JSON list of episodes: [{family, scenario_id, "
                             "seed, steps, difficulty, params|null}, ...]")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--jobs", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    plan_raw = json.loads(Path(arguments.plan).read_text(encoding="utf-8"))
    plan = [TeacherPlanItem(**entry) for entry in plan_raw]
    if arguments.jobs > 1:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with context.Pool(
                processes=int(arguments.jobs),
                initializer=_worker_init,
                initargs=(str(Path(arguments.config).resolve()),),
        ) as pool:
            episodes = pool.map(_worker_collect, [vars(item) for item in plan])
    else:
        _worker_init(str(Path(arguments.config).resolve()))
        episodes = [_worker_collect(vars(item)) for item in plan]
    config = load_v4_config(arguments.config)
    out = save_teacher_dataset(episodes, arguments.out_dir, config, plan)
    labels = __import__("numpy").concatenate(
        [e["labels"] for e in episodes])
    print(json.dumps({
        "out_dir": str(out),
        "episodes": len(plan),
        "rows": int(labels.shape[0]),
        "label_mean": float(labels.mean()),
        "low_alpha_fraction": float((labels <= 0.3).mean()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
