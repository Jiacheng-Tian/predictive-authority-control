"""Tests for v4 transition collection and candidate-alpha ranking."""

from __future__ import annotations

import os

# The collector tests run the online MPC with a wall-time deadline; keep BLAS
# single-threaded so plan acceptance does not depend on machine load.
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

from pac.residual.collector import (
    FEATURE_DIM,
    PLAN_HORIZON,
    TransitionArrays,
    _collect_single_plan,
    load_transition_dataset,
    resolve_collection_plans,
    save_transition_dataset,
)
from pac.residual.config import load_residual_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "pac_v4.yaml"


def _plan(**overrides) -> dict:
    values = {
        "split": "train",
        "seed": 21000,
        "family": "structured",
        "scenario_id": 1,
        "behavior": "constant_alpha",
        "behavior_alpha": 0.5,
        "steps": 40,
    }
    values.update(overrides)
    return values


def _to_transition_arrays(result: dict) -> TransitionArrays:
    return TransitionArrays(
        features=result["features"],
        state=result["state"],
        next_state=result["next_state"],
        physics_next_state=result["physics_next_state"],
        requested=result["requested"],
        applied=result["applied"],
        est_current=result["est_current"],
        true_current=result["true_current"],
        next_true_current=result["next_true_current"],
        context=result["context"],
        solver=result["solver"],
        plan=result["plan"],
        alpha=result["alpha"],
        metadata=pd.DataFrame(result["metadata"]),
    )

