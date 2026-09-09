"""Read-only summaries for archived formal PAC evidence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _method_summary(rows: pd.DataFrame) -> dict[str, float | int]:
    values = pd.to_numeric(rows["rmse_3d"], errors="raise")
    return {
        "rollouts": int(len(rows)),
        "mean_rmse_3d": float(values.mean()),
        "pooled_sample_sd": float(values.std(ddof=1)),
    }


def summarize_formal_results(results_dir: str | Path) -> dict[str, dict[str, float | int]]:
    """Recompute headline statistics without modifying archived files."""
    root = Path(results_dir)
    pac_rows = pd.read_csv(root / "raw_metrics.csv")
    fixed_rows = pd.read_csv(root / "fixed_controllers" / "raw_metrics.csv")

    pac = _method_summary(pac_rows)
    seed_means = pac_rows.groupby("train_seed")["rmse_3d"].mean()
    pac["training_seeds"] = int(seed_means.size)
    pac["training_seed_mean_sd"] = float(seed_means.std(ddof=1))

    smc_rows = fixed_rows[fixed_rows["controller"].eq("real10kg_smc_steady")]
    predictive_rows = fixed_rows[fixed_rows["controller"].eq("real10kg_mpc_event")]
    if smc_rows.empty or predictive_rows.empty:
        raise ValueError("archived fixed-controller evidence is incomplete")
    return {
        "pac": pac,
        "smc": _method_summary(smc_rows),
        "predictive": _method_summary(predictive_rows),
    }
