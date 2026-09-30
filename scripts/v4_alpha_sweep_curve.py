#!/usr/bin/env python3
"""RMSE-alpha curve from the preregistered fixed-alpha sweep (P1).

Reads the 2x120 sweep run (constant_alpha_0.25 / constant_alpha_0.75)
plus the alpha = 0.5 anchor rows of the fixed-TD3 formal run (same paired
episode grid) and writes a per-alpha summary CSV and a journal-style
figure for both blocks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

ALPHA_OF = {
    "constant_alpha_0.25": 0.25,
    "constant_alpha": 0.5,
    "constant_alpha_0.75": 0.75,
}
BLOCK_LABEL = {"seen": "In-distribution", "unseen": "Out-of-distribution"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", required=True,
                        help="formal run dir of the 2x120 alpha sweep")
    parser.add_argument("--anchor-dir", required=True,
                        help="formal run dir holding the alpha=0.5 anchor rows")
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-fig", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    sweep = pd.read_csv(Path(arguments.sweep_dir) / "raw_metrics.csv")
    anchor = pd.read_csv(Path(arguments.anchor_dir) / "raw_metrics.csv")
    anchor = anchor[anchor["method"] == "constant_alpha"]
    frames = [
        anchor.assign(method="constant_alpha"),
        sweep[sweep["method"].isin(["constant_alpha_0.25",
                                    "constant_alpha_0.75"])],
    ]
    raw = pd.concat(frames, ignore_index=True)
    raw["alpha"] = raw["method"].map(ALPHA_OF)
    if raw["alpha"].isna().any():
        raise ValueError("unexpected methods present in the sweep inputs")
    episode_counts = raw.groupby(["method", "block"]).size()
    expected = {"seen": 85, "unseen": 35}
    if not all(
            count == expected[block]
            for (_, block), count in episode_counts.items()):
        raise ValueError(f"sweep grid incomplete:\n{episode_counts}")

    summary = (
        raw.groupby(["block", "alpha"], as_index=False)
        .agg(
            rmse_3d_mean=("rmse_3d", "mean"),
            rmse_3d_sem=("rmse_3d", lambda s: s.std(ddof=1) / len(s) ** 0.5),
            heading_rmse_deg_mean=("heading_rmse_deg", "mean"),
            control_cost_mean=("applied_control_cost", "mean"),
            episodes=("rmse_3d", "size"),
        )
        .sort_values(["block", "alpha"])
    )
    out_csv = Path(arguments.out_csv) if arguments.out_csv else (
        Path(arguments.sweep_dir) / "alpha_sweep_summary.csv"
    )
    summary.to_csv(out_csv, index=False)
    print(summary.to_string(index=False))
    print("wrote", out_csv)

    out_fig = Path(arguments.out_fig) if arguments.out_fig else (
        Path(arguments.sweep_dir) / "alpha_sweep_curve"
    )
    colors = {"seen": "#3F7E72", "unseen": "#B6242E"}
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for block, frame in summary.groupby("block"):
        frame = frame.sort_values("alpha")
        ax.errorbar(
            frame["alpha"], frame["rmse_3d_mean"],
            yerr=1.96 * frame["rmse_3d_sem"], marker="o", ms=4,
            lw=1.4, capsize=2.5, color=colors[block],
            label=BLOCK_LABEL[block],
        )
    ax.set_xlabel(r"Blending coefficient $\alpha$")
    ax.set_ylabel("RMSE-3D (m)")
    ax.set_xticks([0.25, 0.5, 0.75])
    ax.legend()
    for extension in ("svg", "pdf", "png"):
        fig.savefig(f"{out_fig}.{extension}", bbox_inches="tight",
                    dpi=600 if extension == "png" else None)
    plt.close(fig)
    print("wrote", f"{out_fig}.svg/.pdf/.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
