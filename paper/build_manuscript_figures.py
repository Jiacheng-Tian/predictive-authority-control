# -*- coding: utf-8 -*-
"""Paper-grade figures for the PAC manuscript (English, journal style).

Inherits the v2 manuscript figure conventions (Arial, 7.2 in double column,
bold panel letters, SMC/MPC/PAC base palette) and adds the v4 evidence set.
Method naming carries no internal codenames:
  SMC / MPC / Fixed alpha / SSPO / Supervised PAC / Residual PAC
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "manuscript_figures"
OUT.mkdir(exist_ok=True)

STAGE2 = ROOT / "results" / "formal_v4" / "stage2-residual-rl-2026-09-23" / "formal"

METHODS = ["smc", "mpc", "constant_alpha", "sspo", "v3_transformer",
           "residual_rl"]
LABEL = {"smc": "SMC", "mpc": "MPC", "constant_alpha": "Fixed $\\alpha$=0.5",
         "sspo": "SSPO", "v3_transformer": "Supervised PAC",
         "residual_rl": "Residual PAC"}
COLOR = {"smc": "#514D83", "mpc": "#CC9429", "constant_alpha": "#8C9196",
         "sspo": "#3F7E72", "v3_transformer": "#E8730C",
         "residual_rl": "#B6242E"}

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

TINT = "#EFEDF5"
TINT_R = "#F7E9EA"
TINT_G = "#F2EAD4"


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("saved", stem)


def panel_letter(ax, letter):
    ax.text(-0.06, 1.06, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="right")


# ================================================================ Figure 1
def fig1_architecture():
    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.35, 1.0, 1.0], wspace=0.22)

    # ---------------- panel a: closed loop ----------------
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    panel_letter(ax, "a")

    def box(x, y, w, h, text, face, edge="#333333", size=7.0, lw=0.9, ls="-"):
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.06,rounding_size=0.12",
                               facecolor=face, edgecolor=edge, linewidth=lw,
                               linestyle=ls, zorder=2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=size, zorder=3, linespacing=1.25)

    def arrow(x1, y1, x2, y2, style="-", color="#333333", lw=0.9, rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=8,
                                     color=color, linewidth=lw, linestyle=style,
                                     connectionstyle=f"arc3,rad={rad}", zorder=1))

    # plant + environment (left)
    box(0.15, 5.55, 2.5, 2.3,
        "AUV plant (6-DOF)\nocean current disturbance\nRK4 @ 100 Hz", "#FFFFFF",
        edge="#333333", lw=1.1)
    ax.text(1.4, 8.15, r"reference $\mathbf{\eta}_d(t)$", fontsize=7,
            ha="center", color="#555555")
    arrow(1.4, 8.05, 1.4, 7.9, color="#555555")

    # experts (middle column)
    box(3.1, 7.3, 2.15, 1.05, "SMC expert\nprimary controller", "#EFEDF5",
        edge="#514D83", lw=1.0)
    box(3.1, 5.1, 2.15, 1.05, "MPC expert\nauthority controller", "#F2EAD4",
        edge="#CC9429", lw=1.0)
    arrow(2.65, 7.2, 3.1, 7.8)   # state -> experts
    arrow(2.65, 6.2, 3.1, 5.65)

    # blender
    box(5.7, 6.05, 1.9, 1.3,
        r"bounded blend" "\n" r"$\mathbf{u}=(1-\alpha)\,\mathbf{u}_{\mathrm{SMC}}"
        r"$+\alpha\,\mathbf{u}_{\mathrm{MPC}}$", "#FFFFFF", edge="#333333", lw=1.0)
    arrow(5.25, 7.8, 5.7, 6.95)
    arrow(5.25, 5.65, 5.7, 6.45)

    # actuator
    box(5.7, 4.35, 1.9, 0.95, "thruster allocation\n+ slew limits", "#FFFFFF",
        edge="#333333")
    arrow(6.65, 6.05, 6.65, 5.3)
    # feedback to plant
    ax.add_patch(FancyArrowPatch((7.6, 4.82), (2.2, 5.5),
                                 arrowstyle="-|>", mutation_scale=8,
                                 color="#333333", linewidth=0.9,
                                 connectionstyle="arc3,rad=-0.25"))
    ax.text(4.9, 3.85, "applied thruster wrench", fontsize=6.6, color="#555555",
            ha="center")

    # learning stack (right)
    box(7.95, 6.7, 2.0, 1.15, "frozen Transformer\nbackbone  $\\hat{\\alpha}$",
        "#EFEDF5", edge="#514D83", lw=1.0)
    box(7.95, 5.25, 2.0, 1.15,
        "residual head  $\\Delta\\alpha$\nzero-init, $|\\Delta\\alpha|\\leq0.1$",
        "#F7E9EA", edge="#E8730C", lw=1.1)
    box(7.95, 3.85, 2.0, 1.0, "safety filter\nclip / smooth / rate-limit",
        "#FFFFFF", edge="#333333")
    arrow(8.95, 6.7, 8.95, 6.4)
    arrow(8.95, 5.25, 8.95, 4.85)
    ax.text(9.55, 5.02, r"$\alpha_t=\hat{\alpha}+\Delta\alpha$", fontsize=7.2,
            rotation=90, va="center", ha="center", color="#333333")
    # history input
    ax.add_patch(FancyArrowPatch((2.4, 5.5), (7.9, 7.0),
                                 arrowstyle="-|>", mutation_scale=8,
                                 color="#514D83", linewidth=0.9,
                                 connectionstyle="arc3,rad=0.30"))
    ax.text(4.35, 7.35, r"history $\mathbf{X}_t\in\mathbb{R}^{16\times24}$",
            fontsize=7, color="#514D83", ha="center")
    # alpha out to blender
    ax.add_patch(FancyArrowPatch((7.9, 4.35), (6.4, 6.05),
                                 arrowstyle="-|>", mutation_scale=8,
                                 color="#E8730C", linewidth=1.1,
                                 connectionstyle="arc3,rad=0.2"))
    ax.text(7.5, 4.95, r"$\alpha_t$", fontsize=8, color="#E8730C",
            ha="center", fontweight="bold")

    # ---------------- panel b: safety filter ----------------
    axb = fig.add_subplot(gs[0, 1])
    axb.set_xlim(0, 10)
    axb.set_ylim(0, 10)
    axb.axis("off")
    panel_letter(axb, "b")
    axb.text(5, 9.55, "layered safety filter", fontsize=8.5, fontweight="bold",
             ha="center")
    steps = [
        ("bounded residual\n" r"$|\Delta\alpha|\leq 0.1$", TINT, "#514D83"),
        ("clip / smooth /\nrate limit", "#FFFFFF", "#333333"),
        ("forced primary on\nexpert failure", TINT_G, "#CC9429"),
        ("numeric fallback to\nfrozen baseline", TINT_R, "#B6242E"),
    ]
    y = 7.7
    for text, face, edge in steps:
        patch = FancyBboxPatch((0.7, y - 1.15), 8.6, 1.25,
                               boxstyle="round,pad=0.05,rounding_size=0.1",
                               facecolor=face, edgecolor=edge, linewidth=0.9)
        axb.add_patch(patch)
        axb.text(5.0, y - 0.52, text, ha="center", va="center", fontsize=6.8,
                 linespacing=1.2)
        if y > 3.4:
            axb.add_patch(FancyArrowPatch((5.0, y - 1.2), (5.0, y - 1.75),
                                          arrowstyle="-|>", mutation_scale=8,
                                          color="#333333", linewidth=0.9))
        y -= 1.85
    axb.text(5, 1.05, "zero-init head: behavior at $t=0$\nidentical to deployed baseline",
             fontsize=6.4, ha="center", color="#555555", linespacing=1.25)

    # ---------------- panel c: training ----------------
    axc = fig.add_subplot(gs[0, 2])
    axc.set_xlim(0, 10)
    axc.set_ylim(0, 10)
    axc.axis("off")
    panel_letter(axc, "c")
    axc.text(5, 9.55, "two-phase training", fontsize=8.5, fontweight="bold",
             ha="center")
    boxc = FancyBboxPatch((0.7, 5.35), 8.6, 3.3,
                          boxstyle="round,pad=0.06,rounding_size=0.1",
                          facecolor=TINT, edgecolor="#514D83", linewidth=0.9)
    axc.add_patch(boxc)
    axc.text(5.0, 7.75, "phase 1", fontsize=7.5, fontweight="bold",
             ha="center", color="#514D83")
    axc.text(5.0, 6.3, "supervised initialization\nshort-horizon oracle labels\n"
             "over 5-point $\\alpha$ grid", fontsize=6.8, ha="center",
             va="center", linespacing=1.3)
    axc.add_patch(FancyArrowPatch((5.0, 5.25), (5.0, 4.65),
                                  arrowstyle="-|>", mutation_scale=8,
                                  color="#333333", linewidth=0.9))
    boxc2 = FancyBboxPatch((0.7, 1.15), 8.6, 3.3,
                           boxstyle="round,pad=0.06,rounding_size=0.1",
                           facecolor=TINT_R, edgecolor="#E8730C", linewidth=0.9)
    axc.add_patch(boxc2)
    axc.text(5.0, 3.55, "phase 2", fontsize=7.5, fontweight="bold",
             ha="center", color="#E8730C")
    axc.text(5.0, 2.1, "constrained TD3 on frozen backbone\nbehavior regularization\n"
             r"$2\times10^{5}$ warm-start transitions", fontsize=6.8, ha="center",
             va="center", linespacing=1.3)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.02)
    save(fig, "fig1_system_architecture")


# ================================================================ Figure 2
# ================================================================ Figure 4
def fig4_results():
    pe = pd.read_csv(STAGE2 / "paired_effects.csv")
    bf = pd.read_csv(STAGE2 / "block_family_summary.csv")

    fig = plt.figure(figsize=(7.2, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28)

    # (a) forest of registered paired effects (RMSE): the archived
    # PAC-RL vs PAC-S contrast plus the SSPO contrast recomputed from raw
    # episode metrics with the same seed-level procedure.
    ax = fig.add_subplot(gs[0, 0])
    panel_letter(ax, "a")
    rows, ylabels, colors, sig = [], [], [], []
    rmse = pe[pe["metric"] == "rmse_3d"]
    for block in ("unseen", "seen"):
        r = rmse[(rmse["comparison"] == "residual_rl_vs_v3_transformer")
                 & (rmse["block"] == block)].iloc[0]
        rows.append((r["mean_difference"], r["ci95_low"], r["ci95_high"]))
        ylabels.append(f"PAC-RL vs PAC-S\n{'OOD' if block == 'unseen' else 'ID'}")
        colors.append(COLOR["residual_rl"])
        sig.append(r["ci95_low"] * r["ci95_high"] > 0)
    raw = pd.read_csv(STAGE2 / "raw_metrics.csv")
    from scipy import stats as sps
    for block in ("unseen", "seen"):
        b = raw[raw["block"] == block]
        sspo = b[b["method"] == "sspo"].groupby(
            ["family", "scenario_id", "environment_seed"])["rmse_3d"].mean()
        diffs = []
        for seed in (31000, 31001, 31002, 31003, 31004):
            rl = b[(b["method"] == "residual_rl") & (b["model_seed"] == seed)]
            rl_m = rl.groupby(
                ["family", "scenario_id", "environment_seed"])["rmse_3d"].mean()
            diffs.append((rl_m - sspo).mean())
        d = np.array(diffs)
        m = d.mean()
        tc = sps.t.ppf(0.975, 4)
        lo = m - tc * d.std(ddof=1) / np.sqrt(5)
        hi = m + tc * d.std(ddof=1) / np.sqrt(5)
        rows.append((m, lo, hi))
        ylabels.append(f"SSPO vs PAC-RL\n{'OOD' if block == 'unseen' else 'ID'}")
        colors.append(COLOR["sspo"])
        sig.append(lo * hi > 0)
    y = np.arange(len(rows))[::-1]
    for yi, (m, lo, hi), c, sg in zip(y, rows, colors, sig):
        ax.plot([lo, hi], [yi, yi], color=c, lw=1.4, solid_capstyle="butt")
        ax.scatter([m], [yi], s=26, color=c, zorder=3, marker="o",
                   facecolors=c if sg else "white", linewidths=1.2, edgecolors=c)
    ax.axvline(0, color="#777777", lw=0.8, ls="--", alpha=0.8)
    ax.set_yticks(y, ylabels, fontsize=7.5)
    ax.set_xlabel("paired RMSE effect (m),  model-seed t(4) 95% CI")
    ax.set_xlim(-0.082, 0.012)
    ax.grid(True, axis="x", color="#E3E6EA", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.985, 0.03, "filled = CI excludes 0", transform=ax.transAxes,
            fontsize=6.5, color="#555555", ha="right")

    # (b) OOD family heatmap
    axh = fig.add_subplot(gs[0, 1])
    panel_letter(axh, "b")
    unseen = bf[bf["block"] == "unseen"]
    fam_order = ["mass_damping_mismatch", "colored_noise",
                 "ou_current", "random_freq_amp", "actuator_delay_noise",
                 "fast_ou", "estimation_delay"]
    fam_order = [f for f in fam_order if f in set(unseen["family"])]
    fam_label = ["mass-damp.", "colored", "OU", "rand freq",
                 "actuator", "fast OU", "est. delay"]
    mat = np.full((len(fam_order), len(METHODS)), np.nan)
    for i, fam in enumerate(fam_order):
        for j, m in enumerate(METHODS):
            r = unseen[(unseen["family"] == fam) & (unseen["method"] == m)]
            if len(r):
                mat[i, j] = r["rmse_3d"].iloc[0]
    im = axh.imshow(mat, cmap="YlOrRd", aspect="auto",
                    vmin=0, vmax=float(np.nanmax(mat)))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            col = "white" if v > 0.58 * np.nanmax(mat) else "#222222"
            axh.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.6,
                     color=col)
    axh.set_xticks(range(len(METHODS)), [LABEL[m] for m in METHODS],
                   rotation=45, ha="right", fontsize=6.8)
    axh.set_yticks(range(len(fam_order)), fam_label, fontsize=6.8)
    axh.set_ylabel("OOD disturbance family")
    cbar = fig.colorbar(im, ax=axh, fraction=0.035, pad=0.02)
    cbar.set_label("RMSE (m)", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)

    fig.tight_layout()
    save(fig, "fig4_closed_loop_results")


# ================================================================ Figure 5
def fig5_actuator_case():
    case = pd.read_csv(ROOT / "paper" / "data" / "case_actuator_delay.csv")
    case = case[case["environment_seed"] == 43000]
    show = {"mpc": "MPC", "v3_transformer": "Supervised PAC",
            "residual_rl": "Residual PAC"}

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.0, 0.75], hspace=0.42)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2])
    panel_letter(ax1, "a")
    panel_letter(ax2, "b")
    panel_letter(ax3, "c")
    for m, lab in show.items():
        d = case[case["method"] == m].sort_values("time")
        rmse = d["episode_rmse"].iloc[0]
        ax1.plot(d["time"], d["error"], color=COLOR[m], lw=1.1,
                 label=f"{lab} ({rmse:.2f} m)")
        if m in ("v3_transformer", "residual_rl"):
            ax2.plot(d["time"], d["authority_alpha"], color=COLOR[m], lw=1.1,
                     label=lab)
    ax1.set_ylabel("3-D position error (m)")
    ax1.legend(loc="upper left", fontsize=7)
    ax1.grid(True, color="#E3E6EA", linewidth=0.55)
    ax1.spines[["top", "right"]].set_visible(False)
    ax2.axhspan(0.4, 0.6, color="#EFEDF5", zorder=0)
    ax2.set_ylabel("authority $\\alpha$")
    ax2.set_xlabel("time (s)")
    ax2.set_ylim(-0.03, 1.05)
    ax2.legend(loc="upper left", fontsize=7)
    ax2.grid(True, color="#E3E6EA", linewidth=0.55)
    ax2.spines[["top", "right"]].set_visible(False)
    v3a = case[case["method"] == "v3_transformer"]["authority_alpha"]
    rla = case[case["method"] == "residual_rl"]["authority_alpha"]
    hi_pct = (v3a > 0.9).mean() * 100
    band_pct = ((rla >= 0.4) & (rla <= 0.6)).mean() * 100
    ax2.text(0.99, 0.05,
             f"Supervised: {hi_pct:.1f}% of steps $\\alpha>0.9$;  "
             f"Residual: {band_pct:.1f}% of steps $\\alpha\\in[0.4,0.6]$",
             transform=ax2.transAxes, fontsize=6.4, color="#555555",
             ha="right", va="bottom")
    alpha_distribution_axis(ax3)
    fig.tight_layout()
    save(fig, "fig5_actuator_case")


def fig6_cost():
    ov = pd.read_csv(STAGE2 / "overall_summary.csv")
    ood = ov[ov["block"] == "unseen"]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    panel_letter(ax, "a")
    for m in METHODS:
        d = ood[ood["method"] == m]
        ax.scatter(d["solver_deadline_miss_step_fraction"], d["rmse_3d"],
                   s=26, color=COLOR[m], label=LABEL[m], edgecolors="white",
                   linewidths=0.5, zorder=3)
    ax.axvspan(0.685, 0.695, color="#E3E6EA", zorder=0)
    ax.text(0.691, 0.232, "baseline level (0.69):\nSMC / MPC /\nFixed $\\alpha$ / SSPO",
            fontsize=6.6, color="#555555", ha="center", va="top")
    ax.set_xlabel("solver deadline-miss step fraction (OOD)")
    ax.set_ylabel("OOD RMSE (m)")
    ax.set_ylim(0.10, 0.26)
    ax.legend(ncol=6, fontsize=7, loc="upper center")
    ax.grid(True, color="#E3E6EA", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fig6_cost_tradeoff")


def fig2_trajectories():
    """Old-manuscript style: 3-D path, xy projection, vertical tracking."""
    case = pd.read_csv(ROOT / "paper" / "data" / "trajectory_cases.csv")
    show = ["smc", "mpc", "v3_transformer", "residual_rl"]
    cols = [("seen", "ID sinusoidal current"), ("unseen", "OOD actuator degradation")]
    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 1.0, 1.0], hspace=0.20,
                          wspace=0.28)
    ax3d = [fig.add_subplot(gs[0, c], projection="3d") for c in range(2)]
    axes = np.array(ax3d + [fig.add_subplot(gs[1, c]) for c in range(2)]
                    + [fig.add_subplot(gs[2, c]) for c in range(2)]).reshape(3, 2)
    for c, (block, title) in enumerate(cols):
        for m in show:
            d = case[(case["block"] == block) & (case["method"] == m)].sort_values("time")
            lw = 1.0
            axes[0, c].plot(d["x"], d["y"], d["z"], color=COLOR[m], lw=lw,
                            label=LABEL[m])
            axes[1, c].plot(d["x"], d["y"], color=COLOR[m], lw=lw)
            axes[2, c].plot(d["time"], d["z_error"], color=COLOR[m], lw=lw)
        d0 = case[(case["block"] == block) & (case["method"] == show[0])].sort_values("time")
        axes[0, c].plot(d0["target_x"], d0["target_y"], d0["target_z"],
                        color="black", lw=0.8, ls="--", label="reference")
        axes[1, c].plot(d0["target_x"], d0["target_y"], color="black", lw=0.8, ls="--")
        axes[0, c].set_title(title, fontsize=8.5, loc="center", pad=0)
        axes[0, c].view_init(elev=22, azim=-60)
        axes[0, c].set_xlim(-3, 3)
        axes[0, c].set_ylim(-2, 2)
        axes[0, c].set_zlim(-1, 1)
        axes[0, c].set_xticks([-3, -2, -1, 0, 1, 2, 3])
        axes[0, c].set_yticks([-2, -1, 0, 1, 2])
        axes[0, c].set_zticks([-1, -0.5, 0, 0.5, 1])
        axes[0, c].tick_params(labelsize=7, pad=0)
        axes[0, c].set_box_aspect((3, 2, 1))
        axes[1, c].set_ylabel("y (m)")
        axes[1, c].set_xlabel("x (m)")
        axes[2, c].axhline(0, color="#777777", lw=0.7, ls="--", alpha=0.7)
        axes[2, c].set_ylabel("z error (m)")
        axes[2, c].set_xlabel("time (s)")
        for r in (1, 2):
            axes[r, c].grid(True, color="#E3E6EA", linewidth=0.55)
            axes[r, c].spines[["top", "right"]].set_visible(False)
        axes[1, c].text(-0.14, 1.10, chr(97 + c), transform=axes[1, c].transAxes,
                        fontsize=11, fontweight="bold")
    handles, labels_ = ax3d[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.0), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.2)
    save(fig, "fig2_closed_loop_trajectories")


# ================================================================ Figure 3
def fig3_error_attitude():
    case = pd.read_csv(ROOT / "paper" / "data" / "trajectory_cases.csv")
    show = ["smc", "mpc", "v3_transformer", "residual_rl"]
    cols = [("seen", "ID sinusoidal current"), ("unseen", "OOD actuator degradation")]

    def wrap_deg(a):
        return np.rad2deg(np.abs((a + np.pi) % (2 * np.pi) - np.pi))

    fig, axes = plt.subplots(3, 2, figsize=(7.2, 5.6), sharex="col")
    for c, (block, title) in enumerate(cols):
        for m in show:
            d = case[(case["block"] == block) & (case["method"] == m)].sort_values("time")
            axes[0, c].plot(d["time"], d["error"], color=COLOR[m], lw=1.0, label=LABEL[m])
            axes[1, c].plot(d["time"], wrap_deg(d["yaw"] - d["desired_yaw"]),
                            color=COLOR[m], lw=1.0)
            axes[2, c].plot(d["time"], np.rad2deg(d["pitch"]), color=COLOR[m], lw=1.0)
        axes[0, c].set_title(title, fontsize=8.5, loc="center")
        axes[0, c].set_ylabel("3-D position error (m)")
        axes[1, c].set_ylabel("|heading error| (deg)")
        axes[2, c].set_ylabel("pitch angle (deg)")
        axes[2, c].set_xlabel("time (s)")
        for r in range(3):
            axes[r, c].grid(True, color="#E3E6EA", linewidth=0.55)
            axes[r, c].spines[["top", "right"]].set_visible(False)
        axes[0, c].text(0.03, 1.12, chr(97 + c), transform=axes[0, c].transAxes,
                        fontsize=11, fontweight="bold")
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.0), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.0)
    save(fig, "fig3_error_attitude")


# ================================================================ Figure 2
def fig_network_training():
    """Network architecture (backbone + residual head) and two-phase training."""
    fig = plt.figure(figsize=(7.2, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.0], wspace=0.18)

    # ---------------- panel a: network architecture ----------------
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    panel_letter(ax, "a")

    def box(x, y, w, h, text, face, edge="#333333", size=6.8, lw=0.9, ls="-"):
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.05,rounding_size=0.1",
                               facecolor=face, edgecolor=edge, linewidth=lw,
                               linestyle=ls, zorder=2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=size, zorder=3, linespacing=1.2)

    def arrow(x1, y1, x2, y2, color="#333333", lw=0.9):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=8, color=color,
                                     linewidth=lw, zorder=1))

    # frozen backbone pipeline (upper)
    box(0.15, 4.15, 1.5, 1.15,
        "history $\\mathbf{X}_t$\n$16\\times24$", "#FFFFFF", edge="#333333", lw=1.0)
    box(2.0, 4.15, 1.7, 1.15,
        "linear embed 32\n+ positional", TINT, edge="#514D83")
    box(4.0, 3.95, 2.5, 1.55,
        "Transformer encoder layer\n4-head MHA $\\to$ FFN 128 (GELU)\nLayerNorm",
        TINT, edge="#514D83")
    box(6.9, 4.15, 1.6, 1.15, "sigmoid head\n$\\hat{\\alpha}\\in[0,1]$", TINT,
        edge="#514D83")
    arrow(1.65, 4.72, 2.0, 4.72)
    arrow(3.7, 4.72, 4.0, 4.72)
    arrow(6.5, 4.72, 6.9, 4.72)
    ax.text(5.25, 3.72, "frozen, 14,113 params", fontsize=6.2, ha="center",
            color="#514D83")
    ax.text(0.15, 5.55, "\u2744", fontsize=9, color="#514D83")

    # trainable residual head (lower)
    box(4.0, 1.55, 2.5, 1.15,
        "encoder token features (32-d)", "#FFFFFF", edge="#333333", lw=0.8,
        ls="--")
    ax.add_patch(FancyArrowPatch((5.25, 3.9), (5.25, 2.75), arrowstyle="-|>",
                                 mutation_scale=8, color="#8C9196",
                                 linewidth=0.9, linestyle="--"))
    box(6.9, 1.45, 2.6, 1.35,
        "residual head: Linear 32$\\to$64\n(GELU) $\\to$ Linear 64$\\to$1\n"
        "zero-init final layer", TINT_R, edge="#E8730C", lw=1.0)
    arrow(6.5, 2.1, 6.9, 2.1)
    box(9.9, 1.45, 2.0, 1.35,
        "$\\Delta\\alpha=0.1\\,$tanh$(\\cdot)$\n$|\\Delta\\alpha|\\leq0.1$",
        TINT_R, edge="#E8730C")
    arrow(9.5, 2.1, 9.9, 2.1)
    ax.text(8.2, 1.05, "trainable, 2,177 params", fontsize=6.2, ha="center",
            color="#E8730C")
    ax.text(6.9, 2.95, "\u2744  frozen token path", fontsize=6.0,
            color="#8C9196")

    # composition
    ax.add_patch(FancyArrowPatch((8.5, 4.15), (10.6, 3.5), arrowstyle="-|>",
                                 mutation_scale=8, color="#514D83",
                                 linewidth=0.9, connectionstyle="arc3,rad=0.2"))
    ax.add_patch(FancyArrowPatch((11.9, 2.8), (11.4, 3.5), arrowstyle="-|>",
                                 mutation_scale=8, color="#E8730C",
                                 linewidth=0.9))
    box(9.9, 3.55, 2.0, 1.0,
        "$\\alpha_t=\\mathrm{clip}_{[0,1]}(\\hat{\\alpha}+\\Delta\\alpha)$",
        "#FFFFFF", edge="#333333", lw=1.1, size=6.6)
    ax.text(10.9, 3.25, "safety filter", fontsize=6.2, ha="center",
            color="#555555")
    ax.text(0.15, 0.55, "zero initialization: $\\Delta\\alpha\\equiv0$ before "
            "training, so the composite policy equals the supervised baseline",
            fontsize=6.2, color="#555555")

    # ---------------- panel b: two-phase training ----------------
    axb = fig.add_subplot(gs[0, 1])
    axb.set_xlim(0, 10)
    axb.set_ylim(0, 10)
    axb.axis("off")
    panel_letter(axb, "b")
    axb.text(5, 9.6, "two-phase training", fontsize=8.5, fontweight="bold",
             ha="center")
    boxb = FancyBboxPatch((0.6, 5.4), 8.8, 3.0,
                          boxstyle="round,pad=0.06,rounding_size=0.1",
                          facecolor=TINT, edgecolor="#514D83", linewidth=0.9)
    axb.add_patch(boxb)
    axb.text(5.0, 7.7, "phase 1  supervised", fontsize=7.5, fontweight="bold",
             ha="center", color="#514D83")
    axb.text(5.0, 6.3, "imitation of short-horizon\noracle over 5-point "
             "$\\alpha$ grid\n(18,900 labels)", fontsize=6.8, ha="center",
             va="center", linespacing=1.3)
    axb.add_patch(FancyArrowPatch((5.0, 5.3), (5.0, 4.7), arrowstyle="-|>",
                                  mutation_scale=8, color="#333333",
                                  linewidth=0.9))
    boxb2 = FancyBboxPatch((0.6, 1.2), 8.8, 3.4,
                           boxstyle="round,pad=0.06,rounding_size=0.1",
                           facecolor=TINT_R, edgecolor="#E8730C",
                           linewidth=0.9)
    axb.add_patch(boxb2)
    axb.text(5.0, 3.9, "phase 2  residual TD3", fontsize=7.5,
             fontweight="bold", ha="center", color="#E8730C")
    axb.text(5.0, 2.3, "frozen backbone, twin critics,\nbehavior regularization, "
             "$2\\times10^{5}$ warm-start\ntransitions, 5 model seeds",
             fontsize=6.8, ha="center", va="center", linespacing=1.3)

    fig.tight_layout()
    save(fig, "fig_network_training")


# ================================================================ training curves
def fig_training_curves():
    """Three-panel training curves: reward, critic loss, actor loss."""
    hist_dir = ROOT / "runs" / "predictive_authority_v4" / "rl-round1"
    seeds = (31000, 31001, 31002, 31003, 31004)
    frames = []
    for seed in seeds:
        h = pd.read_csv(hist_dir / f"training_history_seed_{seed}.csv")
        frames.append(h)
    steps = frames[0]["env_steps"].to_numpy() / 1e5

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    metrics = [
        ("mean_episode_reward", "training episode reward", 0),
        ("critic_loss", "critic loss", 1),
        ("actor_loss", "actor loss", 2),
    ]
    for ax, (col, label, idx) in zip(axes, metrics):
        panel_letter(ax, chr(ord("a") + idx))
        data = np.array([f[col].to_numpy() for f in frames])
        mean = data.mean(axis=0)
        sd = data.std(axis=0)
        ax.fill_between(steps, mean - sd, mean + sd, alpha=0.18,
                        color="#E8730C", linewidth=0, zorder=1)
        for k in range(len(seeds)):
            ax.plot(steps, data[k], lw=0.5, color="#E8730C", alpha=0.4,
                    zorder=2)
        ax.plot(steps, mean, lw=1.6, color="#E8730C", zorder=3,
                label="mean $\pm$ 1 sd")
        ax.set_xlabel("env steps ($10^5$)")
        ax.set_title(label, fontsize=8)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=6.5)
        ax.grid(True, color="#E3E6EA", linewidth=0.4, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6.5)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_training_curves")


def alpha_distribution_axis(ax):
    """Box plot of per-episode mean alpha."""
    raw = pd.read_csv(STAGE2 / "raw_metrics.csv")
    ood = raw[raw["block"] == "unseen"]
    groups = [("v3_transformer", "Supervised PAC", COLOR["v3_transformer"]),
              ("residual_rl", "Residual PAC", COLOR["residual_rl"]),
              ("constant_alpha", "Fixed $\\alpha$=0.5", COLOR["constant_alpha"])]
    data, labels, cols = [], [], []
    for m, lab, col in groups:
        data.append(ood[ood["method"] == m]["authority_alpha_mean"].to_numpy())
        labels.append(lab)
        cols.append(col)
    bp = ax.boxplot(data, positions=range(len(groups)), widths=0.5,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=3,
                                    markerfacecolor="#8C9196",
                                    markeredgecolor="none", alpha=0.5),
                    medianprops=dict(color="#333333", linewidth=1.2),
                    whiskerprops=dict(color="#555555", linewidth=0.7),
                    capprops=dict(color="#555555", linewidth=0.7))
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
        patch.set_edgecolor(col)
    ax.axhline(0.5, color="#777777", lw=0.7, ls="--", alpha=0.7, zorder=0)
    ax.set_xticks(range(len(groups)), labels, fontsize=7)
    ax.set_ylabel("per-episode mean $\\alpha$")
    ax.set_ylim(0.2, 1.02)
    ax.grid(True, axis="y", color="#E3E6EA", linewidth=0.4, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.5)


def fig_control_quality():
    """Control quality: grouped bars styled after the v2 reference figures."""
    ovd = pd.read_csv(STAGE2 / "overall_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    metrics = [
        ("applied_control_cost", "control cost", 0),
        ("action_saturation_step_fraction", "saturation fraction", 1),
        ("actuator_rate_limit_episode_mean", "rate-limit activation", 2),
    ]
    x = np.arange(len(METHODS))
    w = 0.38
    short = {"smc": "SMC", "mpc": "MPC", "constant_alpha": "Fixed\n$\\alpha$",
             "sspo": "SSPO", "v3_transformer": "Sup.\nPAC",
             "residual_rl": "Res.\nPAC"}
    for ax, (col, label, idx) in zip(axes, metrics):
        panel_letter(ax, chr(ord("a") + idx))
        for k, block in enumerate(("seen", "unseen")):
            vals, sds = [], []
            for m in METHODS:
                d = ovd[(ovd["method"] == m) & (ovd["block"] == block)][col]
                vals.append(d.mean())
                sds.append(d.std(ddof=1) if len(d) > 1 else 0.0)
            bars = ax.bar(x + (k - 0.5) * w, vals, w,
                          color=["#8B87C8", "#B6242E"][k],
                          label=("ID" if k == 0 else "OOD"),
                          zorder=3, edgecolor="white", linewidth=0.4)
            ax.errorbar(x + (k - 0.5) * w, vals,
                        yerr=np.array(sds), fmt="none",
                        ecolor="#555555", elinewidth=0.6, capsize=1.5,
                        zorder=4)
        ax.set_xticks(x, [short.get(m, LABEL[m][:8]) for m in METHODS],
                      fontsize=5.8)
        ax.set_title(label, fontsize=8)
        ax.grid(True, axis="y", color="#E3E6EA", linewidth=0.4, alpha=0.7,
                zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6.2)
    axes[0].legend(fontsize=6.5, loc="upper left", framealpha=0.9)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_control_quality")


if __name__ == "__main__":
    fig1_architecture()
    fig_network_training()
    fig2_trajectories()
    fig3_error_attitude()
    fig4_results()
    fig5_actuator_case()
    fig6_cost()
    fig_training_curves()
    fig_control_quality()
    print(f"all figures -> {OUT}")
