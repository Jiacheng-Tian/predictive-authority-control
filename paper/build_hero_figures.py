# -*- coding: utf-8 -*-
"""ICRA/T-RO style redraw of Fig 1 (system architecture) and Fig 2
(network + training): shaded phase regions, thick colored flow arrows,
neuron-circle network drawings, rounded module cards."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "manuscript_figures"

NAVY = "#514D83"
ORANGE = "#E8730C"
RED = "#B6242E"
GOLD = "#CC9429"
TEAL = "#3F7E72"
GRAY = "#8C9196"
INK = "#2B2B33"
LAV = "#EFEDF5"
TINT_R = "#FBE9E2"
TINT_G = "#E7F0EA"
TINT_B = "#E8EEF7"
TINT_Y = "#FBF3DC"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("saved", stem)


def card(ax, x, y, w, h, text, face, edge, size=7.0, lw=1.0, ls="-",
         tc=None, weight="normal", radius=0.14):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0.03,rounding_size={radius}",
                           facecolor=face, edgecolor=edge, linewidth=lw,
                           linestyle=ls, zorder=3)
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=size, zorder=4, linespacing=1.25, color=tc or INK,
                fontweight=weight)
    return patch


def flow(ax, x1, y1, x2, y2, color, lw=2.6, rad=0.0, label=None, lsize=6.2,
         lsolid=True, lab_off=(0, 0), lab_rot=0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=color, linewidth=lw, zorder=2,
                                 linestyle="-" if lsolid else (0, (4, 2)),
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = (x1 + x2) / 2 + lab_off[0], (y1 + y2) / 2 + lab_off[1]
        ax.text(mx, my, label, fontsize=lsize, color=color, ha="center",
                va="center", rotation=lab_rot, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.85))


def region(ax, x, y, w, h, face, label=None, lsize=8.5, lcolor=None,
           lx=None, ly=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.18",
                                facecolor=face, edgecolor="none", zorder=1))
    if label:
        ax.text(lx if lx is not None else x + 0.25,
                ly if ly is not None else y + h - 0.32, label,
                fontsize=lsize, fontweight="bold",
                color=lcolor or INK, zorder=5, va="top")


def neurons(ax, cx, ys, r=0.09, fc="white", ec=INK, lw=1.0):
    for y in ys:
        ax.add_patch(Circle((cx, y), r, facecolor=fc, edgecolor=ec,
                            linewidth=lw, zorder=6))


def connect(ax, x1, ys1, x2, ys2, color="#7A86B8", lw=0.5, alpha=0.75):
    for y1 in ys1:
        for y2 in ys2:
            ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha,
                    zorder=5)


def auv_icon(ax, cx, cy, s=1.0):
    """Tiny cartoon AUV: hull + fin + thruster."""
    body = FancyBboxPatch((cx - 0.55 * s, cy - 0.22 * s), 1.1 * s, 0.44 * s,
                          boxstyle="round,pad=0.02,rounding_size=0.2",
                          facecolor="#D8DCE3", edgecolor=INK, linewidth=1.0,
                          zorder=6)
    ax.add_patch(body)
    ax.add_patch(Rectangle((cx - 0.72 * s, cy - 0.08 * s), 0.2 * s, 0.32 * s,
                           facecolor=GRAY, edgecolor=INK, linewidth=0.8,
                           zorder=6))
    ax.add_patch(Circle((cx + 0.42 * s, cy - 0.3 * s), 0.1 * s, facecolor=INK,
                        zorder=7))
    ax.plot([cx - 0.3 * s, cx + 0.25 * s], [cy + 0.22 * s, cy + 0.22 * s],
            color=INK, lw=1.4, zorder=6)


def fig1_architecture():
    fig = plt.figure(figsize=(7.2, 4.1))
    ax = fig.add_axes([0, 0, 0.62, 1])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ---------------- shaded regions ----------------
    region(ax, 4.6, 4.6, 7.25, 5.25, LAV, "LEARNED  ·  auditable", 8,
           NAVY)
    region(ax, 0.25, 0.5, 4.05, 4.7, TINT_G, "VERIFIED  ·  fixed", 8, TEAL)
    region(ax, 4.6, 0.5, 7.25, 3.8, "#F5F5F8", "ACTUATION", 8, GRAY)

    # ---------------- plant (bottom-left, verified region) ----------------
    auv_icon(ax, 2.1, 3.4, 1.5)
    ax.text(2.1, 2.35, "AUV plant (6-DOF)", fontsize=7.5, ha="center",
            fontweight="bold", color=INK)
    ax.text(2.1, 1.9, "10 kg · 6 thrusters · RK4 @ 100 Hz", fontsize=6,
            ha="center", color="#555555")
    # current arrows
    for dy in (0.0, 0.45):
        ax.add_patch(FancyArrowPatch((0.5, 4.45 + dy * 0.5), (1.5, 4.45 + dy * 0.5),
                                     arrowstyle="-|>", mutation_scale=10,
                                     color="#6FA8DC", linewidth=1.6, zorder=2))
    ax.text(0.95, 5.15, "current", fontsize=6, color="#6FA8DC", ha="center")

    # ---------------- experts (verified region, right of plant) ----------------
    card(ax, 0.55, 0.75, 1.5, 0.85, "SMC expert", "white", NAVY, 7.2, tc=NAVY)
    card(ax, 2.45, 0.75, 1.5, 0.85, "MPC expert\n(constrained)", TINT_Y, GOLD,
         7.2, tc="#7A5A00")

    # ---------------- learned region ----------------
    # encoder mini-network card
    card(ax, 4.9, 6.6, 3.1, 2.5, None, "white", NAVY, lw=1.2)
    ax.text(6.45, 8.75, "temporal encoder", fontsize=7.5, ha="center",
            fontweight="bold", color=NAVY)
    ax.text(5.12, 8.32, "\u2744 frozen", fontsize=6.5, color=NAVY,
            fontfamily="DejaVu Sans")
    ax.text(4.9, 6.78, "14,113 params", fontsize=6.2, ha="center",
            color=NAVY, fontweight="bold")
    ys_in = [7.0, 7.35, 7.7, 8.05]
    ys_hid = [7.15, 7.55, 7.95]
    ys_out = [7.55]
    neurons(ax, 5.5, ys_in, r=0.11, fc=LAV, ec=NAVY)
    neurons(ax, 6.45, ys_hid, r=0.11, fc="white", ec=NAVY)
    neurons(ax, 7.35, ys_out, r=0.13, fc=NAVY, ec=NAVY)
    connect(ax, 5.62, ys_in, 6.33, ys_hid)
    connect(ax, 6.57, ys_hid, 7.21, ys_out)
    ax.text(6.45, 6.78, "$\\mathbf{X}_t\\in\\mathbb{R}^{16\\times24}$",
            fontsize=6.2, ha="center", color=INK)

    # residual head card
    card(ax, 8.35, 6.6, 3.3, 2.5, None, "white", ORANGE, lw=1.2)
    ax.text(10.0, 8.75, "residual head", fontsize=7.5, ha="center",
            fontweight="bold", color="#B54E00")
    ax.text(8.55, 8.32, "\u25b2 trained", fontsize=6.5, color="#B54E00")
    ax.text(8.4, 6.78, "2,177 params", fontsize=6.2, ha="center",
            color="#B54E00", fontweight="bold")
    ys_h1 = [6.95, 7.3, 7.65, 8.0]
    ys_h2 = [7.3, 7.65]
    neurons(ax, 8.95, ys_h1, r=0.11, fc=TINT_R, ec=ORANGE)
    neurons(ax, 9.9, ys_h2, r=0.11, fc="white", ec=ORANGE)
    neurons(ax, 10.8, [7.45], r=0.13, fc=ORANGE, ec=ORANGE)
    connect(ax, 9.07, ys_h1, 9.78, ys_h2, color="#E3A36B")
    connect(ax, 10.02, ys_h2, 10.65, [7.45], color="#E3A36B")
    ax.text(10.0, 6.78, "zero-init, $|\\Delta\\alpha|\\leq0.1$", fontsize=6.2,
            ha="center", color=INK)

    # blender
    card(ax, 7.0, 5.0, 2.6, 1.1,
        "$\\alpha_t=\\mathrm{clip}_{[0,1]}(\\hat{\\alpha}_t+\\Delta\\alpha_t)$",
        "white", RED, 7.4, lw=1.3, tc=RED)

    # ---------------- flows ----------------
    # history: plant -> encoder
    flow(ax, 3.15, 5.6, 4.85, 7.6, NAVY, lw=3.0, rad=0.25,
         label="history (16 steps)", lsize=6.2, lab_off=(0.1, 0.35))
    # encoder -> blender
    flow(ax, 7.5, 6.55, 7.75, 6.15, NAVY, lw=2.6)
    # residual -> blender
    flow(ax, 9.6, 6.55, 9.2, 6.15, ORANGE, lw=2.6)
    # blender -> actuation
    flow(ax, 8.3, 4.95, 8.3, 4.35, RED, lw=3.0, label="$\\alpha_t$", lsize=7)
    # actuation region content
    card(ax, 5.0, 2.2, 2.9, 1.6,
         "bounded blend\n$\\mathbf{u}=(1-\\alpha)\\,\\mathbf{u}_{\\mathrm{SMC}}"
         "+\\alpha\\,\\mathbf{u}_{\\mathrm{MPC}}$", "white", INK, 7.0, lw=1.2)
    card(ax, 8.5, 2.2, 3.0, 1.6,
         "thruster allocation\n+ slew limits\nfallback on fault", "white",
         INK, 7.0)
    flow(ax, 7.9, 3.0, 8.45, 3.0, INK, lw=2.2)
    # experts -> blend
    flow(ax, 2.1, 1.65, 5.6, 2.55, NAVY, lw=2.2, rad=-0.2, label="$\\mathbf{u}^{\\mathrm{SMC}}$",
         lsize=6.4, lab_off=(0.2, 0.3))
    flow(ax, 3.9, 1.65, 6.4, 2.55, GOLD, lw=2.2, rad=0.2, label="$\\mathbf{u}^{\\mathrm{MPC}}$",
         lsize=6.4, lab_off=(-0.1, -0.4))
    # actuator -> plant (closed loop)
    ax.add_patch(FancyArrowPatch((10.0, 2.15), (3.0, 4.15),
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=TEAL, linewidth=3.0,
                                 connectionstyle="arc3,rad=-0.22", zorder=2))
    ax.text(6.6, 1.0, "thruster wrench", fontsize=6.4, color=TEAL,
            ha="center", rotation=8)
    # safety filter tag inside actuation
    card(ax, 5.0, 0.75, 6.5, 0.75,
         "safety filter:  bounded $\\Delta\\alpha$ → clip / smooth / rate-limit"
         " → forced primary → numeric fallback", "#FFFFFF", GRAY, 6.4, ls="--")

    ax.text(0.25, 9.7, "a", fontsize=12, fontweight="bold")

    # ---------------- panel b: safety ladder ----------------
    axb = fig.add_axes([0.635, 0, 0.365, 1])
    axb.set_xlim(0, 10)
    axb.set_ylim(0, 10)
    axb.axis("off")
    axb.text(0.3, 9.7, "b", fontsize=12, fontweight="bold")
    axb.text(5, 9.55, "layered safety filter", fontsize=8.5,
             fontweight="bold", ha="center")
    steps = [
        ("bounded residual", "$|\\Delta\\alpha|\\leq0.1$", LAV, NAVY),
        ("signal conditioning", "clip · smooth · rate-limit", "#FFFFFF", INK),
        ("expert failure", "forced primary", TINT_Y, GOLD),
        ("numeric fault", "fallback to frozen baseline", TINT_R, RED),
    ]
    y = 8.3
    for i, (t1, t2, face, edge) in enumerate(steps):
        card(axb, 1.0, y - 1.05, 8.0, 1.3, None, face, edge, lw=1.1)
        axb.text(5.0, y - 0.32, f"{i + 1} · {t1}", fontsize=7.4,
                 ha="center", fontweight="bold", color=edge)
        axb.text(5.0, y - 0.78, t2, fontsize=6.8, ha="center", color=INK)
        if i < 3:
            flow(axb, 5.0, y - 1.1, 5.0, y - 1.62, INK, lw=2.0)
        y -= 1.85
    axb.add_patch(FancyBboxPatch((1.0, 0.35), 8.0, 0.85,
                                 boxstyle="round,pad=0.03,rounding_size=0.12",
                                 facecolor=TINT_G, edgecolor=TEAL,
                                 linewidth=1.2, zorder=3))
    axb.text(5.0, 0.78,
             "worst case = deployed baseline\n(zero-init: bitwise start)",
             fontsize=6.8, ha="center", va="center", color=TEAL,
             fontweight="bold", linespacing=1.25, zorder=4)

    fig.subplots_adjust(left=0, right=1, top=0.97, bottom=0.02)
    save(fig, "fig1_system_architecture")


def fig_network_training():
    """Neuron-level architecture + two-phase training, ICRA style."""
    fig = plt.figure(figsize=(7.2, 3.9))
    ax = fig.add_axes([0, 0.30, 0.60, 0.70])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    region(ax, 0.15, 3.4, 11.7, 3.45, LAV, "PHASE 1 → FROZEN  ·  supervised", 7.5,
           NAVY)
    region(ax, 3.3, 0.25, 8.55, 2.85, TINT_R,
           "PHASE 2 → TRAINED  ·  constrained TD3", 7.5, RED)

    # ---- encoder network ----
    card(ax, 0.4, 3.8, 3.5, 2.7, None, "white", NAVY, lw=1.2)
    ax.text(2.15, 6.18, "encoder  ·  14,113 params", fontsize=7.2,
            ha="center", fontweight="bold", color=NAVY)
    ys_in = [4.15, 4.45, 4.75, 5.05, 5.35]
    ys_e = [4.4, 4.75, 5.1]
    neurons(ax, 1.05, ys_in, r=0.1, fc=LAV, ec=NAVY)
    neurons(ax, 2.15, ys_e, r=0.1, fc="white", ec=NAVY)
    connect(ax, 1.16, ys_in, 2.03, ys_e)
    ax.text(1.05, 3.95, "16×24", fontsize=5.8, ha="center", color="#555555")
    ax.text(2.15, 3.95, "embed 32\n+ pos.", fontsize=5.6, ha="center",
            color="#555555", linespacing=1.1)
    # attention block
    card(ax, 2.75, 4.35, 1.0, 1.05, "attn\n×4", NAVY, NAVY, 6.0, tc="white")
    connect(ax, 2.27, ys_e, 2.73, [4.6, 4.9, 5.2])
    card(ax, 3.15, 4.05, 0.55, 0.5, "FFN\n128", "white", NAVY, 5.4)
    flow(ax, 3.25, 4.9, 3.4, 4.6, NAVY, lw=1.4)

    # ---- alpha head ----
    card(ax, 4.15, 4.15, 1.5, 1.3, None, "white", NAVY, lw=1.0)
    ax.text(4.9, 5.2, "$\\hat{\\alpha}_t$", fontsize=9, ha="center",
            color=NAVY, fontweight="bold")
    ax.text(4.9, 4.55, "sigmoid head\nfinal token", fontsize=5.8, ha="center",
            color="#555555", linespacing=1.2)
    connect(ax, 3.72, [4.6, 4.9, 5.2], 4.12, [4.8])

    # ---- residual head ----
    card(ax, 3.55, 0.6, 3.6, 1.9, None, "white", ORANGE, lw=1.2)
    ax.text(5.35, 2.25, "residual head  ·  2,177 params", fontsize=7.2,
            ha="center", fontweight="bold", color="#B54E00")
    ys_r1 = [0.95, 1.25, 1.55, 1.85]
    ys_r2 = [1.25, 1.55]
    neurons(ax, 4.15, ys_r1, r=0.1, fc=TINT_R, ec=ORANGE)
    neurons(ax, 5.2, ys_r2, r=0.1, fc="white", ec=ORANGE)
    connect(ax, 4.27, ys_r1, 5.08, ys_r2, color="#E3A36B")
    neurons(ax, 6.3, [1.4], r=0.13, fc=ORANGE, ec=ORANGE)
    connect(ax, 5.32, ys_r2, 6.16, [1.4], color="#E3A36B")
    ax.text(5.35, 0.75, "Linear 32→64 → GELU → Linear 64→1 (zero-init)",
            fontsize=5.9, ha="center", color="#555555")
    # token path down
    flow(ax, 4.9, 4.1, 5.0, 2.55, GRAY, lw=2.0, lsolid=False,
         label="token features (32-d)", lsize=5.8, lab_off=(1.15, 0))

    # ---- composition ----
    card(ax, 7.6, 3.9, 2.1, 1.5,
         "$\\alpha_t = \\mathrm{clip}_{[0,1]}(\\hat{\\alpha}_t" +
         "+\\Delta\\alpha_t)$", "white", RED, 7.0, lw=1.3, tc=RED)
    flow(ax, 5.7, 4.8, 7.55, 4.7, NAVY, lw=2.4, label="$\\hat{\\alpha}_t$",
         lsize=6.4, lab_off=(0, 0.28))
    flow(ax, 6.45, 1.4, 8.6, 3.85, ORANGE, lw=2.4, rad=-0.25,
         label="$\\Delta\\alpha_t = 0.1\\,\\tanh(z_t)$", lsize=6.2,
         lab_off=(1.7, -0.4))
    card(ax, 10.0, 3.9, 1.8, 1.5, "safety\nfilter", TINT_G, TEAL, 7.4,
         lw=1.2, tc=TEAL, weight="bold")
    flow(ax, 9.75, 4.65, 9.95, 4.65, RED, lw=2.6)

    ax.text(0.15, 6.85, "a", fontsize=12, fontweight="bold")

    # ---------------- panel b: two-phase timeline ----------------
    axb = fig.add_axes([0.62, 0.02, 0.37, 0.96])
    axb.set_xlim(0, 10)
    axb.set_ylim(0, 10)
    axb.axis("off")
    axb.text(0.3, 9.6, "b", fontsize=12, fontweight="bold")
    axb.text(5, 9.45, "two-phase training", fontsize=8.5, fontweight="bold",
             ha="center")

    # phase 1
    card(axb, 0.6, 6.0, 8.8, 2.7, None, LAV, NAVY, lw=1.2)
    axb.text(5.0, 8.25, "phase 1 · supervised initialization", fontsize=7.6,
             fontweight="bold", ha="center", color=NAVY)
    axb.text(5.0, 7.05,
             "oracle labels on a 5-point $\\alpha$ grid\n"
             "18,900 samples · AdamW · 180 epochs\n"
             "$\\to$ five seeds, then frozen forever",
             fontsize=6.6, ha="center", va="center", linespacing=1.45)
    flow(axb, 5.0, 5.9, 5.0, 5.25, INK, lw=2.4)

    # phase 2
    card(axb, 0.6, 2.1, 8.8, 3.05, None, TINT_R, RED, lw=1.2)
    axb.text(5.0, 4.55, "phase 2 · residual TD3", fontsize=7.6,
             fontweight="bold", ha="center", color=RED)
    axb.text(5.0, 3.2,
             "backbone frozen · $2{\\times}10^{5}$ warm-start transitions\n"
             "twin critics · behavior regularization\n"
             "env seeds disjoint from evaluation",
             fontsize=6.6, ha="center", va="center", linespacing=1.45)

    # outcome
    card(axb, 1.4, 0.4, 7.2, 1.2,
         "zero-init $\\Rightarrow$ composite $\\equiv$ deployed baseline at $t=0$\n"
         "(full-episode unit test, tight tolerance)",
         TINT_G, TEAL, 6.8, lw=1.2, tc=TEAL, weight="bold")

    fig.subplots_adjust(left=0, right=1, top=0.98, bottom=0.02)
    save(fig, "fig_network_training")


if __name__ == "__main__":
    fig1_architecture()
    fig_network_training()
