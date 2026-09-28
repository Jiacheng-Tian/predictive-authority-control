# -*- coding: utf-8 -*-
"""Build the English manuscript draft (formal paper register, style-v2).

Restructured per reviewer and author feedback:
- one question carried from title through introduction to conclusion
- motion-performance figures first (trajectories, error, attitude), mechanism second
- numbered limitations, planned-hardware section, displayed equations,
  self-limiting captions, simulation-level disclaimer in the abstract
Table numbers are computed live from the frozen archives.
"""
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "manuscript_figures"
OUT = ROOT / "paper" / "PAC_manuscript_draft_v2.docx"
STAGE2 = ROOT / "results" / "formal_v4" / "stage2-residual-rl-2026-09-23"

BLACK = RGBColor(0, 0, 0)
GREY = RGBColor(0x60, 0x60, 0x60)

ov = pd.read_csv(STAGE2 / "formal" / "overall_summary.csv")
bf = pd.read_csv(STAGE2 / "formal" / "block_family_summary.csv")

METHOD_ORDER = ["smc", "mpc", "constant_alpha", "sspo", "v3_transformer",
                "residual_rl"]
METHOD_NAME = {"smc": "SMC", "mpc": "MPC", "constant_alpha": "Fixed \u03b1 = 0.5",
               "sspo": "SSPO", "v3_transformer": "PAC-S",
               "residual_rl": "PAC-RL"}

FIG_N = {"n": 0}
TAB_N = {"n": 0}


def block_mean(method, block, col):
    return ov[(ov["method"] == method) & (ov["block"] == block)][col].mean()


def fam_mean(method, block, family, col="rmse_3d"):
    return bf[(bf["method"] == method) & (bf["block"] == block)
              & (bf["family"] == family)][col].mean()


def style_run(run, size=11, bold=False, italic=False, color=BLACK,
              font="Times New Roman"):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    rf.set(qn("w:eastAsia"), font)


def para(doc, text="", size=11, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         after=6, color=BLACK, font="Times New Roman"):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.line_spacing = 1.3
    pf.space_after = Pt(after)
    if text:
        run = p.add_run(text)
        style_run(run, size=size, bold=bold, italic=italic, color=color, font=font)
    return p


def rich(doc, segments, size=11, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=6):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.line_spacing = 1.3
    pf.space_after = Pt(after)
    for text, kw in segments:
        run = p.add_run(text)
        style_run(run, size=size, **kw)
    return p


def equation(doc, body, number):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.line_spacing = 1.3
    pf.space_before = Pt(4)
    pf.space_after = Pt(6)
    run = p.add_run(body + "\u2003\u2003\u2003(" + str(number) + ")")
    style_run(run, size=11, italic=True)
    return p


def heading(doc, level, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.3
    run = p.add_run(text)
    if level == 1:
        pf.space_before = Pt(14)
        pf.space_after = Pt(6)
        style_run(run, size=13, bold=True)
    else:
        pf.space_before = Pt(10)
        pf.space_after = Pt(4)
        style_run(run, size=11.5, bold=True)
    return p


def figure(doc, stem, caption, width_in=6.3):
    FIG_N["n"] += 1
    n = FIG_N["n"]
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(8)
    run = p.add_run()
    run.add_picture(str(FIG / f"{stem}.png"), width=Cm(width_in * 2.54))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    cap.paragraph_format.space_after = Pt(10)
    c1 = cap.add_run(f"Figure {n}. ")
    style_run(c1, size=9, bold=True)
    c2 = cap.add_run(caption)
    style_run(c2, size=9)
    return n


def _shade(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def table(doc, header, rows, caption, widths_in):
    TAB_N["n"] += 1
    n = TAB_N["n"]
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    cap.paragraph_format.keep_with_next = True
    cap.paragraph_format.space_before = Pt(8)
    c1 = cap.add_run(f"Table {n}. ")
    style_run(c1, size=9, bold=True)
    c2 = cap.add_run(caption)
    style_run(c2, size=9, bold=True)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for j, h in enumerate(header):
        cell = t.rows[0].cells[j]
        cell.width = Cm(widths_in[j] * 2.54)
        _shade(cell, "F2F2F2")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(h)
        style_run(run, size=8.5, bold=True)
    t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.rows[i + 1].cells[j]
            cell.width = Cm(widths_in[j] * 2.54)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 1.15
            run = p.add_run(str(val))
            style_run(run, size=8.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return n


def build():
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(11)
    st._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Cm(2.54)
        s.left_margin = s.right_margin = Cm(2.54)
    sec = doc.sections[0]
    sec.header.is_linked_to_previous = False
    hp = sec.header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style_run(hp.add_run("Predictive Authority Control for Autonomous Underwater Vehicles"),
              size=9, color=GREY)
    sec.footer.is_linked_to_previous = False
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE \\* arabic \\* MERGEFORMAT")
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "18")
    col = OxmlElement("w:color")
    col.set(qn("w:val"), "808080")
    rpr.append(sz)
    rpr.append(col)
    r.append(rpr)
    t_el = OxmlElement("w:t")
    t_el.text = "1"
    r.append(t_el)
    fld.append(r)
    fp._p.append(fld)

    # ------------------------- title block -------------------------
    para(doc, "Predictive Authority Control: Learning How Much to Trust Each "
              "Controller, and Repairing That Trust Under Distribution Shift*",
         size=15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=10)
    para(doc, "[Author names]", size=11, italic=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    para(doc, "[Affiliations]", size=10, italic=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    para(doc, "* The abbreviation PAC is unrelated to probably approximately "
              "correct learning.", size=9, italic=True, color=GREY,
         align=WD_ALIGN_PARAGRAPH.CENTER, after=12)

    # ------------------------- abstract -------------------------
    heading(doc, 1, "Abstract")
    para(doc, "Classical underwater controllers fail in complementary ways: "
              "sliding-mode control (SMC) is robust but coarse, and constrained "
              "model predictive control (MPC) is precise but collapses under "
              "actuator degradation, where our simulation shows a tracking error "
              "of 0.822 m, nine times the nominal level. Predictive authority "
              "control (PAC) does not replace these controllers; a compact "
              "temporal encoder predicts a bounded authority coefficient that "
              "convexly blends their commands, so no thruster command is ever "
              "produced by the network. We extend the supervised policy (PAC-S) "
              "with a constrained residual stage (PAC-RL): the supervisor is "
              "frozen, and a zero-initialized 2,177-parameter head outputs a "
              "correction bounded by 0.1, so the deployed system is numerically "
              "identical to the baseline at initialization. In a paired, "
              "registered evaluation of 1,680 simulation rollouts over "
              "in-distribution and out-of-distribution disturbance families, "
              "PAC-RL reduces out-of-distribution tracking RMSE by 28% "
              "(0.161 m to 0.116 m; 95% CI [\u22120.074, \u22120.015] m), confines the "
              "degradation failure to 0.090 m, and pays a measured price: "
              "in-distribution error +0.4%, heading error +4.3%, and a solver "
              "deadline-miss increase. A constant blend \u03b1 = 0.5 comes within 3% "
              "overall, so the residual's value is concentrated where adaptation "
              "is hardest, not in the average. Of the six registered criteria "
              "concerning the residual stage, three pass, two fail, and one "
              "is unevaluated; all are reported.")
    rich(doc, [("All evidence is simulation-level; the hardware protocol of "
                "Section 8 remains to be executed.", {"bold": True})], after=12)

    # ------------------------- 1 introduction -------------------------
    heading(doc, 1, "1  Introduction")
    para(doc, "Accurate trajectory tracking is central to autonomous underwater "
              "vehicle (AUV) inspection, intervention, and sampling. The task "
              "remains difficult because hydrodynamic coefficients are "
              "uncertain, environmental currents vary across regions, and "
              "thrusters age with response lag and noise [1,2]. SMC rejects "
              "bounded disturbances through a transparent feedback structure "
              "but pays in conservative gains and chattering. MPC incorporates "
              "prediction and constraints [3], yet its performance depends on "
              "model fidelity: under actuator degradation the internal model "
              "misrepresents the plant, and the optimizer acts confidently on "
              "false premises.")
    para(doc, "This paper asks one question: given two complete and "
              "complementary controllers, when should the vehicle trust each "
              "one, and by how much? The answer varies continuously with the "
              "disturbance regime and cannot be enumerated by hand-designed "
              "switching rules [16].")
    para(doc, "We study this question through authority allocation. Rather than "
              "learning a six-dimensional thruster command, the PAC policy "
              "predicts one scalar \u03b1\u209c \u2208 [0, 1] and applies")
    equation(doc, "u\u209c = (1 \u2212 \u03b1\u209c) u\u209c\u1d34\u1d39\u1d9c + \u03b1\u209c u\u209c\u1d39\u1d33\u1d9c ,   \u03b1\u209c \u2208 [0, 1],", 1)
    para(doc, "where u\u1d34\u1d39\u1d9c and u\u1d39\u1d33\u1d9c are the SMC and MPC commands. The two "
              "controllers remain explicit, the learned output is bounded and "
              "interpretable, and a rate limiter prevents abrupt authority "
              "transfer. A Transformer encodes a short history because "
              "controller preference depends not only on instantaneous error "
              "but on whether that error is growing, recovering, or coincident "
              "with actuator saturation.")
    para(doc, "Our earlier experiments established the supervised variant of "
              "this idea and exposed its weakness: under disturbance families "
              "never seen in training, the learned coefficient degrades below a "
              "non-adaptive constant blend and oscillates over its full range. "
              "This paper answers that weakness with a constrained residual "
              "and makes three contributions, each responding to the question "
              "above:")
    rich(doc, [("Bounded residual learning over a frozen supervisor. ", {"bold": True}),
               ("A reinforcement-learning head perturbs the supervised answer "
                "to the trust question by at most 0.1 per step and is "
                "zero-initialized, so the composite policy is numerically "
                "identical to the deployed baseline before training and can "
                "only revise trust within a bounded envelope afterwards.", {})])
    rich(doc, [("Motion-level evidence under distribution shift. ", {"bold": True}),
               ("Trajectories, attitude, and error time series\u2014not only "
                "aggregate metrics\u2014show where the residual earns its value: "
                "the supervised policy transfers trust to a failing optimizer, "
                "and the residual learns to veto that transfer.", {})])
    rich(doc, [("A registered evaluation that reports its failures. ", {"bold": True}),
               ("Eight methods are compared over 120 paired episodes per "
                "method-seed combination with model-seed confidence intervals "
                "and no post-hoc selection; two failed criteria and one "
                "unevaluated criterion are reported alongside the passes.", {})])
    para(doc, "The remainder of the paper describes the framework (Section 3), "
              "the learning stages (Section 4), the experimental protocol "
              "(Section 5), results at the motion and mechanism levels "
              "(Section 6), limitations (Section 7), and the planned hardware "
              "validation (Section 8).")

    # ------------------------- 2 related work -------------------------
    heading(doc, 1, "2  Related Work")
    heading(doc, 2, "2.1  Underwater vehicle control")
    para(doc, "Marine vehicle control is developed from rigid-body dynamics "
              "with added mass, damping, and restoring forces [1,2]. Hybrid "
              "schemes combine nonlinear MPC with sliding-mode terms under "
              "threshold switching, retaining each expert in its favorable "
              "regime [17]; such switching surfaces are fixed at design time, "
              "whereas PAC answers the trust question online and continuously. "
              "Learning-based AUV control has advanced rapidly: improved TD3 "
              "path following [10] and six-degree-of-freedom control trained "
              "in parallel simulators with zero-shot transfer [8] learn "
              "commands directly. PAC instead learns a one-dimensional "
              "arbitration variable over complete structured controllers.")
    heading(doc, 2, "2.2  Learning with structured controllers")
    para(doc, "Residual reinforcement learning superposes a learned correction "
              "on a hand-designed base policy [5]. Actor-critic MPC embeds a "
              "differentiable optimizer in the actor [6], and learning-based "
              "MPC surveys data-driven refinement with safety considerations "
              "[4]. PAC differs in the interface: the residual applies to a "
              "bounded scalar authority rather than actuator commands, the "
              "supervisor is frozen rather than jointly trained, and zero "
              "initialization yields exact initial equivalence to the deployed "
              "system. Learned safety filters train barrier-like corrections "
              "[11]; the PAC safety filter instead composes fixed bounds with "
              "structured fallbacks and contains no learned component in the "
              "safety path. Embedded solvers such as TinyMPC [7] constrain "
              "what an onboard arbitration layer may assume computationally.")
    heading(doc, 2, "2.3  Temporal encoders for sequential decisions")
    para(doc, "Self-attention [12] and sequence-modeling formulations of "
              "control [13,14] scale with data and width. The PAC encoder "
              "deliberately occupies the opposite regime: one encoder layer "
              "over a 16-step history mapping to a single coefficient, which "
              "suffices for the arbitration interface and keeps edge deployment "
              "plausible. Diffusion policies [15] and imitation navigation [9] "
              "address different output spaces and are complementary.")

    # ------------------------- 3 framework -------------------------
    heading(doc, 1, "3  Predictive Authority Control")
    heading(doc, 2, "3.1  Vehicle model and tracking objective")
    para(doc, "The vehicle pose and body velocity are \u03b7 = [x, y, z, \u03c6, \u03b8, \u03c8]\u1d40 "
              "and \u03bd = [u, v, w, p, q, r]\u1d40. The simulator integrates the "
              "diagonalized six-degree-of-freedom model with added mass, "
              "linear and quadratic damping, and restoring stiffness by "
              "fourth-order Runge\u2013Kutta at 100 Hz for 2,100 steps (21 s) per "
              "episode. Six thrusters in an X layout are limited to 35 N with "
              "magnitude and slew-rate constraints. The reference is a "
              "three-dimensional Lissajous path with desired yaw aligned to "
              "the horizontal tangent; initial pose and velocity are perturbed "
              "with standard deviations 0.03 m and 0.01 m/s.")
    heading(doc, 2, "3.2  Structured controller experts")
    para(doc, "The primary expert is an equivalent-control SMC whose "
              "translational sliding surface combines velocity error and pose "
              "error, with feedforward acceleration, damping compensation, and "
              "a bounded reaching term. The second expert is a "
              "receding-horizon linear-time-varying constrained MPC with a "
              "20-step horizon (0.2 s), normalized thruster limits, warm-start "
              "smoothing, and an online solve-time limit of 7.5 ms with "
              "fallback to the previous plan. This expert replaces the "
              "one-step quadratic tracking law used in an earlier "
              "implementation of this framework; all results here use the "
              "constrained MPC.")
    heading(doc, 2, "3.3  History features and temporal encoder")
    para(doc, "At each step a 24-dimensional feature vector summarizes pose "
              "errors, body velocities, per-controller command statistics and "
              "disagreement, near-saturation fraction, horizontal current "
              "components, and trajectory phase. The most recent 16 vectors "
              "form the encoder input; one Transformer encoder layer (four "
              "heads, 128-unit feed-forward, 14,113 trainable parameters) maps "
              "the final token to \u03b1\u0302\u209c through a sigmoid head. At inference "
              "the prediction is scaled by 1.2, clipped, smoothed with "
              "coefficient 0.5, and rate-limited to 0.0125 per step.")
    heading(doc, 2, "3.4  Layered safety filter")
    para(doc, "Every learned contribution passes a fixed filter stack, in "
              "execution order: (i) the residual is bounded; (ii) the blended "
              "coefficient is clipped, smoothed, and rate-limited; (iii) "
              "expert failure forces primary control; and (iv) non-finite "
              "values fall back to the frozen supervised prediction. Because "
              "the residual head is zero-initialized, the composite policy "
              "reproduces the deployed baseline numerically at "
              "initialization, enforced by a unit test comparing full-episode "
              "rollouts to tight numerical tolerances. The worst case of the "
              "learning components is therefore the baseline policy itself.")
    figure(doc, "fig1_system_architecture",
           "Predictive authority control architecture. (a) Closed loop: the "
           "experts generate commands blended by \u03b1; a frozen temporal encoder "
           "predicts the supervised coefficient and a bounded, zero-initialized "
           "residual head revises it. (b) Layered safety filter in execution order. "
           "(c) Two-phase training.")

    # ------------------------- 4 learning -------------------------
    heading(doc, 1, "4  Learning the Authority Residual")
    heading(doc, 2, "4.1  Supervised initialization")
    para(doc, "The supervised policy (PAC-S) is trained by imitation of a "
              "short-horizon oracle: for each training state, candidate "
              "coefficients from a five-point grid are held for ten steps with "
              "the current frozen, and the label minimizes")
    equation(doc, "J(\u03b1) = \u2016e\u2093\u1d67\u2016\u00b2 + 0.7 e\u1d63\u00b2 + 0.08 e\u03c8\u00b2 + 0.02 mean([|u| \u2212 0.92]\u208a\u00b2) "
              "+ 0.01 mean((u \u2212 u\u209c\u208b\u2081)\u00b2),", 2)
    para(doc, "a cost combining horizontal and vertical position error, heading "
              "error, near-saturation magnitude, and command change. Training "
              "uses 18,900 labeled samples, AdamW for 180 epochs, and a "
              "weighted squared loss. Five model seeds are trained "
              "independently; these frozen policies supervise all later stages "
              "and are never retrained.")
    heading(doc, 2, "4.2  Constrained residual stage")
    para(doc, "The residual policy (PAC-RL) appends a two-layer head with "
              "2,177 parameters to each frozen supervisor. The applied "
              "coefficient is")
    equation(doc, "\u03b1\u209c = clip[0,1]( \u03b1\u0302\u209c + \u0394\u03b1\u209c ),   \u0394\u03b1\u209c = 0.1\u2009tanh(z\u209c),", 3)
    para(doc, "where z\u209c is the head output and the final layer is "
              "zero-initialized, so \u0394\u03b1 \u2261 0 before training. Training uses TD3 "
              "(twin critics, target smoothing, delayed policy updates) over "
              "closed-loop episodes with environment seeds disjoint from all "
              "evaluation seeds, warm-started with 2\u00d710\u2075 offline transitions "
              "collected under PAC-S, with a behavior-regularization term "
              "penalizing deviation from the supervisor. The reward combines "
              "position and heading error, command magnitude and rate, "
              "saturation, deadline flags, and constraint violations; all "
              "weights are frozen in a manifest before training. The learner "
              "emits \u0394\u03b1 only: it cannot command thrusters and cannot alter the "
              "MPC schedule or solver budget.")
    # ------------------------- 5 experimental design -------------------------
    heading(doc, 1, "5  Experimental Design")
    heading(doc, 2, "5.1  Registered protocol")
    para(doc, "All comparisons, seed partitions, and acceptance criteria were "
              "fixed and archived before residual training began. The "
              "evaluation grid comprises 120 paired episodes: 85 "
              "in-distribution (60 structured-current episodes under 20 "
              "held-out environment seeds, plus 25 episodes from five "
              "training-range disturbance families) and 35 "
              "out-of-distribution with respect to disturbance-family "
              "sampling (five test-range families, plus a fast time-scale "
              "current family and an estimation-delay family with sensor lag "
              "and noise). Each episode is shared across all methods: the "
              "same vehicle, current, and reference realization under every "
              "method and seed.")
    heading(doc, 2, "5.2  Methods and statistics")
    para(doc, "Eight methods are compared: the experts alone (SMC, MPC), a "
              "constant blend (Fixed \u03b1 = 0.5), a non-learning adaptive "
              "baseline searching window-level bias schedules by coordinate "
              "descent (SSPO), and the two PAC variants (PAC-S, PAC-RL) each "
              "instantiated "
              "with the same five frozen supervised seeds. Fixed methods "
              "contribute one combination each, giving 14 method-seed "
              "combinations and 1,680 rollouts (about 3.53 million control "
              "steps).")
    para(doc, "Primary inference is at the model-seed level: per-seed paired "
              "effects are summarized with a t-distribution interval with "
              "four degrees of freedom. As a supplementary check, "
              "episode-level bootstrap intervals (resampling the paired "
              "episodes with replacement, 10,000 resamples, generator seed 0) "
              "are reported for the headline contrasts. No comparisons beyond "
              "the registered set are subjected to significance claims, and "
              "no multiple-comparison correction is applied: the registered "
              "set is reported in full rather than selected from.")

    # ------------------------- 6 results -------------------------
    heading(doc, 1, "6  Results")
    heading(doc, 2, "6.1  Closed-loop tracking: how the vehicle actually moves")
    para(doc, "Figure 2 shows representative paired episodes\u2014an "
              "in-distribution sinusoidal current and the out-of-distribution "
              "actuator-degradation episode\u2014as three-dimensional paths, "
              "horizontal projections, and vertical error. In-distribution, "
              "all learning and blending methods follow the Lissajous "
              "reference closely; SMC lags with a visible offset. Under "
              "actuator degradation the character changes: the MPC expert "
              "alone distorts the horizontal loop and drifts in depth (0.886 "
              "m episode RMSE), PAC-S inherits part of that failure (0.206 m), "
              "and PAC-RL stays on the reference (0.088 m), slightly better "
              "than the never-adapting constant blend (0.124 m).")
    figure(doc, "fig2_closed_loop_trajectories",
           "Representative closed-loop trajectories. Columns: in-distribution "
           "sinusoidal current and out-of-distribution actuator degradation (the "
           "lowest-numbered held-out seed of each block, fixed before "
           "inspection); both episodes are shared across methods and "
           "regenerated deterministically from the frozen checkpoints. "
           "Rows: three-dimensional path, horizontal projection, vertical "
           "tracking error (dashed line, zero). The fixed blend is omitted "
           "for legibility; its values appear in Table 1.")
    para(doc, "Figure 3 resolves the same episodes in time. The position-error "
              "row shows the MPC divergence under degradation growing through "
              "the episode rather than spiking, consistent with a steadily "
              "wrong internal model rather than a transient. Heading error "
              "and pitch remain small for PAC-RL in both regimes, while the "
              "degraded MPC episode carries visible attitude excursions. "
              "Table 1 aggregates the full grid.")
    figure(doc, "fig3_error_attitude",
           "Time-resolved motion errors for the two episodes of Fig. 2. Rows: "
           "three-dimensional position error, absolute heading error, pitch "
           "angle. The MPC divergence under actuator degradation grows "
           "through the episode; PAC-RL remains close to the reference in "
           "both regimes. Single episodes; block-level aggregates appear in "
           "Table 1 and Fig. 5.")
    rows = []
    for m in METHOD_ORDER:
        rows.append([
            METHOD_NAME[m],
            f"{block_mean(m, 'seen', 'rmse_3d'):.3f}",
            f"{block_mean(m, 'unseen', 'rmse_3d'):.3f}",
            f"{block_mean(m, 'unseen', 'max_error'):.3f}",
            f"{block_mean(m, 'unseen', 'heading_rmse_deg'):.2f}",
            f"{block_mean(m, 'unseen', 'solver_deadline_miss_step_fraction'):.2f}",
            f"{block_mean(m, 'unseen', 'authority_alpha_mean'):.2f}",
        ])
    table(doc,
          ["Method", "ID RMSE (m)", "OOD RMSE (m)", "OOD max err (m)",
           "OOD heading (\u00b0)", "OOD deadline miss", "OOD mean \u03b1"],
          rows,
          "Paired evaluation over 120 episodes per method-seed combination "
          "(means over five model seeds for learned methods; ID, "
          "in-distribution; OOD, out-of-distribution with respect to "
          "disturbance-family sampling). Five-seed dispersion is reported as "
          "confidence intervals in Fig. 4a.",
          [1.35, 0.95, 0.95, 1.0, 0.95, 1.0, 0.85])
    heading(doc, 2, "6.2  Where the residual earns its value")
    para(doc, "Across the OOD block, PAC-RL reduces tracking RMSE from 0.161 m "
              "(PAC-S) to 0.116 m. The registered model-seed paired effect is "
              "\u22120.0445 m with a 95% t(4) interval of [\u22120.0737, \u22120.0154] m "
              "(28% relative reduction); the episode-level bootstrap over the "
              "35 OOD episodes is [\u22120.0759, \u22120.0167] m, consistent with the "
              "seed-level analysis. Maximum error falls from 0.495 m to "
              "0.414 m (five-seed means) and OOD heading error improves by "
              "0.73\u00b0.")
    para(doc, "The sharpest reading of Table 1 is unfavorable to us: a "
              "constant \u03b1 = 0.5 reaches 0.120 m OOD, within 3% of PAC-RL. That "
              "comparison was not registered and we make no significance "
              "claim; the honest summary is that the residual's overall "
              "advantage over never adapting is small, and its value is "
              "concentrated where adaptation is hardest (\u221228% versus the "
              "constant blend in the actuator-degradation family, Table 2). "
              "The in-distribution effect is +0.0028 m [+0.0009, +0.0047] "
              "(+0.4%) and in-distribution heading error rises by 0.32\u00b0 "
              "[+0.29, +0.35] (+4.3%).")
    para(doc, "The family decomposition in Fig. 4b locates the gain, and the "
              "registered paired effects in Fig. 4a quantify it. The SSPO "
              "versus PAC-RL contrast is not included in the archived "
              "paired-effects summary; it is recomputed from raw episode "
              "metrics with the same seed-level procedure: OOD \u22120.0198 m "
              "[\u22120.0211, \u22120.0185] in favor of PAC-RL, ID +0.0043 m "
              "[+0.0033, +0.0053] in favor of SSPO.")
    figure(doc, "fig4_closed_loop_results",
           "Registered paired effects and family decomposition. (a) RMSE "
           "paired effects with model-seed t(4) 95% intervals by block (OOD, "
           "ID); filled markers denote intervals excluding zero; the SSPO "
           "versus PAC-S contrast is as archived; the SSPO contrast is "
           "recomputed from raw episode metrics. (b) OOD RMSE by "
           "disturbance family and method. Gains concentrate in the "
           "actuator-degradation family; in the mild stochastic and "
           "estimation-delay families the supervised policy and SSPO remain "
           "marginally better. The panel does not support uniform superiority.")
    ad_rows = []
    base = fam_mean("v3_transformer", "unseen", "actuator_delay_noise")
    for m in METHOD_ORDER:
        ood = fam_mean(m, "unseen", "actuator_delay_noise")
        rel = "\u2014" if m == "v3_transformer" else f"{(ood / base - 1) * 100:+.0f}%"
        ad_rows.append([METHOD_NAME[m],
                        f"{fam_mean(m, 'seen', 'actuator_delay_noise'):.3f}",
                        f"{ood:.3f}", rel])
    table(doc,
          ["Method", "ID RMSE (m)", "OOD RMSE (m)", "vs PAC-S"],
          ad_rows,
          "Actuator-degradation family (response lag with measurement noise): "
          "paired family means.",
          [1.6, 1.2, 1.2, 1.1])
    heading(doc, 2, "6.3  How does the residual act?")
    para(doc, "The mechanism is visible in Fig. 5b. Within seconds of "
              "degraded-thruster onset, the supervised policy drives \u03b1 above "
              "0.9 and oscillates there for much of the episode (31.2% of "
              "steps above 0.9; standard deviation 0.331); the residual "
              "policy never exceeds 0.69 and holds [0.4, 0.6] on 97.0% of "
              "steps. The residual has not learned a new schedule\u2014it has "
              "learned to veto the supervisor's excursions. We read this as "
              "conservatism under distribution shift, a single-family "
              "observation we do not extrapolate.")
    figure(doc, "fig5_actuator_case",
           "Actuator-degradation episode (held-out seed 43000, fixed before "
           "inspection; family means superposed as text). (a) Position error. "
           "(b) Authority coefficient. Family-mean coefficients: 0.63 (PAC-S) "
           "versus 0.50 (PAC-RL).")
    heading(doc, 2, "6.4  The price of the gain")
    para(doc, "Two costs accompany the OOD gain. In-distribution position and "
              "heading error increase as quantified above. And the solver "
              "deadline-miss fraction\u2014steps on which the harness MPC instance "
              "exceeds the 10 ms period, measured identically for every "
              "method\u2014rises from a common 0.69 baseline to 0.80 under PAC-RL "
              "(+0.107 [+0.094, +0.120] OOD; +0.028 [+0.010, +0.047] ID). "
              "The constant blend leaves the metric unchanged, so the "
              "coefficient value itself is not the cause; a candidate "
              "explanation is trajectory-dependent solver conditioning, "
              "flagged as a hypothesis rather than a conclusion. Deployed "
              "safeguards are unaffected: fallback to the "
              "previous plan triggers on 0.3\u20130.5% of steps for all methods, "
              "success at the one-metre threshold is unchanged (0.988 ID / "
              "0.943 OOD), and OOD maximum error improves.")
    figure(doc, "fig6_cost_tradeoff",
           "Cost accounting. OOD RMSE versus solver deadline-miss fraction "
           "for every method-seed combination; the metric counts steps on "
           "which the shared-harness MPC instance exceeds the 10 ms period, "
           "collected identically for all methods. Shaded band: common "
           "baseline level (0.69; SMC / MPC / Fixed \u03b1 / SSPO).")
    para(doc, "Against the six registered criteria concerning the residual "
              "stage, three pass (gains not purchased with heading, "
              "saturation, or constraint violations; ablation "
              "separability; stable direction across five seeds), "
              "two fail (overall non-inferiority, through the in-distribution "
              "+0.4% effect; deadline-miss non-worsening), and one\u2014deployment "
              "inference time within the control period\u2014was not evaluated and "
              "is the first gate of the hardware stage. The criteria are "
              "translated verbatim from the registered Chinese original.")

    # ------------------------- 7 limitations -------------------------
    heading(doc, 1, "7  Limitations and Discussion")
    para(doc, "The central result is that a two-thousand-parameter residual, "
              "acting only on a bounded authority coefficient, converts a "
              "supervised arbitration policy that fails under distribution "
              "shift into one that matches a strong constant blend overall "
              "and exceeds it mainly where adaptation is hardest. The safety "
              "argument is architectural: zero initialization guarantees "
              "numerical equivalence to the deployed baseline at deployment, "
              "bounded output limits every perturbation, and fixed fallbacks "
              "bound the worst case by the baseline itself. This positions "
              "the contribution against residual reinforcement learning [5], "
              "which perturbs actuator commands, and learned safety filters "
              "[11], which place a learned component in the safety path.")
    para(doc, "Six boundaries qualify these claims. First, all evidence is "
              "simulation-level. Second, the feature set receives exact "
              "current information, which hardware must estimate. Third, "
              "out-of-distribution means disturbance-family sampling, not "
              "arbitrary field conditions. Fourth, in-distribution "
              "non-inferiority failed (+0.4% position, +4.3% heading). Fifth, "
              "the deadline-miss increase (+0.107) has no established cause. "
              "Sixth, deployment latency of the composite policy is "
              "unmeasured.")
    para(doc, "These boundaries define the next experiments: a second "
              "residual round with the final encoder block unfrozen under a "
              "registered rule targeting the in-distribution effect; "
              "isolation of the deadline mechanism; "
              "and the hardware protocol of Section 8.")

    # ------------------------- 8 hardware -------------------------
    heading(doc, 1, "8  Planned Hardware Validation")
    para(doc, "The paper is not submission-ready without physical "
              "experiments. The planned study deploys the same SMC, MPC "
              "expert, PAC-S, and PAC-RL checkpoints on the 10-kg "
              "six-thruster vehicle. The protocol includes at least ten "
              "independent trials per method and disturbance condition, "
              "randomized controller order, a fixed battery-voltage "
              "acceptance window, and identical trajectory and actuator "
              "limits. Disturbances progress from quiescent water to steady "
              "cross-flow, a repeatable step-like disturbance from a "
              "tethered pull or calibrated flow source, and\u2014specific to this "
              "paper's claims\u2014an injected actuator-degradation condition "
              "realized by a validated first-order command lag plus noise on "
              "a subset of thrusters.")
    para(doc, "The onboard estimator must provide pose, body velocity, and a "
              "causal current estimate from IMU, depth, DVL, and "
              "localization; the exact simulator current must not be "
              "exposed. Beyond position and attitude RMSE, the hardware "
              "section reports maximum error, success rate, measured "
              "electrical energy, per-step inference latency, end-to-end "
              "control-loop latency (the unevaluated criterion), saturation "
              "duration, and emergency interventions. Placeholder items "
              "(platform dimensions, sensor suite, test-site calibration, "
              "trial counts) must be filled with measured values only; they "
              "must not be replaced with simulated or expected data.")

    # ------------------------- 9 conclusion -------------------------
    heading(doc, 1, "9  Conclusion")
    para(doc, "PAC reframes learned underwater control as a trust question "
              "between structured controllers, and the residual stage "
              "repairs that trust under distribution shift: the OOD tracking "
              "error falls by 28% with intervals excluding zero at both seed "
              "and episode level, the degradation failure of the optimizer "
              "is confined to 0.090 m, and every cost is measured against "
              "criteria fixed before training. The learned component remains "
              "bounded, auditable as a single coefficient trajectory, and "
              "worst-case equivalent to the deployed baseline. Hardware "
              "validation, causal current estimation, latency measurement, "
              "and stability analysis of the time-varying blend remain "
              "required before any fielded-robustness claim.")

    heading(doc, 1, "Data Availability")
    para(doc, "All reported quantities derive from hash-frozen archives "
              "(configuration, dataset, and checkpoint digests with per-file "
              "checksums) covering the supervised and residual training stages "
              "and the paired evaluation. Source data for every figure and table are "
              "regenerated from these archives by the repository build "
              "scripts; the trajectory figures replay the archived episodes "
              "deterministically from the frozen checkpoints.")

    heading(doc, 1, "References")
    refs = [
        "T. I. Fossen, Handbook of Marine Craft Hydrodynamics and Motion Control, 2nd ed. Wiley, 2021.",
        "G. Antonelli, Underwater Robots: Motion and Force Control of Vehicle-Manipulator Systems, 3rd ed. Springer, 2014.",
        "E. F. Camacho and C. Bordons Alba, Model Predictive Control, 2nd ed. Springer, 2013.",
        "L. Hewing, K. P. Wabersich, M. Menner, and M. N. Zeilinger, \u201cLearning-based model predictive control: Toward safe learning in control,\u201d Annual Review of Control, Robotics, and Autonomous Systems, vol. 3, pp. 269\u2013296, 2020.",
        "T. Johannink et al., \u201cResidual reinforcement learning for robot control,\u201d in Proc. IEEE ICRA, 2019, pp. 6023\u20136029.",
        "A. Romero, Y. Song, and D. Scaramuzza, \u201cActor-critic model predictive control,\u201d in Proc. IEEE ICRA, 2024, pp. 14777\u201314784.",
        "K. Nguyen, S. Schoedel, A. Alavilli, B. Plancher, and Z. Manchester, \u201cTinyMPC: Model-predictive control on resource-constrained microcontrollers,\u201d in Proc. IEEE ICRA, 2024.",
        "L. Cai, K. Chang, and Y. Girdhar, \u201cLearning to swim: Reinforcement learning for 6-DOF control of thruster-driven autonomous underwater vehicles,\u201d in Proc. IEEE ICRA, 2025, pp. 11286\u201311293.",
        "X. Lin et al., \u201cUIVNAV: Underwater information-driven vision-based navigation via imitation learning,\u201d in Proc. IEEE ICRA, 2024, pp. 5250\u20135256.",
        "Y. Fan, H. Dong, X. Zhao, and P. Denissenko, \u201cPath-following control of unmanned underwater vehicle based on an improved TD3 deep reinforcement learning,\u201d IEEE Trans. Control Systems Technology, vol. 32, no. 5, pp. 1904\u20131919, 2024.",
        "O. So et al., \u201cHow to train your neural control barrier function: Learning safety filters for complex input-constrained systems,\u201d in Proc. IEEE ICRA, 2024, pp. 11532\u201311539.",
        "A. Vaswani et al., \u201cAttention is all you need,\u201d in Advances in Neural Information Processing Systems, vol. 30, 2017, pp. 5998\u20136008.",
        "L. Chen et al., \u201cDecision Transformer: Reinforcement learning via sequence modeling,\u201d in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 15084\u201315097.",
        "M. Janner, Q. Li, and S. Levine, \u201cOffline reinforcement learning as one big sequence modeling problem,\u201d in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 1273\u20131286.",
        "C. Chi et al., \u201cDiffusion policy: Visuomotor policy learning via action diffusion,\u201d in Robotics: Science and Systems, 2023.",
        "D. Liberzon, Switching in Systems and Control. Birkh\u00e4user, 2003.",
        "Z. Zhao, X. Liu, T. Wang, Z. Zhou, and M. Zhang, \u201cHybrid control scheme of nonlinear model prediction and adaptive terminal sliding mode for underwater vehicles based on threshold switching,\u201d ISA Transactions, vol. 176, pp. 580\u2013590, 2026.",
    ]
    for i, ref in enumerate(refs, 1):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.line_spacing = 1.3
        pf.space_after = Pt(3)
        pf.left_indent = Cm(0.75)
        pf.first_line_indent = Cm(-0.75)
        run = p.add_run(f"[{i}]  {ref}")
        style_run(run, size=9.5)

    doc.save(OUT)
    print(f"saved: {OUT}")
    print(f"figures: {FIG_N['n']}, tables: {TAB_N['n']}")


if __name__ == "__main__":
    build()
