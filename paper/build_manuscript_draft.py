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

from latex_math import add_math_run, math_paragraph

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
               "sspo": "SSPO", "v3_transformer": "Supervised PAC",
               "residual_rl": "Residual PAC"}

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
    """body is a LaTeX string; rendered as native Word equation."""
    return math_paragraph(doc, body, number)


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
    para(doc, "Classical underwater controllers provide useful structure but "
              "exhibit complementary failure modes under changing currents: "
              "robust feedback can be conservative, whereas a predictive "
              "controller can be accurate in favorable regimes yet degrade "
              "under model mismatch. We introduce Predictive Authority "
              "Control (PAC), a compact Transformer policy that does not "
              "replace these controllers. Instead, it predicts a bounded "
              "scalar authority coefficient from a short state\u2013controller "
              "history and continuously blends the outputs of a sliding-mode "
              "controller and a constrained model predictive controller. "
              "Training uses supervised predictive initialization from a "
              "short-horizon rollout oracle, followed by constrained "
              "residual reinforcement learning on the frozen backbone. In a "
              "six-degree-of-freedom simulation of a 10-kg, six-thruster "
              "autonomous underwater vehicle, five independently trained "
              "residual policies are evaluated over 1,680 paired rollouts "
              "spanning in-distribution and out-of-distribution disturbance "
              "families. The residual policy obtains an out-of-distribution "
              "3-D position RMSE of $0.1160\\pm0.0011$ m, compared with "
              "$0.1605\\pm0.0239$ m for the supervised baseline, "
              "$0.2482$ m for sliding-mode control, and $0.2213$ m for the "
              "predictive baseline, corresponding to reductions of 27.8\%, "
              "53.3\%, and 47.6\%, respectively. The residual policy also "
              "reduces mean maximum error by 31.1\% and 28.0\% relative "
              "to SMC and MPC. The gains are not free: the residual policy "
              "uses 4.4\% more in-distribution position error and 4.3\% "
              "more heading error than the supervised baseline, and its "
              "solver deadline-miss fraction rises from 0.69 to 0.80 "
              "out-of-distribution. These results support learned temporal "
              "authority allocation as an interpretable alternative to "
              "direct neural control.")
    rich(doc, [("This draft reports simulation evidence only; the planned "
                "hardware validation must be completed before submission.",
                {"bold": True})], after=12)

    # ------------------------- 1 introduction -------------------------
    heading(doc, 1, "1  Introduction")
    para(doc, "Accurate trajectory tracking is central to autonomous "
              "underwater vehicle (AUV) inspection, intervention, and "
              "sampling. The task remains difficult because hydrodynamic "
              "coefficients are uncertain, environmental currents vary "
              "over time, and thruster saturation couples tracking quality "
              "to control effort [1,2]. Sliding-mode control (SMC) can "
              "reject bounded disturbances with a transparent feedback "
              "structure, but its robustness is commonly purchased through "
              "conservative gains or non-smooth control action. Model "
              "predictive control (MPC) incorporates prediction and "
              "constraints, yet its performance depends on model fidelity "
              "and real-time optimization [3,4].")
    para(doc, "Learning-based control offers another route. Recent work "
              "has demonstrated full six-degree-of-freedom learned AUV "
              "control and zero-shot sim-to-real transfer [8], while "
              "imitation learning has enabled information-driven "
              "underwater navigation [9]. More broadly, residual "
              "reinforcement learning combines a fixed controller with a "
              "learned additive correction [5], and actor\u2013critic MPC "
              "embeds a differentiable optimizer inside a learned policy "
              "[6]. These methods establish the value of retaining control "
              "structure, but they do not directly answer a simpler "
              "deployment question: given two complete and complementary "
              "controllers, when should the robot trust each one, and by "
              "how much?")
    para(doc, "We study this question through authority allocation. "
              "Rather than learning a six-dimensional thruster command, "
              "PAC predicts one scalar $\\alpha_t\\in[0,1]$ and applies")
    equation(doc,
             r"\boldsymbol{u}_t=(1-\alpha_t)\boldsymbol{u}^{\mathrm{SMC}}_t"
             r"+\alpha_t\boldsymbol{u}^{\mathrm{MPC}}_t", 1)

    para(doc, "where $\\vect{u}^{\\mathrm{MPC}}_t$ is the "
              "predictive-controller command. The two controllers remain "
              "explicit, the learned output is bounded and interpretable, "
              "and a rate limiter prevents abrupt authority transfer. A "
              "Transformer encodes a short history because controller "
              "preference depends not only on instantaneous error but "
              "also on whether that error is growing, recovering, or "
              "coincident with actuator saturation. We further append a "
              "zero-initialized residual head that outputs "
              "$\\Delta\\alpha_t = 0.1\\,\\tanh(z_t)$, so the "
              "composite policy is numerically identical to the deployed "
              "baseline at initialization and can only revise trust "
              "within a bounded envelope thereafter.")
    para(doc, "This paper makes three contributions:")
    rich(doc, [("Contribution 1. ", {"bold": True}),
               ("We formulate controller selection as continuous, "
                "history-conditioned authority prediction, retaining two "
                "structured controllers while reducing the learned action "
                "space to one bounded variable.", {})])
    rich(doc, [("Contribution 2. ", {"bold": True}),
               ("We develop a two-stage learning procedure: supervised "
                "predictive initialization from a short-horizon oracle, "
                "followed by constrained residual reinforcement learning "
                "on the frozen backbone, with the residual output bounded "
                "by $|\\Delta\\alpha_t| \\leq 0.1$ and zero-initialized.", {})])
    rich(doc, [("Contribution 3. ", {"bold": True}),
               ("We evaluate five independently trained policies over "
                "1,680 paired rollouts spanning in-distribution and "
                "out-of-distribution disturbance families, and report "
                "both improvements and failure cases, including "
                "scenario-wise RMSE, peak error, heading error, control "
                "effort, saturation, and solver deadline-miss fraction.",
                {})])

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
              "learning components is therefore the baseline policy itself. Figure 2 "
              "details the two networks and the two-phase training procedure: the "
              "backbone is frozen throughout phase 2, and the residual head is "
              "zero-initialized, so the composite policy starts exactly at the "
              "supervised baseline.")
    figure(doc, "fig1_system_architecture_ai",
           "Predictive authority control architecture. (a) Closed loop: the "
           "experts generate commands blended by \u03b1; a frozen temporal encoder "
           "predicts the supervised coefficient and a bounded, zero-initialized "
           "residual head revises it. (b) Layered safety filter in execution order. "
           "(c) Two-phase training.")
    figure(doc, "fig_network_training_ai",
           "Network architectures and two-phase training. (a) The frozen "
           "temporal encoder (14,113 parameters) maps a 16\u00d724 history "
           "through a 32-dimensional embedding and one Transformer encoder "
           "layer (four-head attention, 128-unit feed-forward) to the "
           "supervised authority; the trainable residual head (two linear "
           "layers, 2,177 parameters, zero-initialized final layer) revises "
           "it by at most 0.1 per step. (b) Phase 1 trains the backbone by "
           "imitation of the short-horizon oracle; phase 2 trains only the "
           "residual head with TD3, behavior regularization, and a "
           "warm-start replay.")

    # ------------------------- 4 learning -------------------------
    heading(doc, 1, "4  Learning the Authority Residual")
    heading(doc, 2, "4.1  Supervised initialization")
    para(doc, "The supervised PAC policy is trained by imitation of a "
              "short-horizon oracle: for each training state, candidate "
              "coefficients from a five-point grid are held for ten steps with "
              "the current frozen, and the label minimizes")
    equation(doc,
             r"J(\alpha) = \|\boldsymbol{e}_{xy}\|^2 + 0.7\,e_z^2 + 0.08\,e_\psi^2"
             r"\, + 0.02\,\mathrm{mean}([\,|\boldsymbol{u}|-0.92\,]_+^2)"
             r"\, + 0.01\,\mathrm{mean}((\boldsymbol{u}-\boldsymbol{u}_{t-1})^2)", 2)
    para(doc, "a cost combining horizontal and vertical position error, heading "
              "error, near-saturation magnitude, and command change. Training "
              "uses 18,900 labeled samples, AdamW for 180 epochs, and a "
              "weighted squared loss. Five model seeds are trained "
              "independently; these frozen policies supervise all later stages "
              "and are never retrained.")
    heading(doc, 2, "4.2  Constrained residual stage")
    para(doc, "The residual policy appends a two-layer head with "
              "2,177 parameters to each frozen supervisor. The applied "
              "coefficient is")
    equation(doc,
             r"\alpha_t = \mathrm{clip}_{[0,1]}(\hat{\alpha}_t+\Delta\alpha_t),\quad\Delta\alpha_t = 0.1\,\tanh(z_t)", 3)
    para(doc, "where z\u209c is the head output and the final layer is "
              "zero-initialized, so \u0394\u03b1 \u2261 0 before training. Training uses TD3 "
              "(twin critics, target smoothing, delayed policy updates) over "
              "closed-loop episodes with environment seeds disjoint from all "
              "evaluation seeds, warm-started with 2\u00d710\u2075 offline transitions "
              "collected under the supervised policy, with a behavior-regularization term "
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
              "descent (SSPO, a coordinate-search ablation rather than an external "
              "baseline), and the two PAC variants (supervised and residual) each "
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
    para(doc, "Figure 3 shows representative paired episodes\u2014an "
              "in-distribution sinusoidal current and the out-of-distribution "
              "actuator-degradation episode\u2014as three-dimensional paths, "
              "horizontal projections, and vertical error. In-distribution, "
              "all learning and blending methods follow the Lissajous "
              "reference closely; SMC lags with a visible offset. Under "
              "actuator degradation the character changes: the MPC expert "
              "alone distorts the horizontal loop and drifts in depth (0.886 "
              "m episode RMSE), the supervised policy inherits part of that "
              "failure (0.206 m), and the residual policy stays on the "
              "reference (0.088 m), slightly better "
              "than the never-adapting constant blend (0.124 m).")
    figure(doc, "fig2_closed_loop_trajectories",
           "Representative closed-loop trajectories. Columns: in-distribution "
           "sinusoidal current and out-of-distribution actuator degradation (the "
           "lowest-numbered held-out seed of each block, fixed before "
           "inspection); both episodes are shared across methods and "
           "regenerated deterministically from the frozen checkpoints. "
           "Rows: three-dimensional path, horizontal projection, z-axis "
           "tracking error (dashed line, zero). The fixed blend is omitted "
           "for legibility; its values appear in Table 1.")
    para(doc, "Figure 4 resolves the same episodes in time. The position-error "
              "row shows the MPC divergence under degradation growing through "
              "the episode rather than spiking, consistent with a steadily "
              "wrong internal model rather than a transient. Heading error "
              "and pitch remain small for residual PAC in both regimes, while the "
              "degraded MPC episode carries visible attitude excursions. "
              "Table 1 aggregates the full grid.")
    figure(doc, "fig3_error_attitude",
           "Time-resolved motion errors for the two episodes of Fig. 2. Rows: "
           "three-dimensional position error, absolute heading error, pitch "
           "angle. The MPC divergence under actuator degradation grows "
           "through the episode; residual PAC remains close to the reference in "
           "both regimes. Single episodes; block-level aggregates appear in "
           "Table 1 and Fig. 7.")
    para(doc, "Residual training is stable across seeds. Figure 5 shows the "
              "per-seed mean episode reward during phase 2, collected under "
              "exploration noise on the rotating training-family schedule; "
              "the trend therefore reflects the changing episode mix rather "
              "than greedy-policy performance, and no seed diverges. All "
              "five seeds yield policies with consistent closed-loop "
              "behavior in the registered evaluation (Fig. 6, Table 1), "
              "which is the performance measure of record.")
    figure(doc, "fig_training_curves",
           "Residual training across the five model seeds. (a) Episode "
           "reward, (b) critic loss, (c) actor loss. Thin lines: "
           "per-seed values; bold line and shaded band: five-seed "
           "mean $\\pm$ 1 sd. All metrics are collected under "
           "exploration noise on the rotating training-family "
           "schedule, so trends reflect the changing episode mix "
           "rather than greedy-policy performance; no seed diverges. "
           "Greedy closed-loop performance is the registered "
           "evaluation of Fig. 6 and Table 1.")
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
          "confidence intervals in Fig. 6a.",
          [1.35, 0.95, 0.95, 1.0, 0.95, 1.0, 0.85])
    heading(doc, 2, "6.2  Where the residual earns its value")
    para(doc, "Across the OOD block, the residual policy reduces tracking RMSE "
              "from 0.161 m (supervised) to 0.116 m. The registered model-seed paired effect is "
              "\u22120.0445 m with a 95% t(4) interval of [\u22120.0737, \u22120.0154] m "
              "(28% relative reduction); the episode-level bootstrap over the "
              "35 OOD episodes is [\u22120.0759, \u22120.0167] m, consistent with the "
              "seed-level analysis. Maximum error falls from 0.495 m to "
              "0.414 m (five-seed means) and OOD heading error improves by "
              "0.73\u00b0.")
    para(doc, "The sharpest reading of Table 1 is unfavorable to us: a "
              "constant \u03b1 = 0.5 reaches 0.120 m OOD, within 3% of the residual "
              "policy. That "
              "comparison was not registered and we make no significance "
              "claim; the honest summary is that the residual's overall "
              "advantage over never adapting is small, and its value is "
              "concentrated where adaptation is hardest (\u221228% versus the "
              "constant blend in the actuator-degradation family, Table 2). "
              "The in-distribution effect is +0.0028 m [+0.0009, +0.0047] "
              "(+0.4%) and in-distribution heading error rises by 0.32\u00b0 "
              "[+0.29, +0.35] (+4.3%).")
    para(doc, "The family decomposition in Fig. 6b locates the gain, and the "
              "registered paired effects in Fig. 6a quantify it. The SSPO "
              "versus residual PAC contrast is not included in the archived "
              "paired-effects summary; it is recomputed from raw episode "
              "metrics with the same seed-level procedure: OOD \u22120.0198 m "
              "[\u22120.0211, \u22120.0185] in favor of the residual policy, ID +0.0043 m "
              "[+0.0033, +0.0053] in favor of SSPO.")
    figure(doc, "fig4_closed_loop_results",
           "Registered paired effects and family decomposition. (a) RMSE "
           "paired effects with model-seed t(4) 95% intervals by block (OOD, "
           "ID); filled markers denote intervals excluding zero; the SSPO "
           "versus supervised contrast is as archived; the SSPO contrast is "
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
          ["Method", "ID RMSE (m)", "OOD RMSE (m)", "vs supervised PAC"],
          ad_rows,
          "Actuator-degradation family (response lag with measurement noise): "
          "paired family means.",
          [1.6, 1.2, 1.2, 1.1])
    heading(doc, 2, "6.3  How does the residual act?")
    para(doc, "The mechanism is visible in Fig. 7b. Within seconds of "
              "degraded-thruster onset, the supervised policy drives \u03b1 above "
              "0.9 and oscillates there for much of the episode (31.2% of "
              "steps above 0.9; standard deviation 0.331); the residual "
              "policy never exceeds 0.69 and holds [0.4, 0.6] on 97.0% of "
              "steps. The residual has not learned a new schedule\u2014it has "
              "learned to veto the supervisor's excursions. We read this as "
              "conservatism under distribution shift, a single-family "
              "observation we do not extrapolate.")
    para(doc, "The stabilization generalizes beyond the single episode: "
              "across all 35 OOD episodes and five seeds (175 episodes per "
              "method), the per-episode mean authority is 0.82 \u00b1 0.14 "
              "for the supervised policy (10th\u201390th percentile 0.62\u20130.93) "
              "and 0.54 \u00b1 0.03 for the residual policy, with every residual "
              "episode inside [0.48, 0.62] (Fig. 7c).")
    figure(doc, "fig5_actuator_case",
           "Actuator-degradation episode (held-out seed 43000, fixed before "
           "inspection; family means superposed as text). (a) Position error. "
           "(b) Authority coefficient in the representative episode. "
           "(c) Box plots of per-episode mean authority over all 35 OOD "
           "episodes and five seeds (175 points per method; boxes: "
           "IQR, whiskers: range, line: median). Family-mean "
           "coefficients: 0.63 (supervised) versus 0.50 (residual).")
    heading(doc, 2, "6.4  Control quality: the gain is not purchased with actuator abuse")
    para(doc, "On the OOD block the residual policy attains the lowest "
              "saturation-step fraction of all six methods (0.126, versus "
              "0.134 for the fixed blend and 0.168 for the supervised "
              "policy), the lowest applied control cost (31.9 versus 33.7), "
              "and a rate-limit activation of 0.039 below both experts. The "
              "paired effect on control cost is \u22121.78 [\u22122.31, \u22121.26] OOD "
              "and \u22120.52 [\u22120.56, \u22120.48] ID: the residual policy is in fact "
              "cheaper to run than its supervisor while tracking better "
              "OOD. This substantiates the registered criterion that gains "
              "must not be purchased with heading, saturation, or "
              "constraint degradation (Fig. 8).")
    figure(doc, "fig_control_quality",
           "Control quality by method and block (bars: five-seed means "
           "$\\pm$ sd). (a) Applied control cost. (b) Saturation-step "
           "fraction. (c) Rate-limit activation. The residual policy "
           "attains the lowest OOD saturation and control cost of "
           "all methods.")
    heading(doc, 2, "6.5  The price of the gain")
    para(doc, "Two costs accompany the OOD gain. In-distribution position and "
              "heading error increase as quantified above. And the solver "
              "deadline-miss fraction\u2014steps on which the harness MPC instance "
              "exceeds the 10 ms period, measured identically for every "
              "method\u2014rises from a common 0.69 baseline to 0.80 under the residual policy "
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
    table(doc,
          ["Metric", "ID effect [95% CI]", "OOD effect [95% CI]"],
          [["RMSE (m)", "+0.0028 [+0.0009, +0.0047]",
            "\u22120.0445 [\u22120.0737, \u22120.0154]"],
           ["Heading error (deg)", "+0.32 [+0.29, +0.35]",
            "\u22120.73 [\u22121.41, \u22120.06]"],
           ["Applied control cost", "\u22120.52 [\u22120.56, \u22120.48]",
            "\u22121.78 [\u22122.31, \u22121.26]"],
           ["Deadline-miss fraction", "+0.028 [+0.010, +0.047]",
            "+0.107 [+0.094, +0.120]"]],
          "Registered paired effects of the residual policy versus the "
          "supervised baseline across all reported metrics (model-seed "
          "t(4) 95% intervals). The SSPO contrast, recomputed from raw "
          "episode metrics, is RMSE OOD \u22120.0198 [\u22120.0211, \u22120.0185] in "
          "favor of the residual policy and ID +0.0043 [+0.0033, +0.0053] "
          "in favor of SSPO.",
          [2.2, 2.2, 2.2])
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
    para(doc, "These boundaries define the registered next experiments. "
              "First, a direct-RL arm\u2014an end-to-end TD3 policy emitting "
              "thruster commands on the same grid\u2014completes the comparison "
              "hierarchy that residual-RL studies expect. Second, a "
              "constant-authority sweep over \u03b1 \u2208 {0, 0.25, 0.5, 0.75, 1} "
              "exposes the trust landscape that the policy navigates. Both "
              "require new registered runs. Third, a second residual round "
              "with the final encoder block unfrozen targets the "
              "in-distribution effect. Fourth, isolation of the deadline "
              "mechanism. Fifth, the hardware protocol of Section 8, whose "
              "first gate is deployment latency.")

    # ------------------------- 8 hardware -------------------------
    heading(doc, 1, "8  Planned Hardware Validation")
    para(doc, "The paper is not submission-ready without physical "
              "experiments. The planned study deploys the same SMC, MPC "
              "expert, supervised PAC, and residual PAC checkpoints on the 10-kg "
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

    postprocess_inline_math(doc)
    doc.save(OUT)
    print(f"saved: {OUT}")
    print(f"figures: {FIG_N['n']}, tables: {TAB_N['n']}")




def postprocess_inline_math(doc):
    """Scan all paragraphs; convert $...$ LaTeX to native math runs."""
    from docx.oxml.ns import qn
    import re as _re
    for p in doc.paragraphs:
        full = p.text
        if '$' not in full:
            continue
        parts = _re.split(r'(\$[^$]+\$)', full)
        if len(parts) <= 1:
            continue
        # clear existing runs
        for r in list(p.runs):
            r._r.getparent().remove(r._r)
        for part in parts:
            if part.startswith('$') and part.endswith('$') and len(part) > 2:
                latex = part[1:-1]
                try:
                    add_math_run(p, latex, display=False)
                except Exception:
                    run = p.add_run(part)
                    style_run(run)
            else:
                if part:
                    run = p.add_run(part)
                    style_run(run)


if __name__ == "__main__":
    build()
