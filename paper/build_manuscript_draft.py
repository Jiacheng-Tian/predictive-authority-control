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
    para(doc, "The ocean is warming and acidifying at rates that threaten "
              "marine ecosystems more severely than previously recognized [1], "
              "and autonomous underwater vehicles (AUVs) are the primary "
              "platform for the long-term observations needed to understand "
              "and mitigate these changes [2]. These vehicles must maintain "
              "precise trajectory tracking under ocean currents, "
              "parameter drift, and actuator degradation — conditions under "
              "which no single controller remains reliable. Sliding-mode "
              "control (SMC) is robust but coarse; model predictive control "
              "(MPC) is precise but collapses when its model is wrong — in "
              "our simulations, actuator degradation causes MPC to diverge "
              "to 0.822 m, nine times the nominal error and approaching the "
              "1 m mission-failure threshold. Direct reinforcement learning "
              "can adapt to such shifts but produces policies that cannot be "
              "certified for deployment. Here we resolve this safety-robustness "
              "dilemma by restricting what the learning system controls: "
              "instead of generating thruster commands, it learns only a "
              "single bounded trust coefficient that continuously blends the "
              "two verified controllers. A 14,113-parameter Transformer "
              "encodes 0.16 s of closed-loop history to predict this "
              "coefficient; a 2,177-parameter residual head, zero-initialized "
              "and bounded by 0.1 per step, then refines it through "
              "constrained reinforcement learning on the frozen backbone — "
              "guaranteeing that the deployed system is numerically identical "
              "to the verified baseline at startup and can only deviate "
              "within a bounded envelope. In a pre-registered paired "
              "evaluation of 1,680 rollouts across 10 disturbance families, "
              "the residual policy reduces out-of-distribution tracking error "
              "by 28% (0.161 m to 0.116 m), confines the actuator-degradation "
              "failure from 0.822 m to 0.090 m, and achieves the lowest "
              "saturation and control cost among all six methods — at 16,290 "
              "parameters (63.6 KB), The gains carry disclosed costs: "
              "in-distribution error rises by 4.4%, heading error by "
              "4.3%, and the solver deadline-miss fraction from 0.69 to 0.80.")
    rich(doc, [("This draft reports simulation evidence only; the planned "
                "hardware validation must be completed before submission.",
                {"bold": True})], after=12)

    heading(doc, 1, "1  Introduction")
    para(doc, "The ocean has absorbed more than 90% of the excess heat "
              "trapped by greenhouse gases, and the resulting warming and "
              "acidification are altering marine ecosystems more profoundly "
              "than previously recognized: a meta-analysis of hundreds of species "
              "reveals that combined warming and acidification affect twice "
              "as many biological traits as either stressor alone [1]. "
              "Understanding and mitigating these changes requires sustained, "
              "large-scale observation that exceeds the endurance, weather "
              "tolerance, and cost limits of ship-based campaigns [2]. "
              "Autonomous underwater vehicles are the established answer: "
              "a recent Science Robotics review argues for a paradigm shift toward "
              "autonomous robotic organizations capable of making ocean "
              "observations at large scale and low cost [2]. In both "
              "settings, the scientific value of each measurement depends "
              "on where the vehicle is when it is taken; tracking error "
              "misregisters observations, degrades image quality, narrows "
              "collision margins near fragile seabeds, and wastes limited "
              "propulsion energy.")
    para(doc, "Reliable tracking remains an unsolved control problem because "
              "the operating conditions are neither stationary nor fully "
              "known. Sliding-mode control rejects bounded disturbances "
              "through a transparent reaching-law structure [3], but its "
              "robustness is purchased with conservative gains and non-smooth "
              "action. Model predictive control anticipates motion and "
              "enforces actuator limits through online optimization [4,5], "
              "but its advantage evaporates when the internal model no "
              "longer represents the plant. Our simulations expose the "
              "severity of this failure: under actuator degradation, MPC "
              "alone diverges to 0.822 m — nearly an order of magnitude beyond nominal error, close to the 1 m threshold at which a mission is judged failedfailure threshold — because "
              "the optimizer acts confidently on wrong premises. Hybrid "
              "schemes that switch between the two controllers under "
              "fixed thresholds [8] cannot adapt to conditions that vary "
              "continuously within a single control step.")
    para(doc, "Adaptation to shifting conditions requires learning. "
              "Machine learning, however, introduces a verification problem that the underwater "
              "community has not resolved. End-to-end deep reinforcement "
              "learning has demonstrated strong results in six-degree-of-"
              "freedom AUV control [7,8], but the learned policy outputs "
              "six-dimensional thruster commands whose worst-case behavior "
              "cannot be bounded, audited, or certified. Residual "
              "reinforcement learning [11,12] superposes a learned correction "
              "on a scripted controller, reducing the learning burden but "
              "still perturbing the full actuation space without a safety "
              "envelope. Supervised learning — including our own prior "
              "work on temporal authority prediction — can imitate an "
              "oracle only where the oracle is well-defined (deterministic "
              "environments); under distribution shift, the learned "
              "coefficient degrades below a non-adaptive constant blend, "
              "because the training labels carry no information about "
              "conditions the oracle never encountered.")
    para(doc, "This paper resolves the safety-robustness dilemma by "
              "restricting what the learning system is allowed to control. "
              "Instead of generating thruster commands, the learned policy "
              "outputs a single scalar \u2014 the authority coefficient "
              "\u2014 that continuously blends the two verified controllers. "
              "The controllers remain explicit and inspectable; the learned "
              "output is one bounded number per step. We further constrain "
              "the adaptation through a zero-initialized residual head, "
              "bounded to perturb the coefficient by at most 0.1 per "
              "control step, so that the deployed system starts exactly at "
              "the verified baseline and can only deviate within a "
              "bounded envelope. The resulting architecture occupies "
              "63.6 KB and runs at 100 Hz — small enough for embedded "
              "deployment on the vehicle itself.")
    para(doc, "In a pre-registered paired evaluation over 1,680 rollouts "
              "spanning 10 disturbance families, the constrained residual "
              "policy reduces out-of-distribution tracking error by 28% "
              "(0.161 m to 0.116 m, 95% CI [\u22120.074, \u22120.015] m), "
              "confines the actuator-degradation catastrophe from 0.822 m "
              "to 0.090 m, and achieves the lowest saturation fraction and "
              "control cost among all six methods. The gains carry disclosed "
              "costs: in-distribution error rises by 4.4% and the solver "
              "deadline-miss fraction from 0.69 to 0.80. This study makes "
              "four contributions:")
    rich(doc, [("(1) ", {"bold": True}),
               ("Continuous temporal authority allocation between two "
                "complete controllers, reducing the learned action space "
                "to one bounded variable that is directly auditable.", {})])
    rich(doc, [("(2) ", {"bold": True}),
               ("A constrained residual reinforcement learning procedure "
                "on the frozen backbone, with the residual bounded by "
                "$|\\Delta\\alpha_t| \\leq 0.1$ and zero-initialized, so the "
                "worst-case behavior equals the deployed baseline.", {})])
    rich(doc, [("(3) ", {"bold": True}),
               ("A pre-registered paired evaluation protocol covering 10 "
                "disturbance families, six methods, and 1,680 rollouts with "
                "seed-level confidence intervals.", {})])
    rich(doc, [("(4) ", {"bold": True}),
               ("An extremely compact policy (16,290 parameters, 63.6 KB, "
                "0.23M MACs at 100 Hz) that fits resource-constrained "
                "embedded processors.", {})])

    heading(doc, 1, "2  Related Work")
    heading(doc, 2, "2.1  Underwater vehicle control")
    para(doc, "Marine vehicle control is developed from rigid-body dynamics "
              "with added mass, damping, and restoring forces [3,4]. Hybrid "
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
    para(doc, "Self-attention [17] and sequence-modeling formulations of "
              "control [18,19] scale with data and width. The PAC encoder "
              "deliberately occupies the opposite regime: one encoder layer "
              "over a 16-step history mapping to a single coefficient, which "
              "suffices for the arbitration interface and keeps edge deployment "
              "plausible. Diffusion policies [20] and imitation navigation [9] "
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
              "learning components is therefore the baseline policy itself. Figure 1 "
              "shows the complete architecture; Figure 2 details the network "
              "and two-phase training. Figure 2 "
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
           "residual head with TD3 [13], behavior regularization, and a "
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
    para(doc, "Six methods are compared: the experts alone (SMC, MPC), a "
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
           "Time-resolved motion errors for the two episodes of Fig. 3. Rows: "
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
           "The losses are regressions onto a moving TD target that is "
           "recomputed from tracking networks at every update, so their "
           "magnitudes track the growing value scale of that target rather "
           "than divergence; the actor loss is dominated by the negative "
           "action value, whose scale inflates with the critic. Greedy "
           "closed-loop performance is the registered "
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
              "(+4.4%) and in-distribution heading error rises by 0.32\u00b0 "
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
              "+4.4% effect; deadline-miss non-worsening), and one\u2014deployment "
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
              "non-inferiority failed (+4.4% position, +4.3% heading). Fifth, "
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

    # ------------------------- 9 future directions -------------------------
    heading(doc, 1, "9  Future Directions")
    para(doc, "The immediate extension is a second residual round under a "
              "registered rule: unfreeze the final encoder block, target the "
              "in-distribution regressions (+4.4% position, +4.3% heading), and "
              "decide in advance that both rounds are reported whichever way the "
              "comparison ends. The bounded-residual interface makes this an "
              "incremental step rather than a redesign—the worst case of the "
              "new round is still the deployed baseline of the current one.")
    para(doc, "The world-model route has concrete, separable levers. "
              "Calibration comes first, because a rank-valid but miscalibrated "
              "signal (14.5% and 32.8% empirical coverage at 68% and 95% nominal) "
              "cannot be gated safely; conformal or ensemble-recalibration "
              "methods apply directly. Gate policy and refresh interval come "
              "second: the current gate opens on only 4–5% of steps, so raising "
              "engagement where advice can change outcomes is a design variable, "
              "not a property of the model. Joint training comes third, letting "
              "world-model predictions participate in the residual objective "
              "rather than entering only as input features. The promotion rule is "
              "fixed in advance: a closed-loop gain under the registered "
              "evaluation promotes the world model into the method; otherwise it "
              "remains an ablation.")
    para(doc, "The hardware campaign of Section 8 executes the protocol as "
              "written, with two additions informed by this study: the injected "
              "actuator-degradation condition is the primary OOD test, and the "
              "causal current estimator replaces the privileged simulator input "
              "so that authority decisions are made on the same information a "
              "fielded vehicle would have. Before the campaign, an isolation "
              "experiment should separate the two candidate mechanisms behind "
              "the deadline-miss increase—trajectory-dependent solver "
              "conditioning and advisory-computation wall-time—because the "
              "answer determines what the hardware logging must instrument.")
    para(doc, "Two further directions are open. On theory, the time-varying "
              "convex blend lacks a formal stability argument; a common-Lyapunov "
              "or dwell-time and rate condition for the switched structure would "
              "complement the architectural safety argument with an analytical "
              "one. On generality, the authority interface extends beyond two "
              "experts—per-axis coefficients, larger expert sets, and experts "
              "added or removed at run time—and the certification structure "
              "developed here (zero initialization, bounded residual, fixed "
              "fallbacks) is a candidate template for other learned components "
              "in safety-relevant loops.")

    heading(doc, 1, "10  Conclusion")
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
        "K. Alter, J. Jacquemont, J. Claudet, et al., “Hidden impacts of ocean warming and acidification on biological responses of marine animals revealed through meta-analysis,” Nature Communications, vol. 15, art. 2885, 2024.",
        "K. Skaugset, J. Borges de Sousa, and A. J. Sørensen, “Autonomous robotic organizations for marine operations,” Science Robotics, vol. 10, no. 100, eadl2976, 2025.",
        "T. I. Fossen, Handbook of Marine Craft Hydrodynamics and Motion Control, 2nd ed. Wiley, 2021.",
        "G. Antonelli, Underwater Robots: Motion and Force Control of Vehicle-Manipulator Systems, 3rd ed. Springer, 2014.",
        "V. I. Utkin, Sliding Modes in Control and Optimization. Springer, 1992.",
        "E. F. Camacho and C. Bordons Alba, Model Predictive Control, 2nd ed. Springer, 2013.",
        "L. Hewing, K. P. Wabersich, M. Menner, and M. N. Zeilinger, “Learning-based model predictive control: Toward safe learning in control,” Annual Review of Control, Robotics, and Autonomous Systems, vol. 3, pp. 269–296, 2020.",
        "Z. Zhao, X. Liu, T. Wang, Z. Zhou, and M. Zhang, “Hybrid control scheme of nonlinear model prediction and adaptive terminal sliding mode for underwater vehicles based on threshold switching,” ISA Transactions, vol. 176, pp. 580–590, 2026.",
        "Y. Fan, H. Dong, X. Zhao, and P. Denissenko, “Path-following control of unmanned underwater vehicle based on an improved TD3 deep reinforcement learning,” IEEE Transactions on Control Systems Technology, vol. 32, no. 5, pp. 1904–1919, 2024.",
        "L. Cai, K. Chang, and Y. Girdhar, “Learning to swim: Reinforcement learning for 6-DOF control of thruster-driven autonomous underwater vehicles,” in Proc. IEEE ICRA, 2025, pp. 11286–11293.",
        "T. Johannink, S. Bahl, A. Nair, et al., “Residual reinforcement learning for robot control,” in Proc. IEEE ICRA, 2019, pp. 6023–6029.",
        "T. Silver, K. Allen, A. Tenenbaum, and J. Koltun, “Residual policy learning,” arXiv preprint arXiv:1812.06298, 2018.",
        "S. Fujimoto, H. van Hoof, and D. Meger, “Addressing function approximation error in actor-critic methods,” in Proc. ICML, 2018, pp. 1587–1596.",
        "A. Romero, Y. Song, and D. Scaramuzza, “Actor-critic model predictive control,” in Proc. IEEE ICRA, 2024, pp. 14777–14784.",
        "K. Nguyen, S. Schoedel, A. Alavilli, B. Plancher, and Z. Manchester, “TinyMPC: Model-predictive control on resource-constrained microcontrollers,” in Proc. IEEE ICRA, 2024.",
        "O. So, Z. Serlin, M. Mann, et al., “How to train your neural control barrier function: Learning safety filters for complex input-constrained systems,” in Proc. IEEE ICRA, 2024, pp. 11532–11539.",
        "A. Vaswani, N. Shazeer, N. Parmar, et al., “Attention is all you need,” in Advances in Neural Information Processing Systems, vol. 30, 2017, pp. 5998–6008.",
        "L. Chen, K. Lu, A. Rajeswaran, et al., “Decision Transformer: Reinforcement learning via sequence modeling,” in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 15084–15097.",
        "M. Janner, Q. Li, and S. Levine, “Offline reinforcement learning as one big sequence modeling problem,” in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 1273–1286.",
        "C. Chi, S. Feng, Y. Du, et al., “Diffusion policy: Visuomotor policy learning via action diffusion,” in Robotics: Science and Systems, 2023.",
        "D. Liberzon, Switching in Systems and Control. Birkhäuser, 2003.",
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
