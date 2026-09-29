# -*- coding: utf-8 -*-
"""Chinese edition (mirrors build_manuscript_draft.py).

One question carried through the paper; motion-performance figures first;
numbered limitations; planned-hardware section; displayed equations;
self-limiting captions. Table numbers computed live from the frozen archives.
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
OUT = ROOT / "paper" / "PAC_manuscript_draft_CN_v2.docx"
STAGE2 = ROOT / "results" / "formal_v4" / "stage2-residual-rl-2026-09-23"

BLACK = RGBColor(0, 0, 0)
GREY = RGBColor(0x60, 0x60, 0x60)

ov = pd.read_csv(STAGE2 / "formal" / "overall_summary.csv")
bf = pd.read_csv(STAGE2 / "formal" / "block_family_summary.csv")

METHOD_ORDER = ["smc", "mpc", "constant_alpha", "sspo", "v3_transformer",
                "residual_rl"]
METHOD_NAME = {"smc": "SMC", "mpc": "MPC", "constant_alpha": "固定混合 \u03b1=0.5",
               "sspo": "SSPO", "v3_transformer": "监督 PAC",
               "residual_rl": "残差 PAC"}

FIG_N = {"n": 0}
TAB_N = {"n": 0}


def block_mean(method, block, col):
    return ov[(ov["method"] == method) & (ov["block"] == block)][col].mean()


def fam_mean(method, block, family, col="rmse_3d"):
    return bf[(bf["method"] == method) & (bf["block"] == block)
              & (bf["family"] == family)][col].mean()


def style_run(run, size=10.5, bold=False, italic=False, color=BLACK,
              cn="宋体", en="Times New Roman"):
    run.font.name = en
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    rf.set(qn("w:eastAsia"), cn)


def para(doc, text="", size=10.5, bold=False, italic=False, cn="宋体",
         align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=6, color=BLACK, indent=True):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.line_spacing = 1.3
    pf.space_after = Pt(after)
    if indent:
        pf.first_line_indent = Pt(size * 2)
    if text:
        run = p.add_run(text)
        style_run(run, size=size, bold=bold, italic=italic, color=color, cn=cn)
    return p


def rich(doc, segments, size=10.5, after=6):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.line_spacing = 1.3
    pf.space_after = Pt(after)
    pf.first_line_indent = Pt(size * 2)
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
        style_run(run, size=13, bold=True, cn="黑体")
    else:
        pf.space_before = Pt(10)
        pf.space_after = Pt(4)
        style_run(run, size=11.5, bold=True, cn="黑体")
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
    c1 = cap.add_run(f"图 {n}  ")
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
    c1 = cap.add_run(f"表 {n}  ")
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
    st.font.size = Pt(10.5)
    st._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Cm(2.54)
        s.left_margin = s.right_margin = Cm(2.54)
    sec = doc.sections[0]
    sec.header.is_linked_to_previous = False
    hp = sec.header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style_run(hp.add_run("预测权限控制用于自主水下航行器"), size=9, color=GREY)
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

    # ---------------- 标题区 ----------------
    para(doc, "预测权限控制：学习对各控制器的信任程度，并在分布偏移下修复这种信任*",
         size=15, bold=True, cn="黑体", align=WD_ALIGN_PARAGRAPH.CENTER,
         indent=False, after=10)
    para(doc, "[作者姓名]", size=11, italic=True, cn="楷体",
         align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, after=2)
    para(doc, "[作者单位]", size=10, italic=True, cn="楷体",
         align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, after=2)
    para(doc, "* 缩写 PAC 与机器学习中的 probably approximately correct（概率近似正确）学习无关。",
         size=9, italic=True, color=GREY, cn="楷体",
         align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, after=12)

    heading(doc, 1, "摘　要")
    para(doc, "自主水下航行器支撑海洋观测、生态监测、基础设施检查与干预作业，"
              "但这些任务要求在不确定且时变的海流下精确运动。滑模控制（SMC）提供"
              "显式的扰动抑制结构，但可能以跟踪精度换取保守或非平滑动作。模型预测"
              "控制（MPC）预见运动与执行器限制，但其优势依赖运行工况和预测模型的"
              "质量。直接强化学习策略可通过交互适应，但必须学习完整的执行器映射并"
              "承担巨大的验证与部署负担。这些方法都没有直接回答一个更简单的問題："
              "当两个完整的控制器可用时，当前时刻应当赋予各自多少权限？我们引入"
              "预测权限控制（PAC），它学习这种时序分配而非替代控制器。一个 "
              "14,113 参数的 Transformer 编码 0.16 秒的跟踪状态、控制器指令、"
              "海流背景与轨迹相位历史，然后预测一个有界标量系数持续混合 SMC 与 "
              "MPC。一个 2,177 参数的零初始化残差头进一步以每步最多 0.1 的幅度"
              "修正该系数，使组合策略在部署起点与监督基线数值一致。训练首先模仿"
              "短视界滚动 oracle，然后在冻结骨干上进行受约束残差强化学习。在一个"
              "六自由度、10 公斤六推进器水下航行器的仿真中，五个独立初始化的残差"
              "策略在覆盖分布内与分布外扰动族的 1,680 个配对滚动上评估。分布外"
              "平均三维位置 RMSE 为 0.1160 m，对照监督基线 0.1605 m、SMC "
              "0.2482 m、MPC 0.2213 m，对应降幅 27.8%、53.3% 和 47.6%。平均"
              "最大误差较 SMC 和 MPC 分别降低 31.1% 和 28.0%。在执行器退化下，"
              "MPC 专家单独发散至 0.822 m，而残差策略将误差约束在 0.090 m。增益"
              "并非免费：分布内位置误差上升 4.4%，艏向误差上升 4.3%，求解器超时"
              "率从 0.69 升至 0.80。这些结果表明，紧凑的时序策略可以在结构化控制"
              "器之间分配信任，改善分布外跟踪，并保留可验证的最坏情形回退到监督"
              "基线。")
    rich(doc, [("本稿仅报告仿真证据；计划中的硬件验证必须在投稿前完成。",
                {"bold": True})], after=12)

    heading(doc, 1, "1  引言")
    para(doc, "海洋观测需要解析影响海洋生态系统、天气、海洋能量学和全球气候系统的"
              "动态过程[1]。长期观测已经揭示了表层海洋大面积的气候驱动变化[2]。"
              "自主平台将测量扩展到超出船基航次的持续时间、天气容忍度和空间覆盖"
              "范围。水下漂流器群已解析三维亚中尺度动力学[3]，而机动航行器实现了"
              "在脆弱且不规则海床上的近距生态观测[4]。在这两种场景中，记录数据的"
              "价值取决于测量时航行器的位置。跟踪误差因此可能导致观测配准错误、"
              "图像质量下降、碰撞裕度缩窄，并消耗有限的推进能量，而不仅仅是产生"
              "不美观的轨迹。")
    para(doc, "可靠跟踪仍然困难，因为水下航行器耦合了非线性刚体运动、附加质量、"
              "阻尼、恢复力和有界推进器。外部流动既非空间均匀也非完全已知。近期"
              "研究表明，在变化海流中的成功自主导航强烈依赖可用的流动信息[5,6]。"
              "控制器因此必须平衡至少四个要求：精确的位置跟踪、稳定的姿态、有限"
              "的执行器饱和，以及足够平滑的动作以适应硬件。仅改善一个要求可能会"
              "将误差或负担转移至另一个。这种耦合是本文 addressed 的核心工程"
              "问题。")
    para(doc, "基于模型的反馈保留了重要角色，因为它揭示了航行器状态如何产生修正"
              "力。海洋控制模型为补偿和稳定性分析提供了物理可解释的基础[7,8]。"
              "SMC 在有界不确定性下特别有吸引力，因为其趋近律可以在不求解在线"
              "优化问题的情况下抑制扰动[9]。这种鲁棒性并非免费：不连续或高增益"
              "动作可能引起抖振、激励执行器并需要边界层调节[10]。比较 AUV 研究"
              "还表明，控制器的适用性取决于航行器子系统和运行条件，而非单一的"
              "普遍主导设计[11]。这些观察促使我们保留 SMC 作为鲁棒专家，但不"
              "在每一时刻赋予其独占权限。")
    para(doc, "MPC 处理问题的不同部分。它使用预测模型和显式目标函数在跟踪与控制"
              "动作之间权衡，同时执行约束[12]。实验性 AUV 工作已将非线性 MPC 与"
              "扰动处理层结合以改善死区和不确定性下的跟踪[13]。基于学习的 MPC "
              "通过从数据中适应模型、代价或约束来扩展此结构[14]，紧凑求解器表明"
              "预测控制可适应资源受限的处理器[15]。然而，预测性能仍与模型保真度、"
              "目标设计和可用优化时间相关。近期的 AUV 工作因此使用强化学习来调节 "
              "MPC 目标，而非将固定权重视为普遍适用[16]。")
    para(doc, "端到端学习提供了更大的灵活性，但引入了不同的验证问题。六自由度"
              "全 AUV 策略已展示了强大的仿真与迁移结果[17]，深度强化学习改善了"
              "非线性动力学下的路径跟踪[18]。模仿学习也支持了信息驱动的水下导航"
              "[19]。然而，现实世界的强化学习必须处理有限数据、安全约束、部分可观"
              "测性、延迟效应和分布偏移[20]。安全框架可以限制部分风险[21]，但"
              "学习到的六维动作仍然比传统控制律更难检查。本文因此让学习选择权限，"
              "而非重新发现整个执行器机制。")
    para(doc, "结构化机器人学习先前已将名义控制器与学习修正结合。残差强化学习向"
              "传统反馈添加学习动作[22]，相关的共享自治方法学习对另一参与者的最小"
              "修正[23]。Actor-critic MPC 将预测优化嵌入学习策略[24]，而可组合"
              "策略类直接编码任务结构[25]。这些方法证明了学习从强控制先验中受益。"
              "然而，它们并未将两个完整控制器的贡献表述为一个有界的、连续可解释"
              "的、以近期闭环行为为条件的信任变量。")
    para(doc, "PAC 填补了这一空白。在每个 10 毫秒控制步，一个小型 Transformer "
              "接收近期状态-控制器历史并产生一个介于零和一之间的权限系数。该系数"
              "混合完整的 SMC 和 MPC 指令，同时平滑和变化率限制阻止突变转移。"
              "一个 2,177 参数的零初始化残差头进一步以每步最多 0.1 的幅度修正该"
              "系数，使组合策略精确地从监督基线出发，此后只能在有界包络内修正"
              "信任。序列模型刻意远小于典型的 Transformer 策略：它包含 14,113 "
              "个参数、约 55.1 KiB 的 FP32 权重和每次推理约 0.23 百万次乘加运算。"
              "其学习输出是标量，而所有物理执行仍由显式控制器生成。")
    para(doc, "本研究做出四项贡献：")
    rich(doc, [("(1) ", {"bold": True}),
               ("两个完整控制器之间的连续时序权限分配；", {})])
    rich(doc, [("(2) ", {"bold": True}),
               ("滚动 oracle 监督，随后在冻结骨干上进行受约束残差强化学习，"
                "残差以 |Δα| ≤ 0.1 为界并零初始化；", {})])
    rich(doc, [("(3) ", {"bold": True}),
               ("面向边缘时序推理的极紧凑 Transformer，其最坏情形行为等于已部署"
                "的监督基线；", {})])
    rich(doc, [("(4) ", {"bold": True}),
               ("五初始化配对评估，覆盖分布内与分布外扰动族的 1,680 次滚动，"
                "报告跟踪、姿态、平滑度、饱和、控制代价和求解器超时率。", {})])

    heading(doc, 1, "2  相关工作")
    heading(doc, 2, "2.1  水下航行器控制")
    para(doc, "海洋航行器控制建立在带附加质量、阻尼与恢复力的刚体动力学之上[1,2]。"
              "混合方案在阈值切换下将非线性 MPC 与滑模项结合，使各专家停留于有利工况[17]；"
              "这类切换面在设计时固定，而 PAC 在线、连续地回答信任问题。基于学习的水下控制"
              "发展迅速：改进 TD3 路径跟踪[10]、并行仿真器中训练六自由度控制并零样本迁移[8]"
              "均直接学习指令。PAC 则学习关于完整结构化控制器的一维仲裁变量。")
    heading(doc, 2, "2.2  与结构化控制器结合的学习")
    para(doc, "残差强化学习在手工设计的基策略上叠加学习修正[5]；actor-critic MPC 将可微优化器"
              "嵌入策略[6]；学习型 MPC 综述讨论数据驱动修正与安全[4]。PAC 的差异在接口："
              "残差作用于有界标量权限而非执行器指令；监督器冻结而非联合训练；零初始化给出与"
              "已部署系统的精确初始等价。学习型安全滤波训练类屏障修正[11]，而 PAC 的安全滤波"
              "由固定界与结构化回退构成，安全路径中不含学习组件。嵌入式求解器如 TinyMPC[7] "
              "约束了机载仲裁层可假设的算力。")
    heading(doc, 2, "2.3  面向序贯决策的时序编码器")
    para(doc, "自注意力[12]与控制的序列建模表述[13,14]随数据与宽度扩展。PAC 编码器刻意处于相反"
              "规模：单编码层、16 步历史、输出单系数——对仲裁接口已足够，并使边缘部署成为可能。"
              "扩散策略[15]与模仿学习导航[9]处理不同输出空间，属互补工作。")

    heading(doc, 1, "3  预测权限控制")
    heading(doc, 2, "3.1  航行器模型与跟踪目标")
    para(doc, "位姿与体坐标速度记为 η=[x, y, z, φ, θ, ψ]ᵀ、ν=[u, v, w, p, q, r]ᵀ。仿真器以 "
              "100 Hz 四阶龙格–库塔积分对角化六自由度模型（附加质量、线性与二次阻尼、恢复刚度），"
              "每回合 2,100 步（21 s）。六台推进器呈 X 布置、单机限 35 N，受幅值与转速变化率约束。"
              "参考为三维利萨如曲线，期望艏向对齐水平切向；初始位姿与速度分别以 0.03 m、"
              "0.01 m/s 标准差扰动。")
    heading(doc, 2, "3.2  结构化专家控制器")
    para(doc, "主专家为等价控制滑模控制器：平移滑模面组合速度误差与位姿误差，指令力矩含前馈"
              "加速度、阻尼补偿与有界趋近项。第二专家为滚动时域线性时变受约束 MPC：视界 20 步"
              "（0.2 s）、推进器归一化约束、暖启动平滑、在线求解时限 7.5 ms 且超时回退上一拍。"
              "该专家取代本框架早期实现中的一步二次跟踪律；本文全部结果使用该受约束 MPC。")
    heading(doc, 2, "3.3  历史特征与时序编码器")
    para(doc, "每步 24 维特征向量概括位姿误差、体速度、各控制器指令统计与分歧、近饱和比例、"
              "水平海流分量与轨迹相位；最近 16 个向量构成编码器输入。单层 Transformer 编码器"
              "（四头注意力、128 维前馈、14,113 可训练参数）经 Sigmoid 头输出 α̂\u209c。"
              "推理时预测放大 1.2 倍、截断、以系数 0.5 指数平滑、每步变化限制 0.0125。")
    heading(doc, 2, "3.4  分层安全滤波")
    para(doc, "学习分量的全部贡献按执行顺序通过固定滤波栈：(i) 残差有界；(ii) 混合系数经截断、"
              "平滑与变化率限制；(iii) 专家失效强制切回主控；(iv) 非数值回退冻结监督预测。"
              "残差头零初始化，组合策略在部署起点与已部署基线数值一致——该等价由比较全回合滚动的"
              "单元测试在紧数值容差下保证。学习组件的最坏情形因此即基线策略本身。图 2 "
              "给出两个网络与两阶段训练的细节：骨干在第二阶段全程冻结，残差头零"
              "初始化，组合策略的起点精确等于监督基线。")
    figure(doc, "fig1_system_architecture_ai",
           "预测权限控制架构。底色分区区隔已验证的固定组件（对象、SMC 与 MPC 专家）、"
           "可审计的学习组件（冻结时序编码器、已训练残差头）与执行级。"
           "权限 α_t = clip[0,1](α̂_t + Δα_t) 对专家指令做凸混合；"
           "学习分量的全部输出经底部安全条带后才作用于推进器。")
    figure(doc, "fig_network_training_ai",
           "网络架构与两阶段训练。(a) 冻结的时序编码器（14,113 参数）将 "
           "16×24 历史经 32 维嵌入与单层 Transformer 编码器（四头注意力、"
           "128 维前馈）映射为监督权限；可训练的残差头（两层线性、2,177 "
           "参数、末层零初始化）每步最多修正 0.1。(b) 第一阶段以短视界寻优器"
           "的模仿训练骨干；第二阶段仅以 TD3、行为正则与热启动回放训练残差头。")

    heading(doc, 1, "4  权限残差的学习")
    heading(doc, 2, "4.1  监督初始化")
    para(doc, "监督 PAC 策略通过模仿短视界寻优器（oracle）训练：对每个训练状态，"
              "五点网格候选系数保持十步并冻结海流，标签最小化")
    equation(doc,
             r"J(\alpha) = \|\boldsymbol{e}_{xy}\|^2 + 0.7\,e_z^2 + 0.08\,e_\psi^2\, + 0.02\,mathrm{mean}([\,|\boldsymbol{u}|-0.92\,]_+^2)\, + 0.01\,mathrm{mean}((\boldsymbol{u}-\boldsymbol{u}_{t-1})^2)", 2)
    para(doc, "即水平与垂向位置误差、艏向误差、近饱和幅值与指令变化的组合代价。训练使用 "
              "18,900 个带标样本、AdamW 180 轮、加权平方损失。五个模型种子独立训练；"
              "这些冻结策略监督后续全部阶段且从不重训。")
    heading(doc, 2, "4.2  受约束残差阶段")
    para(doc, "残差策略在每条冻结监督策略上附加 2,177 参数的两层头，施加的系数为")
    equation(doc,
             r"\alpha_t = \mathrm{clip}_{[0,1]}(\hat{\alpha}_t+\Delta\alpha_t),\quad\Delta\alpha_t = 0.1\,\tanh(z_t)", 3)
    para(doc, "其中 z_t 为头输出，末层零初始化，故训练前 Δα ≡ 0。训练采用 TD3（双评论家、目标平滑、"
              "延迟策略更新），闭环回合的环境种子与全部评估种子不相交，以监督策略下采集的 "
              "2×10⁵ 条离线过渡热启动，并以行为正则惩罚偏离监督器。奖励综合位置与艏向误差、"
              "指令幅值与变化率、饱和、超时标志与约束违反；全部权重训练前冻结于清单。"
              "学习器仅输出 Δα：不能指令推进器，也不能更改 MPC 调度或求解预算。")

    heading(doc, 1, "5  实验设计")
    heading(doc, 2, "5.1  预注册协议")
    para(doc, "全部比较、种子划分与验收准则在残差训练开始前冻结并归档。评估网格含 120 个配对"
              "回合：85 个分布内（20 个保留环境种子下的 60 个结构化海流回合，加五个训练范围"
              "扰动族的 25 个回合）与 35 个分布外（相对扰动族采样；五个测试范围族，加快时间"
              "尺度海流族与带传感滞后和噪声的估计延迟族）。每个回合在全部方法与种子间共享："
              "相同的航行器、海流与参考实现。")
    heading(doc, 2, "5.2  方法与统计")
    para(doc, "比较六种方法：单独专家（SMC、MPC）、常数混合（固定 α=0.5）、以坐标下降搜索窗口"
              "偏置日程的非学习基线（SSPO），以及共用同一组五个冻结监督种子的两种 PAC 变体"
              "（监督与残差）。SSPO 作为坐标搜索型消融而非外部基线保留。固定方法各贡献一个组合，共 14 个方法-种子组合、1,680 次滚动"
              "（约 353 万控制步）。")
    para(doc, "主推断在模型种子级：逐种子配对效应以自由度 4 的 t 分布区间概括。补充的回合级"
              "自助法区间（对配对回合有放回重采样、10,000 次重采样、随机数种子 0）用于头条对比。"
              "超出预注册集合的比较不做显著性主张，也不做多重比较校正：注册集合被完整报告而非"
              "事后挑选。")

    heading(doc, 1, "6  结果")
    heading(doc, 2, "6.1  闭环跟踪：航行器实际如何运动")
    para(doc, "图 3 以三维路径、水平投影与 z 轴误差展示两个代表性配对回合——分布内正弦海流与"
              "分布外执行器退化回合。分布内，全部学习与混合方法紧贴利萨如参考，SMC 带可见"
              "偏移地滞后。执行器退化下性质改变：MPC 专家单独工作时水平环线畸变、深度漂移"
              "（回合 RMSE 0.886 m），监督策略继承了部分失效（0.206 m），残差策略保持在参考上"
              "（0.088 m），略优于从不自适应的固定混合（0.124 m）。")
    figure(doc, "fig2_closed_loop_trajectories",
           "代表性闭环轨迹。列：分布内正弦海流与分布外执行器退化（各块编号最小的保留种子，"
           "检查前固定）；两个回合在方法间共享，并由冻结检查点确定性重放生成。"
           "行：三维路径、水平投影、z 轴误差（虚线为零参考）。固定混合为清晰起见未画出，"
           "其数值见表 1。")
    para(doc, "图 4 在时间轴上解析同样两个回合。位置误差行显示退化下的 MPC 发散随回合增长"
              "而非尖峰出现——与持续错误的内部模型一致，而非瞬态。两个工况下 残差策略的艏向"
              "误差与俯仰角都保持小值，而退化的 MPC 回合携带可见的姿态偏移。表 1 聚合全网格。")
    figure(doc, "fig3_error_attitude",
           "图 2 两个回合的时间解析运动误差。行：三维位置误差、艏向误差绝对值、俯仰角。"
           "执行器退化下 MPC 的发散随回合增长；残差 PAC 在两个工况下都贴近参考。单回合展示；"
           "块级聚合见表 1 与图 7。")
    para(doc, "残差训练在种子间稳定。图 5 展示第二阶段逐种子的平均回合奖励，"
              "该奖励在探索噪声与轮换训练族日程下采集，其趋势反映的是回合"
              "组合的变化而非贪心策略性能，且没有任何种子发散。五个种子均"
              "产出在注册评估（图 6、表 1）中行为一致的策略——后者才是记录"
              "在案的性能度量。")
    figure(doc, "fig_training_curves",
           "五个模型种子的残差训练。(a) 回合奖励、(b) 评论家损失、(c) 策略损失。"
           "细线：逐种子值；粗线与阴影带：五种子均值 ± 1 标准差。"
           "全部指标在探索噪声与轮换训练族日程下采集，趋势反映回合组合"
           "的变化而非贪心策略性能；无种子发散。两个损失是对移动 TD 目标的"
           "回归——该目标每次更新都由跟踪网络重算，其幅值随目标值尺度的增长"
           "而增长，并非发散；策略损失由负动作价值主导，其尺度随评论家一同"
           "膨胀。贪心闭环性能以图 6 与表 1 的注册评估为准。")
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
          ["方法", "ID RMSE (m)", "OOD RMSE (m)", "OOD 最大误差 (m)",
           "OOD 艏向 (°)", "OOD 超时率", "OOD α 均值"],
          rows,
          "每方法-种子组合 120 配对回合的评估结果（学习方法为五个模型种子均值；ID 分布内；"
          "OOD 相对扰动族采样的分布外）。五种子的离散度以置信区间形式报告于图 6a。",
          [1.5, 0.95, 0.95, 1.1, 0.95, 0.9, 0.85])
    heading(doc, 2, "6.2  残差的价值在哪里")
    para(doc, "在 OOD 块上，残差策略将跟踪 RMSE 由 0.161 m（监督）降至 0.116 m。预注册的"
              "模型种子级配对效应为 −0.0445 m，自由度 4 的 95% t 区间 [−0.0737, −0.0154] m"
              "（相对降低 28%）；35 个 OOD 回合的回合级自助法区间为 [−0.0759, −0.0167] m，"
              "与种子级分析一致。最大误差由 0.495 m 降至 0.414 m（五种子均值），"
              "OOD 艏向误差改善 0.73°。")
    para(doc, "表 1 最尖锐的读法对我们不利：固定 α=0.5 的 OOD 误差 0.120 m，与残差策略相差"
              "不足 3%。该比较未经预注册，本文不做显著性主张；诚实的概括是——残差方法相对"
              "“从不自适应”的总体优势很小，其价值集中在最需要自适应处（执行器退化族较固定混合"
              "再降 28%，见表 2）。分布内效应为 +0.0028 m [+0.0009, +0.0047]（+0.4%），"
              "分布内艏向误差上升 0.32° [+0.29, +0.35]（+4.3%）。")
    para(doc, "图 6b 的族分解定位增益，图 6a 的预注册配对效应量化增益。SSPO 对残差策略的比较"
              "不在归档的配对效应汇总中，以同一种子级程序由原始回合指标补算："
              "OOD −0.0198 m [−0.0211, −0.0185] 利好残差策略，ID +0.0043 m "
              "[+0.0033, +0.0053] 利好 SSPO。")
    figure(doc, "fig4_closed_loop_results",
           "预注册配对效应与族分解。(a) RMSE 配对效应与模型种子 t(4) 95% 区间，按块（OOD、ID）"
           "分组；残差对监督为归档值，SSPO 对比由原始回合指标补算；实心标记为区间不含零。"
           "(b) 各扰动族×方法的 OOD RMSE。增益集中于执行器退化族；温和随机与估计延迟族上"
           "监督策略与 SSPO 略优。本面板不支持“全面占优”的解读。")
    ad_rows = []
    base = fam_mean("v3_transformer", "unseen", "actuator_delay_noise")
    for m in METHOD_ORDER:
        ood = fam_mean(m, "unseen", "actuator_delay_noise")
        rel = "—" if m == "v3_transformer" else f"{(ood / base - 1) * 100:+.0f}%"
        ad_rows.append([METHOD_NAME[m],
                        f"{fam_mean(m, 'seen', 'actuator_delay_noise'):.3f}",
                        f"{ood:.3f}", rel])
    table(doc,
          ["方法", "ID RMSE (m)", "OOD RMSE (m)", "相对监督基线"],
          ad_rows,
          "执行器退化族（响应延迟叠加测量噪声）：配对族均值。",
          [1.7, 1.25, 1.25, 1.1])
    heading(doc, 2, "6.3  残差如何起作用？")
    para(doc, "机制在图 7b 中可见。推进器退化发生后数秒内，监督策略将 α 推过 0.9 并在回合"
              "大部分时间里于该处振荡（31.2% 的步数高于 0.9；标准差 0.331）；残差策略从未"
              "超过 0.69，且在 97.0% 的步数内保持于 [0.4, 0.6]。残差学到的不是一套新的调度"
              "——它学到的是否决监督策略的越界。我们将其解读为分布偏移下的保守化，"
              "这是单一家族的观察，我们不外推。")
    para(doc, "稳定化超出单一回合：在全部 35 个 OOD 回合与五个种子上"
              "（每方法 175 个回合），监督策略的逐回合平均权限为 0.82±0.14"
              "（10–90 分位 0.62–0.93），残差策略为 0.54±0.03，且每个残差"
              "回合都落在 [0.48, 0.62] 内（图 7c）。")
    figure(doc, "fig5_actuator_case",
           "执行器退化回合（保留种子 43000，检查前固定；族均值以文字并置）。"
           "(a) 位置误差。(b) 代表性回合中的权限系数。(c) 全部 35 个 OOD 回合与五个种子的逐回合平均权限（每方法 175 点；横线为均值）。族均值系数：监督 0.63，残差 0.50。")
    heading(doc, 2, "6.4  控制质量：增益未以执行器滥用换取")
    para(doc, "在 OOD 块上，残差策略取得六方法中最低的饱和步占比（0.126，"
              "对照固定混合 0.134、监督策略 0.168）、最低的控制代价（31.9 对 "
              "33.7），率限激活 0.039 低于两位专家的极值。控制代价的配对效应为 "
              "OOD −1.78 [−2.31, −1.26]、ID −0.52 [−0.56, −0.48]：残差策略在 "
              "OOD 跟踪更好的同时，运行代价反而低于其监督器。这为注册准则"
              "“增益不得以艏向、饱和或约束退化换取”提供了实证（图 8）。")
    figure(doc, "fig_control_quality",
           "各方法与各块的控制质量（条形为五种子均值 ± 标准差）。"
           "(a) 控制代价。(b) 饱和步占比。(c) 率限激活。"
           "残差策略在 OOD 上取得全部方法中最低的饱和与控制代价。")
    heading(doc, 2, "6.5  增益的代价")
    para(doc, "两项代价伴随 OOD 增益。分布内位置与艏向误差如上量化地增加。求解器超时率——"
              "评估框架中 MPC 实例墙钟超过 10 ms 控制周期的步数占比，对所有方法同口径测量——"
              "在残差策略下由共同的 0.69 基线升至 0.80（OOD +0.107 [+0.094, +0.120]；"
              "ID +0.028 [+0.010, +0.047]）。固定混合使该指标不变，故系数取值本身并非原因；"
              "候选解释是依赖轨迹的求解器条件数，作为假设而非结论标注。"
              "部署侧保障不受影响：回退上一拍指令的触发率全方法为 0.3–0.5%，"
              "一米阈值任务成功率不变（ID 0.988 / OOD 0.943），OOD 最大误差反而改善。")
    figure(doc, "fig6_cost_tradeoff",
           "代价核算。各方法-种子组合的 OOD RMSE 对求解器超时率散点；该指标统计共享评估框架中 "
           "MPC 实例超过 10 ms 控制周期的步数，对所有方法同口径采集。阴影带：共同基线水平"
           "（0.69：SMC / MPC / 固定 α / SSPO）。")
    table(doc,
          ["指标", "ID 效应 [95% CI]", "OOD 效应 [95% CI]"],
          [["RMSE (m)", "+0.0028 [+0.0009, +0.0047]",
            "−0.0445 [−0.0737, −0.0154]"],
           ["艏向误差 (°)", "+0.32 [+0.29, +0.35]", "−0.73 [−1.41, −0.06]"],
           ["控制代价", "−0.52 [−0.56, −0.48]", "−1.78 [−2.31, −1.26]"],
           ["超时率", "+0.028 [+0.010, +0.047]", "+0.107 [+0.094, +0.120]"]],
          "残差策略相对监督基线在全部报告指标上的注册配对效应（模型种子 "
          "t(4) 95% 区间）。SSPO 对比由原始回合指标补算：RMSE OOD "
          "−0.0198 [−0.0211, −0.0185] 利好残差策略，ID +0.0043 "
          "[+0.0033, +0.0053] 利好 SSPO。",
          [2.2, 2.3, 2.3])
    para(doc, "对照与残差阶段相关的六项预注册验收准则：三项通过（增益未以艏向、饱和"
              "或约束违反换取；消融可分离；五种子方向稳定）；两项未通过（整体非劣性——经由分布内 "
              "+0.4% 效应；超时率不恶化）；一项——部署推理时间满足控制周期——未评估，"
              "且为硬件阶段的第一道门槛。")

    heading(doc, 1, "7  局限与讨论")
    para(doc, "核心结果是：两千参数量级的残差，仅作用于有界权限系数，将在分布偏移下失效的监督"
              "仲裁策略，改造为总体上匹配强常数混合、并在最需要自适应处显著占优的策略。安全论证"
              "是架构性的：零初始化保证部署起点与已部署基线数值一致，有界输出限制此后每次扰动，"
              "固定回退将最坏情形界定为基线本身。这界定了本文相对残差强化学习[5]（扰动执行器"
              "指令）与学习型安全滤波[11]（在学习路径中放置学习组件）的位置。")
    para(doc, "六条边界限定本文主张。第一，全部证据为仿真级别。第二，特征集接收精确海流信息，"
              "硬件上必须估计。第三，分布外指相对扰动族采样，不指任意现场条件。第四，分布内"
              "非劣性未通过（位置 +0.4%、艏向 +4.3%）。第五，超时率上升（+0.107）原因未确立。"
              "第六，组合策略的部署延迟未测量。")
    para(doc, "这些边界定义了注册的下一步实验。第一，直接 RL 对照——在同一"
              "网格上端到端输出推进器指令的 TD3 策略——补齐残差 RL 研究所期待"
              "的比较层级。第二，α ∈ {0, 0.25, 0.5, 0.75, 1} 的常数权限扫描，"
              "揭示策略所航行的信任地形。两者均需新的注册运行。第三，在预注册"
              "规则下解冻最后一个编码器块、针对分布内效应的残差第二轮训练。"
              "第四，隔离超时机制。第五，第 8 节的硬件协议，其第一道门槛为部署"
              "时延。")

    heading(doc, 1, "8  计划中的硬件验证")
    para(doc, "没有物理实验，本文不具备投稿条件。计划在同一台 10 kg 六推进器航行器上部署相同的 "
              "SMC、MPC 专家、监督 PAC 与 残差 PAC 检查点。协议包括每方法每扰动条件至少十次独立"
              "试验、控制器次序随机化、固定的电池电压接受窗、相同的轨迹与执行器限制。扰动从"
              "静水逐步推进到稳态横流、由系留牵引或标定流源产生的可复现阶跃式扰动，以及——"
              "针对本文主张——注入的执行器退化条件：在部分推进器上施加经标定的一阶指令滞后"
              "加噪声。")
    para(doc, "机载估计器须由 IMU、深度、DVL 与定位信息提供位姿、体速度与因果海流估计；"
              "不得暴露仿真器的精确海流。除位置与姿态 RMSE 外，硬件部分报告最大误差、成功率、"
              "实测电能耗、逐步推理时延、端到端控制回路时延（即未评估的那项准则）、饱和时长与"
              "紧急干预。占位项（平台尺寸、传感器套件、试验点标定、试验数）只能以实测值填充，"
              "不得以仿真或预期数据顶替。")

    heading(doc, 1, "9  结论")
    para(doc, "PAC 将学习型水下控制重构为结构化控制器之间的信任问题，残差阶段在分布偏移下修复"
              "这种信任：OOD 跟踪误差降低 28%，种子级与回合级的置信区间均不含零；优化器的退化"
              "失效被约束至 0.090 m；每一项代价都与训练前固定的准则对照测量。学习组件保持有界、"
              "可作为单条系数轨迹审计、最坏情形等价于已部署基线。硬件验证、因果海流估计、时延"
              "测量与时变混合的稳定性分析，是任何面向真实环境的鲁棒性主张之前提。")

    heading(doc, 1, "数据可用性")
    para(doc, "本文全部数值来自哈希冻结的归档（配置、数据集与检查点摘要及逐文件校验和），"
              "覆盖监督与残差两个训练阶段与配对评估。每幅图与每张表的源数据均可由代码库构建脚本"
              "从这些归档重新生成；轨迹图由冻结检查点对归档回合做确定性重放。")

    heading(doc, 1, "参考文献")
    refs = [
        "T. I. Fossen. Handbook of Marine Craft Hydrodynamics and Motion Control[M]. 2nd ed. Wiley, 2021.",
        "G. Antonelli. Underwater Robots: Motion and Force Control of Vehicle-Manipulator Systems[M]. 3rd ed. Springer, 2014.",
        "E. F. Camacho, C. Bordons Alba. Model Predictive Control[M]. 2nd ed. Springer, 2013.",
        "L. Hewing, K. P. Wabersich, M. Menner, M. N. Zeilinger. Learning-based model predictive control: Toward safe learning in control[J]. Annual Review of Control, Robotics, and Autonomous Systems, 2020, 3: 269–296.",
        "T. Johannink, et al. Residual reinforcement learning for robot control[C]//Proc. IEEE ICRA, 2019: 6023–6029.",
        "A. Romero, Y. Song, D. Scaramuzza. Actor-critic model predictive control[C]//Proc. IEEE ICRA, 2024: 14777–14784.",
        "K. Nguyen, S. Schoedel, A. Alavilli, B. Plancher, Z. Manchester. TinyMPC: Model-predictive control on resource-constrained microcontrollers[C]//Proc. IEEE ICRA, 2024.",
        "L. Cai, K. Chang, Y. Girdhar. Learning to swim: Reinforcement learning for 6-DOF control of thruster-driven autonomous underwater vehicles[C]//Proc. IEEE ICRA, 2025: 11286–11293.",
        "X. Lin, et al. UIVNAV: Underwater information-driven vision-based navigation via imitation learning[C]//Proc. IEEE ICRA, 2024: 5250–5256.",
        "Y. Fan, H. Dong, X. Zhao, P. Denissenko. Path-following control of unmanned underwater vehicle based on an improved TD3 deep reinforcement learning[J]. IEEE Transactions on Control Systems Technology, 2024, 32(5): 1904–1919.",
        "O. So, et al. How to train your neural control barrier function: Learning safety filters for complex input-constrained systems[C]//Proc. IEEE ICRA, 2024: 11532–11539.",
        "A. Vaswani, et al. Attention is all you need[C]//Advances in Neural Information Processing Systems, 2017, 30: 5998–6008.",
        "L. Chen, et al. Decision Transformer: Reinforcement learning via sequence modeling[C]//Advances in Neural Information Processing Systems, 2021, 34: 15084–15097.",
        "M. Janner, Q. Li, S. Levine. Offline reinforcement learning as one big sequence modeling problem[C]//Advances in Neural Information Processing Systems, 2021, 34: 1273–1286.",
        "C. Chi, et al. Diffusion policy: Visuomotor policy learning via action diffusion[C]//Robotics: Science and Systems, 2023.",
        "D. Liberzon. Switching in Systems and Control[M]. Birkhäuser, 2003.",
        "Z. Zhao, X. Liu, T. Wang, Z. Zhou, M. Zhang. Hybrid control scheme of nonlinear model prediction and adaptive terminal sliding mode for underwater vehicles based on threshold switching[J]. ISA Transactions, 2026, 176: 580–590.",
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
