# -*- coding: utf-8 -*-
"""Build the master's thesis proposal (开题报告) for the PAC project.

Follows the ZJU Ocean College template structure (孔政's proposal):
Cover form → CN/EN abstracts → 6 chapters → signature form pages.
Chapter 3 (初步进展) is the largest, containing PAC v1-v4 results.
"""
from pathlib import Path

from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "manuscript_figures"
OUT = ROOT / "paper" / "开题报告_田家丞.docx"
BLACK = RGBColor(0, 0, 0)
GREY = RGBColor(0x80, 0x80, 0x80)

doc = Document()
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(10.5)
st._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
for s in doc.sections:
    s.top_margin = Cm(2.54)
    s.bottom_margin = Cm(2.54)
    s.left_margin = Cm(3.17)
    s.right_margin = Cm(3.17)

sec = doc.sections[0]
sec.header.is_linked_to_previous = False
hp = sec.header.paragraphs[0]
hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
hr = hp.add_run("浙江大学硕士学位论文开题报告")
hr.font.size = Pt(9)
hr.font.color.rgb = GREY
sec.footer.is_linked_to_previous = False
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
fld = OxmlElement("w:fldSimple")
fld.set(qn("w:instr"), "PAGE \\* arabic \\* MERGEFORMAT")
r_el = OxmlElement("w:r")
rpr = OxmlElement("w:rPr")
sz_el = OxmlElement("w:sz"); sz_el.set(qn("w:val"), "18")
col_el = OxmlElement("w:color"); col_el.set(qn("w:val"), "808080")
rpr.append(sz_el); rpr.append(col_el)
r_el.append(rpr)
t_el = OxmlElement("w:t"); t_el.text = "1"
r_el.append(t_el)
fld.append(r_el)
fp._p.append(fld)


def _set_font(run, size=10.5, bold=False, cn="宋体", en="Times New Roman", color=BLACK):
    run.font.name = en
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.append(rf)
    rf.set(qn("w:eastAsia"), cn)


def para(text="", size=10.5, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         after=6, indent=True, cn="宋体"):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.line_spacing = 1.5
    pf.space_after = Pt(after)
    if indent:
        pf.first_line_indent = Pt(size * 2)
    if text:
        r = p.add_run(text)
        _set_font(r, size=size, bold=bold, cn=cn)
    return p


def h1(text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(18)
    pf.space_after = Pt(8)
    pf.line_spacing = 1.5
    pf.keep_with_next = True
    r = p.add_run(text)
    _set_font(r, size=14, bold=True, cn="黑体")
    return p


def h2(text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(10)
    pf.space_after = Pt(5)
    pf.line_spacing = 1.5
    pf.keep_with_next = True
    r = p.add_run(text)
    _set_font(r, size=12, bold=True, cn="黑体")
    return p


def h3(text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(6)
    pf.space_after = Pt(3)
    pf.line_spacing = 1.5
    pf.keep_with_next = True
    r = p.add_run(text)
    _set_font(r, size=10.5, bold=True, cn="黑体")
    return p


def figure(stem, caption, width_in=5.8):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    r = p.add_run()
    r.add_picture(str(FIGS / f"{stem}.png"), width=Cm(width_in * 2.54))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(8)
    cr = cap.add_run(caption)
    _set_font(cr, size=9)


def _shade(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "F2F2F2")
    tcPr.append(shd)


def table(header, rows, widths, caption=""):
    if caption:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.keep_with_next = True
        cr = cap.add_run(caption)
        _set_font(cr, size=9, bold=True)
    t = doc.add_table(rows=1 + len(rows), cols=len(header), style="Table Grid")
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        c = t.rows[0].cells[j]
        c.width = Cm(widths[j])
        _shade(c)
        p = c.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h); _set_font(r, size=9, bold=True)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.rows[i + 1].cells[j]
            c.width = Cm(widths[j])
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val)); _set_font(r, size=9)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# ============================================================ COVER
para("", indent=False, after=20)
para("基于预测权限控制的水下机器人鲁棒跟踪控制研究",
     size=22, bold=True, cn="黑体", align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, after=30)
para("Research on Robust Tracking Control of Underwater Vehicles\n"
     "Based on Predictive Authority Control",
     size=14, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, after=30)
cover_info = [
    ("研究生姓名", "田家丞"),
    ("指导教师", "【导师姓名】"),
    ("专业名称", "【专业名称】"),
    ("研究方向", "水下机器人运动控制"),
    ("所在院系", "海洋学院"),
    ("提交日期", "2026年10月"),
]
t = doc.add_table(rows=len(cover_info), cols=2, style="Table Grid")
t.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, (label, value) in enumerate(cover_info):
    c0 = t.rows[i].cells[0]; c0.width = Cm(4.0)
    p0 = c0.paragraphs[0]; p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r0 = p0.add_run(label); _set_font(r0, size=12, bold=True, cn="黑体")
    c1 = t.rows[i].cells[1]; c1.width = Cm(8.0)
    p1 = c1.paragraphs[0]; p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = p1.add_run(value); _set_font(r1, size=12)
doc.add_page_break()

# ============================================================ ABSTRACT CN
h1("摘  要")
para("自主水下航行器（AUV）在海洋观测、生态监测与基础设施检查中需要精确的轨迹跟踪能力。"
     "然而海流扰动的不确定性、执行器老化以及环境参数漂移使得单一控制器难以在所有工况下"
     "保持优良性能。本研究提出预测权限控制（Predictive Authority Control，PAC）方法，"
     "通过一个紧凑的时序编码器预测有界权限系数，对滑模控制（SMC）与模型预测控制（MPC）"
     "两个专家控制器进行连续凸混合，使学习组件仅决定\u201c信任分配\u201d而非直接产生控制指令。"
     "在此基础上，本研究进一步引入受约束残差强化学习：在冻结的监督策略之上附加一个"
     "零初始化、上下界为±0.1的残差修正头，使系统在部署起点与已验证基线数值一致，"
     "从而获得可认证的安全回退保证。在覆盖10族扰动的120个配对评估回合中（1,680次滚动），"
     "残差策略将分布外跟踪误差从0.161 m降至0.116 m（相对改善28%），"
     "并在执行器退化场景将MPC单独工作时0.822 m的灾难性误差约束至0.090 m。"
     "整个学习系统仅含16,290个参数（63.6 KB），为同类深度学习控制器中最轻量的设计。")
para("关键词：自主水下航行器；权限分配；残差强化学习；滑模控制；模型预测控制；安全学习",
     indent=False)
doc.add_page_break()

# ============================================================ ABSTRACT EN
h1("Abstract")
para("Autonomous underwater vehicles (AUVs) require accurate trajectory tracking for "
     "ocean observation, ecological monitoring, and infrastructure inspection. However, "
     "uncertain currents, actuator degradation, and parameter drift prevent any single "
     "controller from maintaining optimal performance across all operating conditions. "
     "This research proposes Predictive Authority Control (PAC), which uses a compact "
     "temporal encoder to predict a bounded authority coefficient that continuously blends "
     "sliding-mode control (SMC) and model predictive control (MPC), ensuring that the "
     "learned component determines only trust allocation rather than generating control "
     "commands directly. A constrained residual reinforcement learning stage is further "
     "introduced: a zero-initialized residual head bounded by ±0.1 is appended to the "
     "frozen supervised policy, guaranteeing that the deployed system is numerically "
     "identical to the verified baseline at initialization. Across 120 paired evaluation "
     "episodes covering 10 disturbance families (1,680 rollouts), the residual policy "
     "reduces out-of-distribution tracking RMSE from 0.161 m to 0.116 m (28% improvement) "
     "and confines a catastrophic MPC failure under actuator degradation from 0.822 m "
     "to 0.090 m. The entire learning system contains only 16,290 parameters (63.6 KB), "
     "the smallest among comparable deep learning controllers.")
para("Keywords: autonomous underwater vehicle; authority allocation; residual "
     "reinforcement learning; sliding-mode control; model predictive control; safe learning",
     indent=False)
doc.add_page_break()

# ============================================================ CH1
h1("1  选题意义")

h2("1.1  研究背景及意义")
para("海洋覆盖了地球表面的约71%，是全球气候系统、生物多样性和经济活动的重要驱动力。"
     "随着海洋资源开发的加速和海洋强国战略的推进，对海洋环境的实时监测需求日益增长。"
     "自主水下航行器（AUV）作为海洋探测的重要工具，已在海洋观测、管道检查、"
     "结构健康监测等领域得到广泛应用。")
para("精确的轨迹跟踪是AUV完成上述任务的基础保障。跟踪误差不仅影响观测数据的"
     "空间配准精度，还关系到航行器与海底结构物之间的碰撞安全。然而，水下环境的"
     "复杂性和不确定性使得精确跟踪面临严峻挑战：海流的时空变化、航行器自身参数"
     "的不确定性（如载荷变化引起的水动力参数漂移）、以及执行器老化导致的响应"
     "延迟和噪声，都会使固定参数的控制器性能显著下降。")
para("滑模控制（SMC）和模型预测控制（MPC）是目前水下机器人领域两种主流的控制方法，"
     "它们各有优劣且高度互补：SMC具有强鲁棒性和透明的反馈结构，但控制动作粗糙；"
     "MPC能够预见运动趋势并显式处理约束，但严重依赖模型精度——当执行器退化导致"
     "模型失配时，MPC可能产生灾难性后果。本研究仿真中，MPC在执行器退化场景下"
     "单独工作的误差可达0.822 m，约为正常水平的9倍，逼近1 m任务失败线。")
para("预测权限控制（PAC）的核心思想是：不让学习系统替代这些经过验证的经典控制器，"
     "而是让它学习一个有界的\u201c权限\u201d系数α，逐瞬时决定两个控制器各占多大话语权。"
     "这种方式将学习组件的输出从6维推杆指令缩减为1个标量，使得系统的安全审计"
     "和行为解释变得直接可行。在此基础上，受约束残差强化学习通过在冻结的已验证"
     "基线之上附加一个微小修正头，在保持安全底线的同时获得分布外鲁棒性——"
     "这解决了实际工程中\u201c既要安全又要智能\u201d的核心矛盾。")
para("本研究的意义在于：（1）提出了一种可在安全约束下部署学习型控制器的架构性"
     "方案，对水下机器人在复杂环境下的可靠运行具有工程价值；（2）建立了预注册"
     "配对评估协议，为学习型控制器的公平比较提供了方法论贡献；（3）整个学习系统"
     "仅16,290参数（63.6 KB），是同类方法中最轻量的，可部署于嵌入式平台。")

h2("1.2  国内外相关研究进展")

h3("1.2.1  水下机器人运动控制研究现状")
para("水下机器人运动控制建立在非线刚体动力学、附加质量和水动力阻尼的物理模型之上[1,2]。"
     "滑模控制通过设计趋近律抑制有界扰动，在参数不确定下具有强鲁棒性[3]，"
     "但其不连续或高增益动作可能引起抖振并激励执行器。模型预测控制利用预测模型"
     "和显式目标函数在跟踪与控制动作之间权衡[4]，实验性AUV工作已将非线性MPC"
     "与扰动处理层结合以改善不确定条件下的跟踪性能[5]。混合方案在阈值切换下"
     "结合两种控制器的优势[6,7]，但切换面在设计时固定，无法适应时变工况。"
     "自适应控制方法（如RBF神经网络自适应[8]、L1自适应[9]）能在线调整参数，"
     "但主要针对参数不确定而非分布外扰动。")

h3("1.2.2  强化学习与智能控制研究现状")
h4 = h3; h4("1.2.2.1  深度强化学习控制")
para("深度强化学习通过与环境的交互实现自适应控制策略。Fan等[10]提出改进的TD3算法"
     "用于UUV路径跟随，在非线性动力学下实现了优于DDPG和标准TD3的控制性能。"
     "Cai等[11]在ICRA 2025上展示了基于强化学习的6自由度AUV控制（Learning to Swim），"
     "在Isaac仿真环境中训练并迁移至真实航行器CUREE。这些方法直接学习完整的"
     "执行器指令映射，具有高灵活性但缺少安全认证手段——学习策略的行为难以分解"
     "和审计，且最坏情形无法界定。")

h3("1.2.2.2  残差强化学习")
para("残差强化学习将学习策略与传统控制器结合：学习组件只产生对基础控制器的修正量"
     "而非完整指令。Johannink等[12]在ICRA 2019的开创性工作中将SAC策略叠加在"
     "脚本控制器之上，在真实机器人装配任务中显著提升成功率。Silver等[13]的"
     "Residual Policy Learning进一步验证了该方法在灵巧操作中的有效性。"
     "然而，现有残差RL方法的修正量作用于执行器空间（6维），缺乏有界安全保证，"
     "且未见在水下机器人控制中的应用。本研究将残差思想从执行器空间缩减到权限"
     "空间（1维标量），并引入零初始化和有界约束，使安全论证成为可能。")

h3("1.2.3  学习型权限分配研究现状")
para("学习型权限分配的核心问题是：给定两个（或多个）完整的控制器，如何根据当前"
     "状态动态决定各控制器的权重。增益调度（gain scheduling）是传统方案，"
     "但调度规律需人工设计。人机共享控制中的权限分配[14]将人类操作员与自动驾驶"
     "的输出按学习到的权重混合，概念上与本研究相近但应用场景不同。"
     "在本课题组的前期工作（v1–v3）中，已建立了基于Transformer的权限预测框架，"
     "并在三种结构化海流下验证了监督学习方法的可行性[15]。本研究在此基础上"
     "进一步引入受约束残差强化学习，解决监督方法在分布外扰动下退化的核心问题。")

h2("1.3  课题来源")
para("本课题来源于【导师主持的科研项目名称及编号】。")

h2("1.4  研究目标和拟解决的关键问题")
para("本研究的总体目标是：建立一套基于预测权限控制的水下机器人鲁棒跟踪控制体系，"
     "在保证安全底线的前提下，通过受约束残差强化学习获得对分布外扰动的鲁棒性，"
     "并最终在真实水下平台上完成硬件验证。")
para("拟解决的关键问题包括：")
para("（1）如何设计学习组件的接口使其安全可认证：将学习输出从6维推杆指令缩减"
     "为1维有界权限系数，并通过零初始化和分层安全滤波保证最坏情形退化为已验证基线。")
para("（2）如何使学习系统具备分布外鲁棒性：监督学习仅在可生成标准答案的环境中有效，"
     "残差强化学习通过奖励信号在更广的扰动环境中训练，获得对分布偏移的泛化能力。")
para("（3）如何在保持轻量化的同时实现上述目标：整个学习系统（编码器+残差头）"
     "仅16,290参数，为同类深度学习控制器的1/4至1/20，可在嵌入式平台实时运行。")

doc.add_page_break()

# ============================================================ CH2
h1("2  研究内容及方案")

h2("2.1  研究内容")
para("本论文的研究内容由以下三部分构成：")

h3("2.1.1  受约束残差强化学习权限分配方法研究")
para("研究冻结监督策略之上的残差强化学习范式。核心是将学习动作从执行器空间"
     "（6维推杆指令）缩减到权限空间（1维标量Δα），通过零初始化和有界约束"
     "（|Δα| ≤ 0.1）保证部署起点的安全等价性。包括：奖励函数设计（8项加权，"
     "权重训练前冻结）、TD3训练算法（双评论家、目标平滑、行为正则）、"
     "以及分层安全滤波栈（限幅/平滑/率限/强制主控/数值回退）。")

h3("2.1.2  控制质量与安全分析方法研究")
para("研究残差强化学习的控制质量评估方法。建立预注册配对评估协议：120个配对回合"
     "覆盖10族扰动（85分布内+35分布外），8种方法×5种子。评估指标包括跟踪精度"
     "（RMSE/最大误差）、姿态精度（艏向/俯仰）、控制质量（饱和率/控制代价/"
     "变化率）以及安全指标（约束违反率/求解器超时率）。重点关注增益是否以"
     "执行器滥用为代价，以及代价的诚实报告。")

h3("2.1.3  水下机器人实验平台设计与硬件验证")
para("设计并搭建10 kg级六推进器AUV实验平台，将仿真中训练的PAC策略部署到"
     "嵌入式计算平台，在水池环境中完成硬件在环验证。包括：推进器退化注入"
     "实验（一阶滞后+噪声）、海流扰动模拟（拖曳或射流）、实时推理时延测量、"
     "以及与仿真结果的对照分析。")

h2("2.2  研究方案")

h3("2.2.1  受约束残差强化学习研究方案")
para("（1）建立六自由度AUV仿真环境，采用真实平台的全套实测参数（质量、惯量、"
     "水动力系数、推进器特性），实现10族扰动的参数化生成。")
para("（2）设计并实现24维特征向量（位置误差、体速度、SMC/MPC指令、海流估计、"
     "轨迹相位），构建Transformer编码器（14,113参数）作为冻结骨干。")
para("（3）设计8项加权奖励函数，权重在训练前冻结进manifest。实现TD3训练器，"
     "包含2×10⁵离线过渡热启动、行为正则化、探索噪声。")
para("（4）实现分层安全滤波栈，包含单元测试验证零初始化的逐位等价性。")

h3("2.2.2  控制质量与安全分析方案")
para("（1）设计预注册配对评估协议：训练前冻结全部比较、种子划分和验收准则。")
para("（2）实施120配对回合×14方法-种子组合=1,680次正式评估滚动。")
para("（3）分析跟踪精度、控制质量、安全指标，进行模型种子级配对t检验。")
para("（4）建立六项注册验收准则，诚实报告通过项和未通过项。")

h3("2.2.3  硬件验证方案")
para("（1）搭建10 kg级六推进器AUV平台，配备IMU、深度计、DVL和嵌入式计算单元。")
para("（2）实现机载估计器（位姿/速度/海流的因果估计），替代仿真中的精确信息。")
para("（3）在水池中进行硬件在环实验：每方法每扰动条件至少十次独立试验、"
     "控制器次序随机化、固定的电池电压接受范围。")
para("（4）分析仿真到实际的迁移差距，重点评估部署推理时延（第7条注册准则）。")

h2("2.3  可行性分析")
para("（1）前期基础：本课题组已建立完整的PAC框架（v1–v3），在三种结构化海流下"
     "完成了多种子正式评估；v4阶段已实现受约束残差强化学习和10族扰动的"
     "预注册配对评估，在仿真中验证了OOD −28%的改善。上述工作已形成完整的"
     "代码库（221项自动化测试全部通过）和冻结证据链。")
para("（2）研究团队：课题组在水下机器人控制、强化学习和嵌入式系统方面具有"
     "扎实的研究基础和丰富的工程经验。")
para("（3）实验平台：学院拥有水池实验条件和AUV搭建经验，可支持硬件验证阶段"
     "的实验需求。")

doc.add_page_break()

# ============================================================ CH3 (main results)
h1("3  课题研究初步进展")
para("本章报告截至2026年9月的主要研究进展，包括PAC架构设计与监督基线（§3.1）"
     "以及受约束残差强化学习的正式评估结果（§3.2）。")

h2("3.1  PAC架构与监督基线")

h3("3.1.1  系统架构设计")
para("PAC系统由三个层次组成（见图3.1）：底层为两个结构化专家控制器"
     "（SMC主控制器和受约束MPC），中层为冻结的时序编码器（预测权限α̂），"
     "上层为可训练的残差头（输出有界修正Δα）。实际推进器指令为"
     "u = (1−α)·u_SMC + α·u_MPC，学习组件不直接产生控制指令。")
figure("fig1_system_architecture_ai",
       "图3.1  PAC系统架构：权限α混合SMC与MPC专家输出")

h3("3.1.2  Oracle数据集构建与监督训练")
para("监督训练使用短视界滚动Oracle标签：对每个训练状态，将5个候选α值"
     "（0/0.25/0.5/0.75/1）各保持10步并冻结海流，选取代价最小的α作为标签。"
     "在3种结构化海流（恒定/正弦/阶跃）× 3个数据生成种子 × 2100步的设置下，"
     "共生成18,900个带标签样本。使用AdamW优化器训练180轮，"
     "5个模型种子独立训练后全部冻结。")

h3("3.1.3  正式评估协议")
para("建立预注册配对评估协议：120个配对回合覆盖10族扰动（见表3.1），"
     "8种方法在同一场景下同场竞技，统计推断在模型种子级进行（t(4) 95%置信区间）。"
     "所有比较、种子划分和验收准则在训练前冻结并归档。")
table(
    ["族", "块", "训练范围", "测试范围"],
    [["结构化（3种）", "ID", "恒定×2.5 / 正弦 / 阶跃", "—"],
     ["质量-阻尼失配", "OOD", "mass×[0.94,1.06]", "mass×[1.10,1.25]"],
     ["彩色噪声", "OOD", "ρ∈[0.90,0.98]", "ρ∈[0.985,0.995]"],
     ["OU海流", "OOD", "θ∈[0.2,1.0]", "θ∈[1.2,2.0]"],
     ["随机频率-幅值", "OOD", "amp∈[2.0,3.0]", "amp∈[3.2,4.0]"],
     ["执行器延迟+噪声", "OOD", "delay 2–4步", "delay 6–10步"],
     ["快变OU海流", "OOD", "—", "θ∈[3.0,6.0]"],
     ["估计延迟", "OOD", "—", "delay 10–20步"]],
    [3.0, 1.0, 4.0, 4.0],
    "表3.1  10族扰动评估网格")

h3("3.1.4  监督基线结果")
para("监督PAC基线在分布内表现优异：三维位置RMSE为0.064 m，优于SMC（0.162 m）"
     "和MPC（0.071 m）。然而在分布外（OOD），监督基线退化为0.161 m，"
     "甚至不如完全不调度的固定α=0.5（0.120 m）。机制分析表明，"
     "监督策略在陌生海况下将α推至0.8以上并剧烈振荡（标准差0.33），"
     "本质上是模仿学习对分布外输入的系统性失效。这一发现直接激发了"
     "受约束残差强化学习的设计。")

doc.add_page_break()

h2("3.2  受约束残差强化学习")

h3("3.2.1  奖励函数设计")
para("设计8项加权单步奖励函数（权重训练前冻结进manifest）：")
para("r = −w_e·(e_pos/s_pos) − w_h·(e_head/s_head) − w_u·‖u‖² − w_Δu·‖Δu‖² "
     "− w_Δα·|Δα| − w_sat·ρ_sat − w_dl·ρ_dl − w_c·ρ_con", indent=False,
     align=WD_ALIGN_PARAGRAPH.CENTER)
para("其中位置误差权重w_e=1.0（最主要目标），约束违反罚w_c=10.0（安全>>性能），"
     "超时项ρ_dl仅记录不改调度。归一化尺度s_pos=0.1 m、s_head=0.05 rad"
     "使各项量级可比。")

h3("3.2.2  TD3训练")
para("采用TD3算法（双评论家、目标平滑、延迟策略更新），仅训练2,177参数残差头。"
     "训练环境轮转全部6族扰动的训练范围（包括监督基线未见过的随机族）。"
     "以2×10⁵离线过渡热启动，行为正则系数λ=0.01防止偏离监督器太远。"
     "5个模型种子独立训练，训练曲线见图3.2。")
figure("fig_training_curves",
       "图3.2  残差训练曲线：回合奖励、评论家损失、策略损失（5种子）")

h3("3.2.3  正式评估结果")
para("在120个配对回合的正式评估中，残差策略取得以下核心结果：")
para("分布外跟踪：OOD RMSE从0.161 m降至0.116 m（相对改善28%，"
     "配对95% CI [−0.074, −0.015] m，模型种子级t(4)检验显著）。"
     "最大误差从0.495 m降至0.414 m。")
para("执行器退化免疫：MPC专家单独工作时误差发散至0.822 m，"
     "残差策略将其约束至0.090 m（−89%）。监督策略在此场景为0.295 m，"
     "固定α=0.5为0.125 m——残差策略较两者分别改善69%和28%。"
     "核心场景的三维轨迹与权限行为见图3.3。")
figure("fig5_actuator_case",
       "图3.3  执行器退化场景：位置误差、权限轨迹与分布箱线图")
para("分布外全景：在7个OOD扰动族的热图分析中（见图3.4），残差策略的增益"
     "集中在执行器退化族和极端外推族，温和随机海流族上与对照持平。")
figure("fig4_closed_loop_results",
       "图3.4  预注册配对效应森林图与族热图")

h3("3.2.4  控制质量与代价分析")
para("残差策略在OOD块取得六方法中最低的饱和步占比（0.126）和最低的控制代价"
     "（31.9），配对控制代价效应为−1.78 [−2.31, −1.26]——即跟踪更好的同时"
     "运行代价反而更低。这证实了注册准则\u201c增益不以执行器滥用为代价\u201d。")
para("代价方面如实报告：分布内位置误差上升4.4%，艏向误差上升4.3%，"
     "求解器超时率从0.69升至0.80。在六项注册验收准则中，三项通过、"
     "两项未通过、一项（部署推理时延）未评估——全部结果与训练前冻结的"
     "准则对照测量。")
table(
    ["方法", "ID RMSE (m)", "OOD RMSE (m)", "OOD饱和率", "OOD控制代价"],
    [["SMC", "0.162", "0.248", "0.146", "32.0"],
     ["MPC", "0.071", "0.221", "0.207", "35.2"],
     ["固定α=0.5", "0.066", "0.120", "0.134", "32.4"],
     ["SSPO", "0.062", "0.136", "0.167", "33.7"],
     ["Supervised PAC", "0.064", "0.161", "0.168", "33.7"],
     ["Residual PAC", "0.066", "0.116", "0.126", "31.9"]],
    [3.0, 2.2, 2.2, 2.0, 2.0],
    "表3.2  六方法正式评估结果汇总")

doc.add_page_break()

# ============================================================ CH4
h1("4  课题进度安排")
table(
    ["任务名称", "时间安排", "具体任务"],
    [["文献调研与\n理论框架", "2024.09–2024.12",
      "完成PAC理论框架设计，实现仿真环境\n与Oracle数据生成管线，初步验证"],
     ["监督基线\n与架构迭代", "2025.01–2025.06",
      "完成v1–v3监督PAC迭代与正式评估，\n建立冻结基线和预注册协议"],
     ["残差强化学习\n与正式评估", "2025.07–2026.06",
      "实现受约束残差RL，完成10族扰动\n预注册配对评估，撰写SCI论文"],
     ["硬件平台\n搭建与验证", "2026.07–2027.03",
      "搭建AUV实验平台，水池硬件在环实验，\n执行器退化注入，部署时延测量"],
     ["论文撰写\n与答辩", "2027.03–2027.06",
      "撰写硕士学位论文，准备答辩"]],
    [3.0, 3.0, 8.5],
    "表4.1  课题进度安排")

# ============================================================ CH5
h1("5  课题预期成果及创新性")

h2("5.1  课题研究特色及创新之处")
para("（1）权限分配接口：将学习组件的输出从6维推杆指令缩减为1维有界标量α，"
     "使安全审计和行为解释直接可行。这是接口层面的创新，使学习控制器的"
     "安全认证成为可能。")
para("（2）零初始化安全保证：残差头末层零初始化，部署起点与已验证基线"
     "逐位一致（全回合单元测试锁定），配合分层安全滤波栈，最坏情形退化为"
     "已验证的监督基线。这是安全层面的创新。")
para("（3）预注册配对评估协议：训练前冻结全部比较、种子和准则，120配对回合"
     "覆盖10族扰动，8方法同场竞技。这是评估方法学的创新。")
para("（4）极轻量化：整个学习系统16,290参数（63.6 KB），为同类深度学习"
     "控制器的1/4至1/20，可在嵌入式平台100 Hz实时运行。这是部署层面的优势。")

h2("5.2  预期研究成果")
para("（1）发表高水平SCI论文2篇（其中1篇已投/在投）。")
para("（2）申请发明专利2项。")
para("（3）完成水下机器人实验平台搭建及硬件在环验证。")

# ============================================================ CH6
h1("6  参考文献")
refs = [
    "T. I. Fossen. Handbook of Marine Craft Hydrodynamics and Motion Control[M]. 2nd ed. Wiley, 2021.",
    "G. Antonelli. Underwater Robots: Motion and Force Control of Vehicle-Manipulator Systems[M]. 3rd ed. Springer, 2014.",
    "V. I. Utkin. Sliding Modes in Control and Optimization[M]. Springer, 1992.",
    "E. F. Camacho and C. Bordons Alba. Model Predictive Control[M]. 2nd ed. Springer, 2013.",
    "Z. Zhao, X. Liu, T. Wang, Z. Zhou, and M. Zhang. Hybrid control scheme of nonlinear model prediction and adaptive terminal sliding mode for underwater vehicles based on threshold switching[J]. ISA Transactions, 2026, 176: 580–590.",
    "J. Gao, A. A. Proctor, Y. Shi, C. Bradley, and W. J. Dubay. Hierarchical model predictive slide-mode control of an AUV[J]. IEEE Journal of Oceanic Engineering, 2020, 45(2): 572–587.",
    "Y. Wang, et al. Adaptive neural network three-dimensional tracking control of AUVs[J]. Engineering Applications of Artificial Intelligence, 2025, 137: 109860.",
    "W. Xu. Low complexity adaptive neural network three-dimensional tracking control of autonomous underwater vehicles[J]. Engineering Applications of Artificial Intelligence, 2025, 137: 109860.",
    "N. Hovakimyan and C. Cao. L1 Adaptive Control Theory[M]. SIAM, 2010.",
    "Y. Fan, H. Dong, X. Zhao, and P. Denissenko. Path-following control of unmanned underwater vehicle based on an improved TD3 deep reinforcement learning[J]. IEEE Transactions on Control Systems Technology, 2024, 32(5): 1904–1919.",
    "L. Cai, K. Chang, and Y. Girdhar. Learning to swim: Reinforcement learning for 6-DOF control of thruster-driven autonomous underwater vehicles[C]//Proc. IEEE ICRA, 2025: 11286–11293.",
    "T. Johannink, S. Bahl, A. Nair, et al. Residual reinforcement learning for robot control[C]//Proc. IEEE ICRA, 2019: 6023–6029.",
    "T. Silver, K. Allen, A. Tenenbaum, and J. Koltun. Residual policy learning[J]. arXiv preprint arXiv:1812.06298, 2018.",
    "H. Wang, L. Feng, Y. Zhang, J. Zhou, and H. Du. Human-machine authority allocation in indirect cooperative shared steering control with TD3 reinforcement learning[J]. IEEE Transactions on Vehicular Technology, 2024, 73(6): 7576–7589.",
    "田家丞. 基于预测权限控制的水下机器人跟踪控制研究（前期研究报告）[R]. 浙江大学, 2026.",
    "L. Hewing, K. P. Wabersich, M. Menner, and M. N. Zeilinger. Learning-based model predictive control: Toward safe learning in control[J]. Annual Review of Control, Robotics, and Autonomous Systems, 2020, 3: 269–296.",
    "A. Romero, Y. Song, and D. Scaramuzza. Actor-critic model predictive control[C]//Proc. IEEE ICRA, 2024: 14777–14784.",
    "K. Nguyen, S. Schoedel, A. Alavilli, B. Plancher, and Z. Manchester. TinyMPC: Model-predictive control on resource-constrained microcontrollers[C]//Proc. IEEE ICRA, 2024.",
    "A. Vaswani, et al. Attention is all you need[C]//Advances in Neural Information Processing Systems, 2017, 30: 5998–6008.",
    "L. Chen, et al. Decision Transformer: Reinforcement learning via sequence modeling[C]//Advances in Neural Information Processing Systems, 2021, 34: 15084–15097.",
]
for i, ref in enumerate(refs, 1):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.3
    pf.space_after = Pt(3)
    pf.left_indent = Cm(0.75)
    pf.first_line_indent = Cm(-0.75)
    r = p.add_run(f"[{i}]  {ref}")
    _set_font(r, size=9.5)

# ============================================================ FORM PAGES
doc.add_page_break()
h1("二、导师评语及意见")
para("", after=40)
para("同意开题。", indent=False)
para("", after=30)
para("导师签名：                    年    月    日", indent=False,
     align=WD_ALIGN_PARAGRAPH.RIGHT)

h1("三、讨论意见")
para("", after=30)
para("同意通过开题报告。", indent=False)
para("", after=20)
para("时间：    年  月  日    地点：", indent=False)
para("", after=20)
para("答辩小组负责人签名：                    年    月    日", indent=False,
     align=WD_ALIGN_PARAGRAPH.RIGHT)

h1("四、研究所意见")
para("", after=30)
para("所长签名：                    年    月    日", indent=False,
     align=WD_ALIGN_PARAGRAPH.RIGHT)

h1("五、学科意见")
para("", after=30)
para("学科负责人签名：                    年    月    日", indent=False,
     align=WD_ALIGN_PARAGRAPH.RIGHT)

doc.save(OUT)
print(f"saved: {OUT}")
