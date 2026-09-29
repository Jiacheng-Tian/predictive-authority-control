# -*- coding: utf-8 -*-
"""Rebuild new slides with proper template fidelity + LaTeX formula images
+ parameter explanation tables."""
import io
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "manuscript_figures"
GAL = FIGS / "gallery"
TEMPLATE = ROOT / "paper" / "PAC_组会答辩_20260929.pptx"
OUT = ROOT / "paper" / "PAC_组会答辩_完整版.pptx"

INK = RGBColor(0x26, 0x26, 0x2E)
NAVY = RGBColor(0x51, 0x4D, 0x83)
RED = RGBColor(0xB6, 0x24, 0x2E)
ORANGE = RGBColor(0xE8, 0x73, 0x0C)
TEAL = RGBColor(0x3F, 0x7E, 0x72)
GRAY = RGBColor(0x6B, 0x6F, 0x85)
FONT = "微软雅黑"

shutil.copy(TEMPLATE, OUT)
prs = Presentation(OUT)
tmpl = list(prs.slides)
content_layout = tmpl[1].slide_layout

# Extract reusable template elements
logo_blob = None
freeform_xml = None
for sh in tmpl[1].shapes:
    if sh.shape_type == 13 and sh.left and sh.left > Inches(10):
        logo_blob = sh.image.blob
    if sh.shape_type == 5:  # FREEFORM = decorative header bar
        freeform_xml = sh._element

# Also get the page-number placeholder XML
pagenum_xml = None
for sh in tmpl[1].shapes:
    if sh.shape_type == 14:  # PLACEHOLDER
        pagenum_xml = sh._element
        break


def add_text(slide, x, y, w, h, lines, default_size=14):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(lines):
        txt, sz, bd, col = (item + (None,) * (4 - len(item)))[:4]
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = txt
        r.font.size = Pt(sz or default_size)
        r.font.bold = bool(bd)
        r.font.name = FONT
        if col:
            r.font.color.rgb = col
        p.space_after = Pt(3)
        p.line_spacing = 1.15
    return box


def add_pic(slide, path, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def add_table(slide, x, y, w, rows, col_w=None, fs=11):
    nr, nc = len(rows), len(rows[0])
    shape = slide.shapes.add_table(nr, nc, Inches(x), Inches(y),
                                   Inches(w), Inches(0.30 * nr))
    t = shape.table
    if col_w:
        tot = sum(col_w)
        for j, cw in enumerate(col_w):
            t.columns[j].width = Emu(int(Inches(w) * cw / tot))
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(fs)
                    r.font.name = FONT
                    r.font.bold = (i == 0)
                    r.font.color.rgb = INK
    return shape


def new_slide(page_num):
    """Create slide with exact template elements (freeform bar, nav, logo, etc.)."""
    s = prs.slides.add_slide(content_layout)

    # 1. Copy the FREEFORM decorative header bar
    if freeform_xml is not None:
        el = etree.fromstring(etree.tostring(freeform_xml))
        s.shapes._spTree.append(el)

    # 2. English header (exact position/size from template)
    add_text(s, 0.31, 0.09, 10.2, 0.6, [
        ("Predictive Authority Control: Learning When and How Much to Trust "
         "Baseline Controllers for Robust Underwater Robot Tracking",
         14, False, GRAY)])

    # 3. Nav tabs (exact positions from template)
    for label, x, w in [("个人简介", 3.14, 1.5), ("本科经历", 5.50, 2.2),
                        ("读研展望", 11.44, 1.3)]:
        box = s.shapes.add_textbox(Inches(x), Inches(0.20), Inches(w), Inches(0.4))
        r = box.text_frame.paragraphs[0].add_run()
        r.text = label
        r.font.name = FONT
        r.font.bold = True

    # 4. Logo
    if logo_blob:
        s.shapes.add_picture(io.BytesIO(logo_blob), Inches(11.07), Inches(0.06),
                             width=Inches(1.9))

    # 5. PAC title bar (exact position from template)
    add_text(s, 0.0, 0.81, 13.7, 0.4, [
        ("PAC：面向水下机器人鲁棒跟踪的基线控制器时序信任权重学习",
         16, True, NAVY)])

    # 6. Page number placeholder
    pn = s.shapes.add_textbox(Inches(12.71), Inches(6.97), Inches(0.6), Inches(0.4))
    r = pn.text_frame.paragraphs[0].add_run()
    r.text = str(page_num)

    return s


# ============================================================ S4: training
s = new_slide(4)
add_text(s, 0.5, 1.4, 12.3, 0.5, [("训练设置", 18, True, NAVY)])

# LaTeX formula image
add_pic(s, FIGS / "formula_reward.png", 0.5, 2.0, w=7.0)
add_text(s, 0.5, 3.0, 7.0, 0.4, [
    ("单步奖励（训练前冻结进 manifest）", 13, True, NAVY)])

# Parameter explanation table (left side)
param_table = [
    ["符号", "含义", "值"],
    ["w_e", "位置误差权重", "1.0"],
    ["w_h", "艏向误差权重", "0.3"],
    ["w_u", "控制量幅值权重", "0.01"],
    ["w_Δu", "控制量变化率权重", "0.01"],
    ["w_Δα", "权限修正惩罚", "0.05"],
    ["w_sat", "饱和惩罚", "0.1"],
    ["w_dl", "超时惩罚（仅记录）", "0.02"],
    ["w_c", "约束违反罚（最大）", "10.0"],
    ["s_pos", "位置归一化尺度", "0.1 m"],
    ["s_head", "艏向归一化尺度", "0.05 rad"],
]
add_table(s, 0.5, 3.5, 6.0, param_table, col_w=[1.0, 2.5, 1.5], fs=10)

# TD3 hyperparameter table (right side)
add_text(s, 7.0, 1.5, 5.8, 0.4, [("TD3 超参数", 14, True, NAVY)])
td3_table = [
    ["参数", "含义", "值"],
    ["δ_max", "Δα 上界", "0.1"],
    ["α_lr", "Actor 学习率", "1e-4"],
    ["c_lr", "Critic 学习率", "1e-3"],
    ["γ", "折扣因子", "0.99"],
    ["τ", "目标网络软更新率", "0.005"],
    ["d", "策略更新延迟", "每 2 步"],
    ["σ_t", "目标策略平滑噪声", "0.2"],
    ["c_t", "目标噪声裁剪", "0.5"],
    ["σ_e", "探索噪声", "0.1"],
    ["B", "批大小", "256"],
    ["N_warm", "热启动过渡数", "2×10⁵"],
    ["λ", "行为正则系数", "0.01"],
]
add_table(s, 7.0, 2.0, 5.8, td3_table, col_w=[1.0, 2.5, 1.3], fs=10)

# Formula images for residual head and blend
add_pic(s, FIGS / "formula_residual.png", 0.5, 6.5, w=5.5)
add_pic(s, FIGS / "formula_blend.png", 6.8, 6.5, w=5.5)

# ============================================================ S5: panorama
s = new_slide(5)
add_text(s, 0.3, 1.35, 12.7, 0.5, [("10 族扰动全景", 18, True, NAVY)])
pan = [
    ["族", "块", "训练范围", "测试范围", "环境\n种子"],
    ["结构化-恒定", "ID", "幅值 ×2.5，方向恒定", "—", "20"],
    ["结构化-正弦", "ID", "多频正弦叠加海流", "—", "20"],
    ["结构化-阶跃", "ID", "t=10s 海流方向突变", "—", "20"],
    ["质量-阻尼失配", "OOD", "mass ×[0.94,1.06]\ndamp ×[0.94,1.06]",
     "mass ×[1.10,1.25]\ndamp ×[0.75,0.90]", "5"],
    ["彩色噪声", "OOD", "ρ∈[0.90,0.98]\nσ∈[0.05,0.20]",
     "ρ∈[0.985,0.995]\nσ∈[0.25,0.40]", "5"],
    ["OU 海流", "OOD", "θ∈[0.2,1.0] rad/s\nσ∈[0.05,0.20]",
     "θ∈[1.2,2.0] rad/s\nσ∈[0.25,0.40]", "5"],
    ["随机频率-幅值", "OOD", "amp ∈[2.0,3.0]\nfreq∈[0.8,1.2]",
     "amp ∈[3.2,4.0]\nfreq∈[1.4,1.8]", "5"],
    ["执行器延迟+噪声", "OOD", "delay∈[2,4] 步\nnoise∈[0,0.02]",
     "delay∈[6,10] 步\nnoise∈[0.04,0.08]", "5"],
    ["快变 OU 海流", "OOD", "—",
     "θ∈[3.0,6.0] rad/s\nσ∈[0.30,0.50]", "5"],
    ["估计延迟", "OOD", "—",
     "delay∈[10,20] 步\nnoise∈[0.05,0.10]", "5"],
]
add_table(s, 0.3, 1.9, 12.7, pan, col_w=[2.0, 0.5, 3.0, 3.0, 0.8], fs=9)

# Parameter explanations
add_text(s, 0.3, 5.5, 12.7, 1.5, [
    ("参数说明：mass=质量缩放因子 | damp=阻尼缩放因子 | ρ=AR(1) 自相关系数 | "
     "σ=噪声标准差 | θ=OU 回归速率 | amp=幅值缩放 | freq=频率缩放 | "
     "delay=传感器/执行器延迟步数 | noise=噪声标准差", 11, False, GRAY),
    ("「OOD」= 测试范围系统性超出训练范围 | 85 ID + 35 OOD = 120 配对回合 × 14 组合 = 1,680 次滚动", 12, True, NAVY),
])

# ============================================================ S6-S13: gallery
GAL_PAGES = [
    ("结构化海流（ID）",
     ["gallery_00_structured_seen.png", "gallery_01_structured_seen.png",
      "gallery_02_structured_seen.png"],
     "恒定流×2.5（方向不变）| 多频正弦叠加 | 阶跃 t=10s 突变",
     "共 60 回合（3 场景 × 20 seeds）",
     "全方法贴近参考；SMC 带可见偏移"),
    ("质量-阻尼失配（OOD）",
     ["gallery_03_mass_damping_mismatch_unseen.png"],
     "训练: mass ×[0.94,1.06], damp ×[0.94,1.06]\n"
     "测试: mass ×[1.10,1.25], damp ×[0.75,0.90]\n"
     "mass=航行器质量缩放 | damp=水动力阻尼缩放",
     "测试范围系统性超出训练采样，检验参数外推能力",
     "残差策略将跟踪误差从 0.161 降至 0.116 m"),
    ("彩色噪声（OOD）",
     ["gallery_04_colored_noise_unseen.png"],
     "训练: ρ∈[0.90,0.98], σ∈[0.05,0.20]\n"
     "测试: ρ∈[0.985,0.995], σ∈[0.25,0.40]\n"
     "ρ=AR(1) 自相关系数（越大越持久）| σ=扰动标准差",
     "时间相关性与幅值同时增大，海流更难预测",
     "残差策略在此族保持稳健"),
    ("OU 海流（OOD）",
     ["gallery_05_ou_current_unseen.png"],
     "训练: θ∈[0.2,1.0] rad/s, σ∈[0.05,0.20]\n"
     "测试: θ∈[1.2,2.0] rad/s, σ∈[0.25,0.40]\n"
     "θ=Ornstein-Uhlenbeck 回归速率 | σ=随机力强度",
     "时间尺度加快（θ 增大 = 更快回归均值）",
     "残差策略适应更快变化的海流"),
    ("随机频率-幅值（OOD）",
     ["gallery_06_random_freq_amp_unseen.png"],
     "训练: amp∈[2.0,3.0], freq∈[0.8,1.2]\n"
     "测试: amp∈[3.2,4.0], freq∈[1.4,1.8]\n"
     "amp=幅值缩放因子 | freq=频率缩放因子",
     "频率与幅值同时超出训练范围",
     "本族全体方法退化，残差策略退化最少"),
    ("执行器延迟+噪声（OOD）★ 核心场景",
     ["gallery_07_actuator_delay_noise_unseen.png"],
     "训练: delay∈[2,4] 步, noise∈[0,0.02]\n"
     "测试: delay∈[6,10] 步, noise∈[0.04,0.08]\n"
     "delay=推进器响应延迟（1步=10ms）| noise=加性噪声标准差",
     "MPC 因模型失配持续输出发散指令（0.822 m）\n"
     "残差策略将误差约束至 0.090 m（−89%）",
     "监督策略 α 振荡至 0.9+；残差策略稳定于 [0.4,0.6]"),
    ("快变 OU 海流（OOD）",
     ["gallery_08_fast_ou_unseen.png"],
     "仅评估（无训练对应）:\nθ∈[3.0,6.0] rad/s, σ∈[0.30,0.50]\n"
     "θ 远超全部训练族（最快训练值 θ=1.0）",
     "时间尺度为最快训练族的 3–6 倍",
     "极端外推场景，检验最远泛化边界"),
    ("估计延迟（OOD）",
     ["gallery_09_estimation_delay_unseen.png"],
     "仅评估（无训练对应）:\ndelay∈[10,20] 步, noise∈[0.05,0.10]\n"
     "delay=传感器数据滞后 | noise=观测噪声",
     "传感器信息延迟 100–200 ms + 噪声",
     "检验因果估计下的控制鲁棒性"),
]

for idx, (title, files, params, interp, result) in enumerate(GAL_PAGES):
    s = new_slide(6 + idx)
    add_text(s, 0.3, 1.35, 12.7, 0.5, [(title, 17, True, NAVY)])
    if len(files) == 3:
        for j, fn in enumerate(files):
            add_pic(s, GAL / fn, 0.5 + j * 4.3, 2.0, h=5.0)
        add_text(s, 0.5, 7.15, 12.3, 0.4, [(params, 11)])
    else:
        add_pic(s, GAL / files[0], 0.5, 2.0, h=5.0)
        add_text(s, 5.5, 2.2, 7.3, 5.2, [
            ("参数范围与含义", 15, True, NAVY),
            (params, 12),
            ("", 8),
            ("物理意义", 15, True, ORANGE),
            (interp, 12),
            ("", 8),
            ("结果", 15, True, TEAL),
            (result, 12, True),
            ("", 6),
            ("每回合 2,100 步（21 s）· 6 方法 × 5 种子", 11, False, GRAY),
        ])

# ============================================================ S14: verdict
s = new_slide(14)
add_text(s, 0.5, 1.4, 12.3, 5.5, [
    ("判定（六项注册准则）", 17, True, NAVY),
    ("3 过：不以艏向/饱和/约束换增益 · 消融可分离 · 五种子方向稳定", 14),
    ("2 挂：整体非劣性 ID +0.4% · 超时不恶化 +0.107", 14),
    ("1 未评估：部署推理时延 = 硬件第一门槛", 14), ("", 10),
    ("注册的下一步", 17, True, ORANGE),
    ("① 直接 RL 对照臂　② α 扫描　③ 残差第二轮　④ 隔离超时　⑤ 硬件在环", 14),
    ("", 10),
    ("结论：残差策略以可证明的安全结构买到 OOD 鲁棒性 −28% 与"
     "执行器退化免疫（0.822→0.090 m）", 15, True, TEAL),
])

# ============================================================ reorder
sldIdLst = prs.slides._sldIdLst
ids = list(sldIdLst)
# [0..7]=original 8, [8]=training, [9]=panorama, [10..17]=gallery, [18]=verdict
order = [0, 1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 3, 4, 5, 6, 18, 7]
new_ids = [ids[i] for i in order]
for el in list(sldIdLst):
    sldIdLst.remove(el)
for el in new_ids:
    sldIdLst.append(el)

# page numbers
final = list(prs.slides)
for i, s in enumerate(final):
    for sh in s.shapes:
        if (sh.has_text_frame and sh.left and sh.left > Inches(12.5)
                and sh.top and sh.top > Inches(6.8)):
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    r.text = str(i + 1)
            break

prs.save(OUT)
print(f"saved {OUT} with {len(final)} slides")
