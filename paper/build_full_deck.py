# -*- coding: utf-8 -*-
"""Build 20-page deck from the user's 8-page template (PAC_组会答辩_20260929.pptx).

Keeps all 8 original slides, adds 12 new slides (training setup, panorama,
8 gallery, verdict) with recreated header elements, then reorders.
"""
import io
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

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
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "微软雅黑"

shutil.copy(TEMPLATE, OUT)
prs = Presentation(OUT)
tmpl_slides = list(prs.slides)  # 8 slides
content_layout = tmpl_slides[1].slide_layout
logo_blob = None
for sh in tmpl_slides[1].shapes:
    if sh.shape_type == 13 and sh.left and sh.left > Inches(10) and sh.top and sh.top < Inches(0.8):
        logo_blob = sh.image.blob
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
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def add_table(slide, x, y, w, rows_data, col_widths=None, font_size=11):
    n_rows = len(rows_data)
    n_cols = len(rows_data[0]) if rows_data else 0
    shape = slide.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y),
                                   Inches(w), Inches(0.30 * n_rows))
    table = shape.table
    if col_widths:
        total = sum(col_widths)
        for j, cw in enumerate(col_widths):
            table.columns[j].width = Emu(int(Inches(w) * cw / total))
    for i, row in enumerate(rows_data):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(font_size)
                    r.font.name = FONT
                    r.font.bold = (i == 0)
                    r.font.color.rgb = INK
    return shape


def new_content_slide():
    s = prs.slides.add_slide(content_layout)
    add_text(s, 0.3, 0.05, 10.0, 0.5, [
        ("Predictive Authority Control: Learning When and How Much to Trust "
         "Baseline Controllers for Robust Underwater Robot Tracking", 10, False, GRAY)])
    for label, x in [("个人简介", 3.1), ("本科经历", 5.5), ("读研展望", 11.4)]:
        box = s.shapes.add_textbox(Inches(x), Inches(0.15), Inches(1.5), Inches(0.35))
        r = box.text_frame.paragraphs[0].add_run()
        r.text = label
        r.font.size = Pt(11)
        r.font.bold = True
        r.font.name = FONT
        r.font.color.rgb = RGBColor(0x8C, 0x87, 0xC8)
    add_text(s, 0.0, 0.8, 13.4, 0.4, [
        ("PAC：面向水下机器人鲁棒跟踪的基线控制器时序信任权重学习", 15, True, NAVY)])
    if logo_blob:
        prs.slides[-1].shapes.add_picture(io.BytesIO(logo_blob),
                                          Inches(11.1), Inches(0.1),
                                          width=Inches(1.9))
    pn = s.shapes.add_textbox(Inches(12.7), Inches(7.0), Inches(0.6), Inches(0.35))
    r = pn.text_frame.paragraphs[0].add_run()
    r.text = "•"
    r.font.size = Pt(10)
    return s


# ---- S9: training setup ----
s = new_content_slide()
add_text(s, 0.5, 1.4, 12.3, 0.5, [("训练设置：奖励函数与超参数", 18, True, NAVY)])
add_text(s, 0.5, 2.0, 6.0, 5.0, [
    ("单步奖励（训练前冻结）", 14, True, NAVY),
    ("r_t = −w_e·(e_pos/s_pos) − w_h·(e_head/s_head)", 13),
    ("      − w_u·‖u‖² − w_Δu·‖Δu‖² − w_Δα·|Δα|", 13),
    ("      − w_sat·ρ_sat − w_dl·ρ_dl − w_c·ρ_con", 13),
    ("", 8),
    ("权重", 13, True, NAVY),
    ("w_e=1.0  w_h=0.3  w_u=0.01  w_Δu=0.01", 12),
    ("w_Δα=0.05  w_sat=0.1  w_dl=0.02  w_con=10.0", 12),
    ("归一化：s_pos=0.1 m  s_head=0.05 rad", 12),
    ("", 8),
    ("约束罚 w_con=10 最大：安全 > 精度", 12, True, RED),
    ("超时项 ρ_dl 仅记录：RL 不改 MPC 调度", 12, False, GRAY),
])
add_table(s, 7.0, 2.0, 5.8, [
    ["参数", "值"], ["算法", "TD3"], ["Δα 上界", "0.1"],
    ["actor lr", "1e-4"], ["critic lr", "1e-3"], ["γ", "0.99"],
    ["τ", "0.005"], ["策略延迟", "每2步"], ["目标噪声", "0.2/0.5"],
    ["探索噪声", "0.1"], ["批大小", "256"], ["热启动", "2×10⁵"],
    ["行为正则", "0.01"], ["训练步数", "2×10⁵"],
], col_widths=[2.5, 2.0], font_size=11)

# ---- S10: panorama ----
s = new_content_slide()
add_text(s, 0.3, 1.3, 12.7, 0.5, [("10 族扰动全景", 18, True, NAVY)])
pan = [["族", "块", "训练范围", "测试范围", "种子"],
       ["结构化-恒定", "ID", "幅值×2.5 恒定流", "—", "20"],
       ["结构化-正弦", "ID", "多频正弦叠加", "—", "20"],
       ["结构化-阶跃", "ID", "t=10s 换向", "—", "20"],
       ["质量-阻尼失配", "OOD", "mass 0.94–1.06", "mass 1.10–1.25", "5"],
       ["彩色噪声", "OOD", "ρ 0.90–0.98, σ 0.05–0.20", "ρ 0.985–0.995, σ 0.25–0.40", "5"],
       ["OU 海流", "OOD", "θ 0.2–1.0, σ 0.05–0.20", "θ 1.2–2.0, σ 0.25–0.40", "5"],
       ["随机频率-幅值", "OOD", "amp 2.0–3.0, freq 0.8–1.2", "amp 3.2–4.0, freq 1.4–1.8", "5"],
       ["执行器延迟+噪声", "OOD", "delay 2–4步, noise 0–0.02", "delay 6–10步, noise 0.04–0.08", "5"],
       ["快变 OU", "OOD", "—", "θ 3.0–6.0, σ 0.30–0.50", "5"],
       ["估计延迟", "OOD", "—", "delay 10–20步, noise 0.05–0.10", "5"],
       ]
add_table(s, 0.3, 1.85, 12.7, pan, col_widths=[2.2, 0.6, 3.5, 3.5, 0.8], font_size=10)
add_text(s, 0.3, 5.6, 12.7, 1.0, [
    ("85 ID + 35 OOD = 120 配对回合 × 14 组合 = 1,680 次滚动", 13, True, NAVY)])

# ---- S11-S18: gallery ----
GAL_PAGES = [
    ("结构化海流（ID）",
     ["gallery_00_structured_seen.png", "gallery_01_structured_seen.png",
      "gallery_02_structured_seen.png"],
     "恒定×2.5 | 正弦多频 | 阶跃 t=10s", "全方法贴近参考"),
    ("质量-阻尼失配（OOD）", ["gallery_03_mass_damping_mismatch_unseen.png"],
     "训练: mass ×0.94–1.06, damp ×0.94–1.06\n测试: mass ×1.10–1.25, damp ×0.75–0.90",
     "参数外推能力检验"),
    ("彩色噪声（OOD）", ["gallery_04_colored_noise_unseen.png"],
     "训练: ρ=0.90–0.98, σ=0.05–0.20\n测试: ρ=0.985–0.995, σ=0.25–0.40",
     "时间相关性与幅值同时增大"),
    ("OU 海流（OOD）", ["gallery_05_ou_current_unseen.png"],
     "训练: θ=0.2–1.0, σ=0.05–0.20\n测试: θ=1.2–2.0, σ=0.25–0.40",
     "时间尺度加快的随机海流"),
    ("随机频率-幅值（OOD）", ["gallery_06_random_freq_amp_unseen.png"],
     "训练: amp 2.0–3.0, freq 0.8–1.2\n测试: amp 3.2–4.0, freq 1.4–1.8",
     "频率与幅值同时超范围"),
    ("执行器延迟+噪声（OOD）★", ["gallery_07_actuator_delay_noise_unseen.png"],
     "训练: delay 2–4步, noise 0–0.02\n测试: delay 6–10步, noise 0.04–0.08",
     "核心场景：MPC 0.822m → 残差 0.090m"),
    ("快变 OU 海流（OOD）", ["gallery_08_fast_ou_unseen.png"],
     "仅评估: θ=3.0–6.0, σ=0.30–0.50", "时间尺度远超全部训练族"),
    ("估计延迟（OOD）", ["gallery_09_estimation_delay_unseen.png"],
     "仅评估: delay 10–20步, noise 0.05–0.10", "传感器滞后+噪声"),
]

for title, files, params, key in GAL_PAGES:
    s = new_content_slide()
    add_text(s, 0.3, 1.35, 12.7, 0.5, [(title, 17, True, NAVY)])
    if len(files) == 3:
        for j, fn in enumerate(files):
            add_pic(s, GAL / fn, 0.5 + j * 4.3, 2.0, h=5.0)
        add_text(s, 0.5, 7.15, 12.3, 0.4, [(params + " | " + key, 11)])
    else:
        add_pic(s, GAL / files[0], 0.5, 2.0, h=5.0)
        add_text(s, 5.5, 2.3, 7.3, 4.8, [
            ("参数范围", 15, True, NAVY), (params, 13), ("", 8),
            ("关键读图", 15, True, ORANGE), (key, 13), ("", 8),
            ("每回合 2,100 步（21 s）\n6 方法 × 5 种子", 12, False, GRAY),
        ])

# ---- S19: verdict ----
s = new_content_slide()
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
# Current indices: [0..7]=original 8 slides, [8]=training, [9]=panorama,
#                  [10..17]=gallery, [18]=verdict
# Desired: [0]=cover, [1]=arch, [2]=network, [8]=training, [9]=panorama,
#          [10..17]=gallery, [3]=traj, [4]=results, [5]=case, [6]=quality,
#          [18]=verdict, [7]=closing

sldIdLst = prs.slides._sldIdLst
ids = list(sldIdLst)
order = [0, 1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 3, 4, 5, 6, 18, 7]
new_ids = [ids[i] for i in order]

for el in list(sldIdLst):
    sldIdLst.remove(el)
for el in new_ids:
    sldIdLst.append(el)

# ============================================================ page numbers
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
