# -*- coding: utf-8 -*-
"""Build 20-page defense deck from scratch using template's layout.
Avoids XML-level slide copying (which corrupts the package)."""
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "manuscript_figures"
GAL = FIGS / "gallery"
TEMPLATE = Path(r"C:\Users\admin\WPSDrive\1772108122\WPS企业云盘\浙大校友"
                r"\我的企业文档\Nautilus\PAC_20260617.pptx")
OUT = ROOT / "paper" / "PAC_组会答辩_完整版.pptx"

INK = RGBColor(0x26, 0x26, 0x2E)
NAVY = RGBColor(0x51, 0x4D, 0x83)
RED = RGBColor(0xB6, 0x24, 0x2E)
ORANGE = RGBColor(0xE8, 0x73, 0x0C)
TEAL = RGBColor(0x3F, 0x7E, 0x72)
GRAY = RGBColor(0x6B, 0x6F, 0x85)
LAV = RGBColor(0xEF, 0xED, 0xF5)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "微软雅黑"

import shutil as _sh
_sh.copy(TEMPLATE, OUT)
prs = Presentation(OUT)
prs.slide_width = prs.slide_width  # keep template size
layouts = prs.slide_masters[0].slide_layouts
blank_layout = None
for l in layouts:
    if 'blank' in l.name.lower() or '空白' in l.name or 'Blank' in l.name:
        blank_layout = l
        break
if blank_layout is None:
    blank_layout = layouts[6] if len(layouts) > 6 else layouts[-1]

# Remove all template slides properly
from pptx.oxml.ns import qn as _qn
_sldIdLst = prs.slides._sldIdLst
for _sldId in list(_sldIdLst):
    _rId = _sldId.get(_qn('r:id'))
    prs.part.drop_rel(_rId)
    _sldIdLst.remove(_sldId)


def new_slide():
    return prs.slides.add_slide(blank_layout)


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


def header(slide, page_num):
    """Recreate the template's header: nav bar, title bar, eng header, page num."""
    # English header (top-left)
    add_text(slide, 0.15, 0.05, 9.5, 0.5, [
        ("Predictive Authority Control: Learning When and How Much to Trust "
         "Baseline Controllers for Robust Underwater Robot Tracking", 10,
         False, GRAY)])
    # Nav tabs (top, small)
    for label, x, w in [("个人简介", 2.8, 1.5), ("本科经历", 5.2, 2.2),
                        ("读研展望", 10.9, 1.3)]:
        box = slide.shapes.add_textbox(Inches(x), Inches(0.1), Inches(w),
                                       Inches(0.35))
        p = box.text_frame.paragraphs[0]
        r = p.add_run()
        r.text = label
        r.font.size = Pt(11)
        r.font.bold = True
        r.font.name = FONT
        r.font.color.rgb = RGBColor(0x8C, 0x87, 0xC8)
    # PAC title bar
    add_text(slide, 0.0, 0.72, 13.4, 0.4, [
        ("PAC：面向水下机器人鲁棒跟踪的基线控制器时序信任权重学习", 15, True, NAVY)])
    # Page number (bottom-right)
    pn = slide.shapes.add_textbox(Inches(12.5), Inches(7.05), Inches(0.6),
                                  Inches(0.35))
    r = pn.text_frame.paragraphs[0].add_run()
    r.text = str(page_num)
    r.font.size = Pt(10)
    r.font.name = FONT


# ============================================================ S1: cover
s = new_slide()
# Dark background
from pptx.enum.shapes import MSO_SHAPE
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width,
                        prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0x1B, 0x1F, 0x3B)
bg.line.fill.background()
add_text(s, 0.7, 2.3, 11.9, 1.0, [
    ("预测权限控制（PAC）阶段答辩", 40, True, WHITE)])
add_text(s, 0.7, 3.5, 11.9, 0.7, [
    ("受约束残差强化学习修复控制器信任：OOD ↓28%，代价如实报告", 20, False,
     RGBColor(0x8B, 0x87, 0xC8))])
add_text(s, 0.7, 5.5, 5, 0.5, [("姓名：田家丞", 16, True, WHITE)])
add_text(s, 0.7, 6.2, 5, 0.5, [("日期：2026年9月29日", 16, True, WHITE)])

# ============================================================ S2: overview
s = new_slide()
header(s, 2)
add_text(s, 0.5, 1.4, 12.3, 0.5, [("一、工作概况", 18, True, NAVY)])
add_text(s, 0.5, 2.0, 12.3, 1.2, [
    ("架构定型为「SMC + MPC 专家 + 冻结 Transformer 编码器 + 零初始化残差头」。"
     "学习组件只输出有界权限修正 Δα∈[−0.1, +0.1]，推进器指令始终由经典专家产生。"
     "10 族扰动 × 14 方法-种子组合 × 120 配对回合 = 1,680 次滚动 ≈ 353 万控制步。", 14)])
add_text(s, 0.5, 3.5, 5.8, 3.5, [
    ("本轮更新（顶刊体例）", 15, True, NAVY),
    ("• 网络级架构图与训练图（生图+矢量双版）", 13),
    ("• 10 族扰动逐场景展示（本 PPT 新增 8 页）", 13),
    ("• 训练方程与超参数全透明化", 13),
    ("• 控制质量图：饱和率/代价全方法最低", 13),
    ("• 方法名改描述式：Supervised / Residual PAC", 13),
])
add_text(s, 6.8, 3.5, 5.8, 3.5, [
    ("评估规模（预注册、配对）", 15, True, NAVY),
    ("• 120 配对回合（85 ID + 35 OOD）", 13),
    ("• 模型种子级 t(4) 95% CI + bootstrap", 13),
    ("• 注册准则先于训练冻结", 13),
    ("• 残差相关六项准则 3 过 / 2 挂 / 1 未评", 13),
    ("• 代价如实报告：ID +0.4% / 超时 +0.107", 13),
])

# ============================================================ S3: problem
s = new_slide()
header(s, 3)
add_text(s, 0.5, 1.4, 12.3, 1.2, [
    ("问题：执行器退化下 MPC 按错误模型持续输出「自信的失效指令」，"
     "OOD 执行器族误差 0.822 m ≈ 正常 9 倍，逼近 1 m 任务失败线。"
     "仲裁问题（每一时刻信任谁、信多少）随海况连续变化，人工切换规则无法枚举。", 14)])
add_text(s, 0.5, 2.8, 12.3, 4.2, [
    ("核心思想：权限 α", 17, True, NAVY),
    ("学习系统不碰推进器，只输出 α∈[0,1]；u = (1−α)·u_SMC + α·u_MPC。", 14),
    ("受约束残差：Δα = 0.1·tanh(·)，末层零初始化", 17, True, ORANGE),
    ("冻结监督策略（14,113 参数），仅训练 2,177 参数残差头；"
     "开机行为与已部署基线逐位一致。最坏情形 = 基线本身。", 14),
    ("四层安全栈", 17, True, TEAL),
    ("① |Δα|≤0.1  ② clip/平滑/率限  ③ 专家失效强制回主控  ④ 数值异常回退冻结基线", 14),
])

# ============================================================ S4: architecture
s = new_slide()
header(s, 4)
add_pic(s, FIGS / "fig1_system_architecture_ai.png", 0.55, 1.5, w=8.9)
add_text(s, 9.7, 1.8, 3.4, 5.2, [
    ("读图要点", 16, True, NAVY),
    ("绿区=已验证固定组件\n紫区=可审计学习组件\n灰区=执行级", 13),
    ("编码器冻结（❄ 14,113 参数）\n残差头可训练（▲ 2,177 参数、零初始化）", 13),
    ("α_t = clip[0,1](α̂_t + Δα_t)", 13, True, RED),
    ("底部安全条带：\n有界Δα → clip/平滑/率限\n→ 强制主控 → 数值回退", 13),
])

# ============================================================ S5: network
s = new_slide()
header(s, 5)
add_pic(s, FIGS / "fig_network_training_ai.png", 0.3, 1.5, w=7.9)
add_text(s, 8.5, 1.8, 4.5, 5.2, [
    ("两阶段训练", 16, True, NAVY),
    ("阶段 1 · 监督初始化", 14, True, NAVY),
    ("oracle 标签：5 点 α 网格\n18,900 样本 · AdamW · 180 轮", 12),
    ("阶段 2 · 受约束残差 TD3", 14, True, ORANGE),
    ("骨干冻结 · 残差头 32→64→GELU→1\n2×10⁵ 热启动 · 行为正则 0.01", 12),
    ("零初始化 ⇒ 开机 = 基线", 13, True, TEAL),
])

# ============================================================ S6: training setup
s = new_slide()
header(s, 6)
add_text(s, 0.5, 1.4, 12.3, 0.5, [("训练设置：奖励函数与超参数", 18, True, NAVY)])
add_text(s, 0.5, 2.0, 6.0, 5.0, [
    ("单步奖励（训练前冻结进 manifest）", 14, True, NAVY),
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
    ["参数", "值"],
    ["算法", "TD3"],
    ["Δα 上界", "0.1"],
    ["actor 学习率", "1×10⁻⁴"],
    ["critic 学习率", "1×10⁻³"],
    ["折扣 γ", "0.99"],
    ["软更新 τ", "0.005"],
    ["策略延迟", "每 2 步"],
    ["目标噪声", "0.2 / 0.5"],
    ["探索噪声", "0.1"],
    ["批大小", "256"],
    ["热启动", "2×10⁵"],
    ["行为正则", "0.01"],
    ["训练步数", "2×10⁵"],
], col_widths=[2.5, 2.0], font_size=11)

# ============================================================ S7: panorama
s = new_slide()
header(s, 7)
add_text(s, 0.3, 1.3, 12.7, 0.5, [("10 族扰动全景（配对评估网格）", 18, True, NAVY)])
pan = [["族", "块", "训练范围", "测试范围", "种子"],
       ["结构化-恒定", "ID", "幅值 ×2.5 恒定流", "—", "20"],
       ["结构化-正弦", "ID", "多频正弦叠加", "—", "20"],
       ["结构化-阶跃", "ID", "t=10s 换向", "—", "20"],
       ["质量-阻尼失配", "OOD", "mass 0.94–1.06, damp 0.94–1.06", "mass 1.10–1.25, damp 0.75–0.90", "5"],
       ["彩色噪声", "OOD", "ρ 0.90–0.98, σ 0.05–0.20", "ρ 0.985–0.995, σ 0.25–0.40", "5"],
       ["OU 海流", "OOD", "θ 0.2–1.0, σ 0.05–0.20", "θ 1.2–2.0, σ 0.25–0.40", "5"],
       ["随机频率-幅值", "OOD", "amp 2.0–3.0, freq 0.8–1.2", "amp 3.2–4.0, freq 1.4–1.8", "5"],
       ["执行器延迟+噪声", "OOD", "delay 2–4 步, noise 0–0.02", "delay 6–10 步, noise 0.04–0.08", "5"],
       ["快变 OU", "OOD", "—", "θ 3.0–6.0, σ 0.30–0.50", "5"],
       ["估计延迟", "OOD", "—", "delay 10–20 步, noise 0.05–0.10", "5"],
       ]
add_table(s, 0.3, 1.85, 12.7, pan, col_widths=[2.2, 0.6, 3.5, 3.5, 0.8], font_size=10)
add_text(s, 0.3, 5.6, 12.7, 1.2, [
    ("85 ID + 35 OOD = 120 配对回合 × 14 组合 = 1,680 次滚动 ≈ 353 万控制步", 13, True, NAVY),
    ("「OOD」= 相对扰动族采样：测试范围系统性超出训练范围，检验分布外泛化", 12, False, GRAY),
])

# ============================================================ S8-S15: gallery
GAL_PAGES = [
    ("结构化海流（ID）：恒定/正弦/阶跃",
     ["gallery_00_structured_seen.png", "gallery_01_structured_seen.png",
      "gallery_02_structured_seen.png"],
     "恒定 ×2.5 | 正弦多频 | 阶跃 t=10s | 60 回合", "全方法贴近参考；SMC 偏移"),
    ("质量-阻尼失配（OOD）", ["gallery_03_mass_damping_mismatch_unseen.png"],
     "训练: mass ×0.94–1.06, damp ×0.94–1.06\n测试: mass ×1.10–1.25, damp ×0.75–0.90",
     "测试范围超出训练采样\n检验参数外推能力"),
    ("彩色噪声（OOD）", ["gallery_04_colored_noise_unseen.png"],
     "训练: ρ=0.90–0.98, σ=0.05–0.20\n测试: ρ=0.985–0.995, σ=0.25–0.40",
     "时间相关性与幅值同时增大"),
    ("OU 海流（OOD）", ["gallery_05_ou_current_unseen.png"],
     "训练: θ=0.2–1.0, σ=0.05–0.20\n测试: θ=1.2–2.0, σ=0.25–0.40",
     "Ornstein-Uhlenbeck 时间尺度加快"),
    ("随机频率-幅值（OOD）", ["gallery_06_random_freq_amp_unseen.png"],
     "训练: amp 2.0–3.0, freq 0.8–1.2\n测试: amp 3.2–4.0, freq 1.4–1.8",
     "频率与幅值同时超范围"),
    ("执行器延迟+噪声（OOD）", ["gallery_07_actuator_delay_noise_unseen.png"],
     "训练: delay 2–4 步, noise 0–0.02\n测试: delay 6–10 步, noise 0.04–0.08",
     "★ 本文核心场景\nMPC 0.822m → 残差 0.090m"),
    ("快变 OU 海流（OOD）", ["gallery_08_fast_ou_unseen.png"],
     "仅评估: θ=3.0–6.0, σ=0.30–0.50",
     "时间尺度远超全部训练族"),
    ("估计延迟（OOD）", ["gallery_09_estimation_delay_unseen.png"],
     "仅评估: delay 10–20 步, noise 0.05–0.10",
     "传感器滞后+噪声，检验因果估计"),
]

for idx, (title, files, params, key) in enumerate(GAL_PAGES):
    s = new_slide()
    header(s, 8 + idx)
    add_text(s, 0.3, 1.35, 12.7, 0.5, [(title, 17, True, NAVY)])
    if len(files) == 3:
        for j, fn in enumerate(files):
            add_pic(s, GAL / fn, 0.5 + j * 4.3, 2.0, h=5.0)
        add_text(s, 0.5, 7.15, 12.3, 0.4, [(params + " | " + key, 11)])
    else:
        add_pic(s, GAL / files[0], 0.5, 2.0, h=5.0)
        add_text(s, 5.5, 2.3, 7.3, 4.8, [
            ("参数范围", 15, True, NAVY),
            (params, 13),
            ("", 10),
            ("关键读图", 15, True, ORANGE),
            (key, 13),
            ("", 10),
            ("每回合 2,100 步（21 s）\n6 方法 × 5 种子配对评估", 12, False, GRAY),
        ])

# ============================================================ S16: main results
s = new_slide()
header(s, 16)
add_text(s, 0.3, 1.35, 12.7, 0.9, [
    ("主结果：OOD RMSE 0.161→0.116 m（−28%，CI [−0.074,−0.015]）。"
     "固定 α=0.5 达 0.120 m——残差价值集中在最需要自适应处"
     "（执行器族 −28% vs 固定混合）。ID +0.4% 如实报告。", 13)])
add_pic(s, FIGS / "fig4_closed_loop_results.png", 1.95, 2.45, w=9.4)

# ============================================================ S17: actuator case
s = new_slide()
header(s, 17)
add_text(s, 0.3, 1.35, 12.7, 0.9, [
    ("机制：残差学到的是「否决」监督策略的越界。监督 α 推过 0.9 并振荡（31.2%步数）；"
     "残差 97.0% 步数稳在 [0.4,0.6]。箱线图：0.82±0.14 vs 0.54±0.03。", 13)])
add_pic(s, FIGS / "fig5_actuator_case.png", 3.4, 2.4, w=6.1)

# ============================================================ S18: control quality
s = new_slide()
header(s, 18)
add_text(s, 0.3, 1.35, 12.7, 0.5, [("控制质量与代价", 17, True, NAVY)])
add_pic(s, FIGS / "fig_control_quality.png", 0.3, 2.0, w=7.0)
add_pic(s, FIGS / "fig6_cost_tradeoff.png", 0.3, 4.95, w=7.0)
add_text(s, 7.6, 2.1, 5.0, 5.0, [
    ("OOD 控制质量", 14, True, NAVY),
    ("饱和步占比 0.126：最低", 12),
    ("控制代价 31.9：最低", 12),
    ("配对代价 −1.78 [−2.31,−1.26]", 12),
    ("", 8),
    ("代价（如实报告）", 14, True, RED),
    ("ID +0.4%、艏向 +4.3%", 12),
    ("超时 0.69→0.80", 12),
    ("回退 0.3–0.5%，成功率持平", 12),
])

# ============================================================ S19: verdict
s = new_slide()
header(s, 19)
add_text(s, 0.5, 1.4, 12.3, 5.5, [
    ("判定（六项注册准则）", 17, True, NAVY),
    ("3 过：不以艏向/饱和/约束换增益 · 消融可分离 · 五种子方向稳定", 14),
    ("2 挂：整体非劣性 ID +0.4% · 超时不恶化 +0.107", 14),
    ("1 未评估：部署推理时延 = 硬件第一门槛", 14),
    ("", 10),
    ("注册的下一步", 17, True, ORANGE),
    ("① 直接 RL 对照臂　② α 扫描　③ 残差第二轮　④ 隔离超时　⑤ 硬件在环", 14),
    ("", 10),
    ("结论：残差策略以可证明的安全结构买到 OOD 鲁棒性 −28% 与"
     "执行器退化免疫（0.822→0.090 m），每一项代价与冻结准则对照测量。", 15, True, TEAL),
])

# ============================================================ S20: closing
s = new_slide()
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width,
                        prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0x1B, 0x1F, 0x3B)
bg.line.fill.background()
add_text(s, 0.7, 2.5, 11.9, 1.0, [("感谢各位老师，恳请批评指正！", 36, True, WHITE)])
add_text(s, 0.7, 3.7, 11.9, 0.6, [("THANK YOU", 20, False,
                                   RGBColor(0x8B, 0x87, 0xC8))])
add_text(s, 0.7, 5.3, 5, 0.5, [("姓名：田家丞", 16, True, WHITE)])

prs.save(OUT)
print(f"saved {OUT} with {len(prs.slides._sldIdLst)} slides")
