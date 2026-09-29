# -*- coding: utf-8 -*-
"""Rebuild the defense deck on the WPS 组会 template (clone-and-fill).

Keeps the template's visual identity (nav tabs, PAC title bar, university
logo, page-number placeholders, fonts) and replaces body content with the
current WM-free defense narrative + updated figures.
"""
import copy
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "manuscript_figures"
TEMPLATE = Path(r"C:\Users\admin\WPSDrive\1772108122\WPS企业云盘\浙大校友"
                r"\我的企业文档\Nautilus\PAC_20260617.pptx")
OUT = ROOT / "paper" / "PAC_组会答辩_20260929.pptx"

WORK = ROOT / "paper" / "_template_work.pptx"
shutil.copy(TEMPLATE, WORK)
prs = Presentation(WORK)
slides = list(prs.slides)

BLUE = 0x1F3864
NAVY = 0x514D83
RED = 0xB6242E
INK = 0x26262E


def find_shape(slide, contains, idx=0):
    hits = [sh for sh in slide.shapes
            if sh.has_text_frame and contains in sh.text_frame.text]
    return hits[idx] if len(hits) > idx else None


def set_text(shape, text, size=None, bold=None, color=None):
    """First-run replace preserving formatting."""
    tf = shape.text_frame
    lines = text.split("\n")
    paras = tf.paragraphs
    for i, line in enumerate(lines):
        if i < len(paras):
            p = paras[i]
            if p.runs:
                p.runs[0].text = line
                for r in p.runs[1:]:
                    r.text = ""
            else:
                p.add_run().text = line
            for r in p.runs:
                if size:
                    r.font.size = Pt(size)
                if bold is not None:
                    r.font.bold = bold
                if color is not None:
                    r.font.color.rgb = __import__("pptx.dml.color",
                                                  fromlist=["RGBColor"]
                                                  ).RGBColor(color)
        else:
            break


def body_replace(slide, old_snippet, new_text, size=14):
    sh = find_shape(slide, old_snippet)
    assert sh is not None, old_snippet[:40]
    tf = sh.text_frame
    # wipe extra paragraphs beyond the first, then set
    for p in list(tf.paragraphs[1:]):
        p._p.getparent().remove(p._p)
    p0 = tf.paragraphs[0]
    if p0.runs:
        p0.runs[0].text = new_text
        for r in p0.runs[1:]:
            r.text = ""
    else:
        p0.add_run().text = new_text
    for r in p0.runs:
        r.font.size = Pt(size)
    return sh


def add_picture_safe(slide, path, x, y, w=None, h=None):
    for sh in list(slide.shapes):
        # remove non-logo pictures
        if sh.shape_type == 13 and not (sh.left > Inches(10.5)
                                        and sh.top < Inches(0.8)):
            sh._element.getparent().remove(sh._element)
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def add_textbox(slide, x, y, w, h, lines, size=14, bold_first=False,
                color=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, (txt, sz, bd) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = txt
        r.font.size = Pt(sz)
        r.font.bold = bd
        r.font.name = "微软雅黑"
        if color:
            from pptx.dml.color import RGBColor
            r.font.color.rgb = RGBColor(*color)
        p.space_after = Pt(4)
    return box


# =================================================== slide 1: cover
s = slides[0]
for sh in s.shapes:
    if sh.has_text_frame:
        t = sh.text_frame.text
        if "组会汇报" in t:
            set_text(sh, "预测权限控制（PAC）阶段答辩")
        elif "求是创新" in t:
            set_text(sh, "受约束残差强化学习修复控制器信任：OOD ↓28%，代价如实报告",
                    size=20)
        elif "日期" in t:
            set_text(sh, "日期：2026年9月29日")

# =================================================== slide 2: overview
s = slides[1]
body_replace(s, "一、工作概况",
             "一、工作概况：架构定型为「SMC + MPC 专家 + 冻结 Transformer 编码器"
             " + 零初始化残差头」，学习组件只输出有界权限修正 Δα∈[−0.1, +0.1]，"
             "推进器指令始终由经典专家产生。", size=14)
sh = find_shape(s, "二、方法与机制更新")
if sh:
    set_text(sh, "二、本轮更新：顶刊体例重构", size=16)
    sh.width = Inches(5.8)
sh = find_shape(s, "三、训练与评估规模更新")
if sh:
    set_text(sh, "三、评估规模（预注册、配对）", size=16)
    sh.width = Inches(5.8)
# add the two side texts under the 二/三 headers
add_textbox(s, 0.7, 3.0, 5.8, 3.4, [
    ("• 网络级架构图与训练图（冻结/可训练、参数量、零初始化）", 14, False),
    ("• 轨迹与姿态的运动层证据图（真实三维路径）", 14, False),
    ("• 控制质量图：残差策略 OOD 饱和率与控制代价全方法最低", 14, False),
    ("• 方法命名改描述式：Supervised PAC / Residual PAC", 14, False),
    ("• 全部世界模型内容移出主线（另行探索）", 14, False),
], size=14)
add_textbox(s, 6.8, 3.0, 5.8, 3.4, [
    ("• 120 个配对回合（85 ID + 35 OOD）", 14, False),
    ("• 14 个方法-种子组合 · 1,680 次滚动 · ≈353 万控制步", 14, False),
    ("• 模型种子级 t(4) 95% CI + 回合级自助法（10,000 次）", 14, False),
    ("• 注册准则先于训练冻结：残差相关六项 → 3 过 / 2 挂 / 1 未评", 14, False),
], size=14)

# =================================================== slide 3: problem+idea
s = slides[2]
for sh in list(s.shapes):
    if sh.shape_type == 13 and not (sh.left > Inches(10.5)
                                    and sh.top < Inches(0.8)):
        sh._element.getparent().remove(sh._element)
body_replace(s, "一、中间是核心控制结构",
             "问题：执行器退化下 MPC 按错误模型持续输出「自信的失效指令」，"
             "OOD 执行器族单独工作误差 0.822 m ≈ 正常 9 倍，逼近 1 m 任务失败线。"
             "仲裁问题（每一时刻信任谁、信多少）随海况连续变化，人工切换规则无法"
             "枚举。", size=14)
body_replace(s, "PAC 模块不直接输", "") if find_shape(s, "PAC 模块不直接输") else None
add_textbox(s, 0.7, 2.6, 12.0, 4.4, [
    ("核心思想：权限 α", 17, True),
    ("学习系统不碰推进器，只输出 α∈[0,1]；u = (1−α)·u_SMC + α·u_MPC。"
     "α=0 全听 SMC，α=1 全听 MPC。", 14, False),
    ("受约束残差：Δα = 0.1·tanh(·)，末层零初始化", 17, True),
    ("冻结监督策略（编码器 14,113 参数），仅训练 2,177 参数残差头；"
     "开机行为与已部署基线逐位一致（全回合单元测试）。最坏情形 = 基线本身。", 14, False),
    ("四层安全栈（按执行顺序）", 17, True),
    ("① 有界输出（|Δα|≤0.1）② clip/平滑/率限 ③ 专家失效强制回主控 "
     "④ 数值异常回退冻结监督预测", 14, False),
], size=14)

# =================================================== slide 4: Fig1 AI
s = slides[3]
# remove big body text box, insert AI architecture image
sh = find_shape(s, "一、模型以长度为 16")
if sh:
    sh._element.getparent().remove(sh._element)
add_picture_safe(s, FIGS / "fig1_system_architecture_ai.png",
                 0.55, 1.55, w=8.9)
add_textbox(s, 9.7, 1.9, 3.4, 5.0, [
    ("读图要点", 16, True),
    ("绿区=已验证固定组件；紫区=可审计学习组件；灰区=执行级", 13, False),
    ("编码器冻结（❄ 14,113 参数）；残差头可训练（▲ 2,177 参数、零初始化）", 13, False),
    ("α_t = clip[0,1](α̂_t + Δα_t)：学习只修正信任，不产生指令", 13, False),
    ("底部安全条带：有界 Δα → clip/平滑/率限 → 强制主控 → 数值回退", 13, False),
], size=13)

# =================================================== slide 5: Fig2 AI + training
s = slides[4]
# replace body section header + text
sh = find_shape(s, "四、数据表现更新")
if sh:
    set_text(sh, "网络架构与两阶段训练", size=16)
sh = find_shape(s, "上一版中，Transformer-PAC")
if sh:
    set_text(sh, "编码器 16×24 历史 → 嵌入 32 → 单层 4 头 Transformer → α̂；"
                 "残差头 32→64→GELU→1（零初始化）输出 Δα。"
                 "第一阶段模仿短视界 oracle（18,900 样本）；"
                 "第二阶段仅训练残差头（TD3 + 行为正则 + 2×10⁵ 热启动，"
                 "训练种子与评估种子严格分区）。", size=14)
    sh.height = Inches(0.8)
    sh.top = Inches(1.75)
sh = find_shape(s, "分场景结果显示")
if sh:
    sh._element.getparent().remove(sh._element)
add_picture_safe(s, FIGS / "fig_network_training_ai.png",
                 0.3, 3.0, w=7.9)
add_textbox(s, 8.6, 3.0, 4.4, 4.2, [
    ("训练稳定性（5 种子）", 15, True),
    ("奖励/评论家/策略损失三面板：探索噪声 + 轮换训练族日程下采集", 13, False),
    ("趋势反映回合组合变化，非贪心性能", 13, False),
    ("无种子发散；注册评估（图 6、表 1）为记录在案的性能", 13, False),
], size=13)

# =================================================== slide 6: trajectories + attitude
s = slides[5]
sh = find_shape(s, "三种水流扰动条件下的真实闭环跟踪结果")
if sh:
    set_text(sh, "运动层证据：航行器实际如何运动。左：三维路径/水平投影/垂向误差"
                 "（列=ID 正弦海流、OOD 执行器退化；检查前固定的保留种子）。"
                 "右：位置误差/艏向误差/俯仰角时序——退化下 MPC 发散随回合增长"
                 "（持续错误的内部模型），残差策略两工况都贴近参考。", size=14)
    sh.height = Inches(1.1)
add_picture_safe(s, FIGS / "fig2_closed_loop_trajectories.png",
                 0.3, 2.6, w=6.2)
add_picture_safe(s, FIGS / "fig3_error_attitude.png",
                 6.8, 2.6, w=6.2)

# =================================================== slide 7: main results
s = slides[6]
sh = find_shape(s, "用于解释 PAC 的性能来源")
if sh:
    set_text(sh, "主结果：OOD 跟踪 RMSE 0.161→0.116 m（−28%，CI [−0.074, −0.015]；"
                 "回合级自助法一致）。最尖锐的读法对我们不利：固定 α=0.5 达 0.120 m，"
                 "相差不足 3%——残差的价值集中在最需要自适应处：执行器退化族较固定"
                 "混合再降 28%。分布内代价 +0.4% / 艏向 +4.3%，如实报告。", size=14)
    sh.height = Inches(1.3)
add_picture_safe(s, FIGS / "fig4_closed_loop_results.png",
                 1.95, 2.95, w=9.4)

# =================================================== slide 8: actuator case
s = slides[7]
sh = find_shape(s, "展示 control smoothness")
if sh:
    set_text(sh, "机制：残差学到的不是新调度，而是「否决」监督策略的越界。"
                 "退化发生后数秒内监督策略把 α 推过 0.9 并振荡（31.2% 步数、"
                 "std 0.331）；残差策略 97.0% 的步数稳在 [0.4, 0.6]。"
                 "c 面板：全部 35 个 OOD 回合 × 5 种子的逐回合平均权限箱线图——"
                 "监督 0.82±0.14（P10–P90 0.62–0.93）对残差 0.54±0.03，"
                 "全部残差回合落在 [0.48, 0.62]。", size=14)
    sh.height = Inches(1.5)
add_picture_safe(s, FIGS / "fig5_actuator_case.png",
                 3.4, 3.15, w=6.1)

# =================================================== slide 9: control quality + cost
s = slides[8]
sh = find_shape(s, "新增指标体系")
if sh:
    set_text(sh, "控制质量与代价：增益不是靠执行器滥用换来的", size=16)
add_picture_safe(s, FIGS / "fig_control_quality.png",
                 0.3, 2.0, w=7.0)
add_picture_safe(s, FIGS / "fig6_cost_tradeoff.png",
                 0.3, 4.95, w=7.0)
add_textbox(s, 8.8, 2.1, 4.3, 4.8, [
    ("控制质量（OOD）", 15, True),
    ("饱和步占比 0.126：六方法最低", 13, False),
    ("控制代价 31.9：六方法最低", 13, False),
    ("配对代价效应 −1.78 [−2.31, −1.26]", 13, False),
    ("准则③「不以饱和/约束换增益」的实证", 13, False),
    ("代价（如实报告）", 15, True),
    ("ID 误差 +0.4%、艏向 +4.3%", 13, False),
    ("求解超时率 0.69→0.80（基线 0.69 对所有非学习方法相同；固定 α 不变；"
     "候选解释=轨迹工作点，假设非结论）", 13, False),
    ("部署保障不受影响：回退触发 0.3–0.5%，成功率持平", 13, False),
], size=13)

# =================================================== slide 10: verdict + future
s = slides[9]
# locate the big body box by position/size (12.0x3.8 at y=2.1), not by text
big = None
for sh in s.shapes:
    if sh.has_text_frame and sh.width > Inches(11) and sh.height > Inches(3):
        big = sh
        break
assert big is not None, "slide10 body box not found"
tf = big.text_frame
for pp in list(tf.paragraphs[1:]):
    pp._p.getparent().remove(pp._p)
p0 = tf.paragraphs[0]
for r in list(p0.runs):
    r._r.getparent().remove(r._r)
lines = [
    ("判定（残差相关六项注册准则）：3 过（不以艏向/饱和/约束换增益；消融可分离；"
     "五种子方向稳定）· 2 挂（整体非劣性 ID +0.4%；超时不恶化 +0.107）· "
     "1 未评估（部署推理时延＝硬件阶段第一门槛）", False),
    ("", False),
    ("注册的下一步：① 直接 RL 对照臂（端到端 TD3 输出推进器指令）　"
     "② α∈{0, 0.25, 0.5, 0.75, 1} 常数权限扫描（信任地形）　"
     "③ 解冻最后编码器块的残差第二轮　④ 隔离超时机制　"
     "⑤ 硬件在环：十次试验 / 次序随机化 / 电压窗 / 注入式执行器退化 / 逐推理时延",
     False),
    ("", False),
    ("结论：残差策略以可证明的安全结构（零初始化＝基线、有界修正、固定回退）"
     "买到 OOD 鲁棒性 −28% 与执行器退化免疫（0.822→0.090 m），每一项代价"
     "与训练前冻结的准则对照测量。", True),
]
for i, (txt, bd) in enumerate(lines):
    pp = p0 if i == 0 else tf.add_paragraph()
    r = pp.add_run()
    r.text = txt
    r.font.size = Pt(14)
    r.font.bold = bd
    r.font.name = "微软雅黑"

# =================================================== slide 11: closing — keep

prs.save(OUT)
print(f"saved {OUT} with {len(slides)} slides")
