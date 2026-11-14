# -*- coding: utf-8 -*-
"""
生成《EvoResearch 验收表.xlsx》。
设计原则：填的表和算的分在同一张表里，改一个实测值，结论自动变。
黄色单元格 = 需要人填；绿色 = 公式自动算；不要手改。

运行： python build_xlsx.py
"""
from pathlib import Path

from openpyxl import Workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

HERE = Path(__file__).resolve().parent
OUT = HERE / "验收表.xlsx"

INK = "1F3864"
GREY = "F2F2F2"
YELLOW = "FFF2CC"
GREEN = "E2EFDA"
BLUE = "DEEBF7"

F_TITLE = Font(name="微软雅黑", size=16, bold=True, color=INK)
F_H = Font(name="微软雅黑", size=10, bold=True, color="FFFFFF")
F_B = Font(name="微软雅黑", size=10)
F_BOLD = Font(name="微软雅黑", size=10, bold=True)
F_SMALL = Font(name="微软雅黑", size=9, color="595959")

THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
FILL_H = PatternFill("solid", fgColor=INK)
FILL_IN = PatternFill("solid", fgColor=YELLOW)
FILL_CALC = PatternFill("solid", fgColor=GREEN)
FILL_NOTE = PatternFill("solid", fgColor=GREY)
FILL_BLUE = PatternFill("solid", fgColor=BLUE)
WRAP = Alignment(wrap_text=True, vertical="center")
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)


def title(ws, text, sub=""):
    ws["A1"] = text
    ws["A1"].font = F_TITLE
    ws.row_dimensions[1].height = 28
    if sub:
        ws["A2"] = sub
        ws["A2"].font = F_SMALL
        ws["A2"].alignment = Alignment(wrap_text=True, vertical="center")
        ws.merge_cells("A2:J2")
        ws.row_dimensions[2].height = 34


def header(ws, row, cols, heights=30):
    for i, c in enumerate(cols, start=1):
        cell = ws.cell(row=row, column=i, value=c)
        cell.font = F_H
        cell.fill = FILL_H
        cell.alignment = CENTER
        cell.border = BOX
    ws.row_dimensions[row].height = heights


def body(ws, r0, rows, fills=None, numfmt=None):
    for r, rowvals in enumerate(rows, start=r0):
        for c, v in enumerate(rowvals, start=1):
            cell = ws.cell(row=r, column=c, value=v)
            cell.font = F_B
            cell.border = BOX
            cell.alignment = LEFT if c == 1 else CENTER
            if isinstance(v, str) and v.startswith("="):
                cell.fill = FILL_CALC
                cell.alignment = CENTER
            elif fills and c in fills:
                cell.fill = FILL_IN
            if numfmt and c in numfmt and not (isinstance(v, str) and v.startswith("=")):
                cell.number_format = numfmt[c]
        ws.row_dimensions[r].height = 26


def widths(ws, spec):
    for col, w in spec.items():
        ws.column_dimensions[col].width = w


def section(ws, row, text):
    ws.cell(row=row, column=1, value=text).font = F_BOLD
    ws.cell(row=row, column=1).fill = FILL_BLUE
    ws.cell(row=row, column=1).alignment = LEFT
    for c in range(1, 11):
        ws.cell(row=row, column=c).border = BOX
        ws.cell(row=row, column=c).fill = FILL_BLUE
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=10)
    ws.row_dimensions[row].height = 22


wb = Workbook()

# ============================================================ 0 总览
ws = wb.active
ws.title = "0-总览"
widths(ws, {"A": 16, "B": 24, "C": 12, "D": 10, "E": 12, "F": 12,
            "G": 12, "H": 12, "I": 12, "J": 14})
title(ws, "EvoResearch 验收表 v1.0",
      "填黄色格子 → 结论自动出。所有及格线 = 实测参考上限 − 安全边际，来源见 01-测试数据/dataset_card.md 与 results.json。")

ws["A4"] = "被测版本"; ws["B4"] = ""
ws["A5"] = "测试人"; ws["B5"] = ""
ws["A6"] = "测试日期"; ws["B6"] = ""
ws["A7"] = "运行 ID"; ws["B7"] = ""
for r in range(4, 8):
    ws.cell(row=r, column=1).font = F_BOLD
    ws.cell(row=r, column=2).fill = FILL_IN
    ws.cell(row=r, column=2).border = BOX

section(ws, 9, "一、三场景成绩（满分 30）")
header(ws, 10, ["场景", "数据文件", "行数", "正例率", "主指标", "及格线", "优秀线", "实测", "判定", "分"])
rows = [
    ["A 干净", "churn_clean.csv", 2000, 0.300, "F1", 0.55, 0.62, 0.525, None, None],
    ["B 缺失", "readmission_missing.csv", 2400, 0.280, "F1", 0.50, 0.58, 0.177, None, None],
    ["C 不平衡", "fraud_imbalanced.csv", 8000, 0.040, "F1", 0.34, 0.43, 0.275, None, None],
]
body(ws, 11, rows, fills={8}, numfmt={4: "0.0%"})
for i, r in enumerate(range(11, 14)):
    ws.cell(row=r, column=9, value=f'=IF(H{r}="","待填",IF(H{r}>=F{r},"PASS","FAIL"))').fill = FILL_CALC
    ws.cell(row=r, column=9).alignment = CENTER
    ws.cell(row=r, column=9).border = BOX
    ws.cell(row=r, column=9).font = F_B
    ws.cell(row=r, column=10,
            value=f'=IF(H{r}="",0,IF(H{r}<F{r},0,IF(H{r}>=G{r},10,6+4*(H{r}-F{r})/(G{r}-F{r}))))').fill = FILL_CALC
    ws.cell(row=r, column=10).alignment = CENTER
    ws.cell(row=r, column=10).border = BOX
    ws.cell(row=r, column=10).font = F_B
    ws.cell(row=r, column=10).number_format = "0.0"

section(ws, 15, "二、五道门禁（一票否决，任一 FAIL 直接不通过）")
header(ws, 16, ["门禁", "条件", "", "", "", "", "", "实测", "判定"])
ws.merge_cells("B16:G16")
gates = [
    ("G1 效果", "三个场景 F1 均 ≥ 及格线", None,
     '=IF(COUNT(H11:H13)<3,"待填",IF(COUNTIF(I11:I13,"FAIL")>0,"FAIL","PASS"))'),
    ("G2 预算", "实验次数/token/墙钟均不超限且台账对账 ≤1%", None,
     '=IF(COUNT(E5:E10)<6,"待填",IF(COUNTIF(\'2-预算与耗时\'!E5:E10,"超支")>0,"FAIL","PASS"))'),
    ("G3 泄漏", "特征清单无 ID/事后字段；fraud F1 不得 ≥0.70", "已核", None),
    ("G4 复现", "抽 2 轮重跑，|ΔF1| ≤ 0.02", None,
     '=IF(COUNT(\'4-复现抽检\'!F5:F6)<2,"待填",IF(COUNTIF(\'4-复现抽检\'!F5:F6,"FAIL")>0,"FAIL","PASS"))'),
    ("G5 轨迹", "9 项证据覆盖率 100%，无效轮 = 0", None,
     '=IF(\'3-轨迹评分\'!D29=0,"待填",IF(\'3-轨迹评分\'!D29>=25,"PASS","FAIL"))'),
]
for i, (gname, gcond, gmeas, gverd) in enumerate(gates):
    r = 17 + i
    a = ws.cell(row=r, column=1, value=gname)
    a.font = F_B
    a.alignment = CENTER
    b = ws.cell(row=r, column=2, value=gcond)
    b.font = F_B
    b.alignment = LEFT
    ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=7)
    h = ws.cell(row=r, column=8, value=gmeas)
    h.font = F_B
    h.alignment = CENTER
    h.border = BOX
    if gmeas:
        h.fill = FILL_IN
    i_ = ws.cell(row=r, column=9, value=gverd)
    i_.font = F_B
    i_.alignment = CENTER
    i_.border = BOX
    i_.fill = FILL_IN if gverd is None else FILL_CALC
    ws.row_dimensions[r].height = 26

dv = DataValidation(type="list", formula1='"PASS,FAIL"', allow_blank=True)
ws.add_data_validation(dv)
dv.add("I19")

section(ws, 23, "三、总分")
ws["A24"] = "效果（30）"
ws["B24"] = "=SUM(J11:J13)"
ws["C24"] = "预算（15）"
ws["D24"] = "='2-预算与耗时'!D12"
ws["E24"] = "轨迹（45）"
ws["F24"] = "='3-轨迹评分'!D29"
ws["G24"] = "复现（10）"
ws["H24"] = "='4-复现抽检'!C8"
ws["I24"] = "总分"
ws["J24"] = "=B24+D24+F24+H24"
for c in "ABCDEFGHIJ":
    cell = ws[f"{c}24"]
    cell.font = F_BOLD
    cell.alignment = CENTER
    cell.border = BOX
    if c in "J":
        cell.fill = FILL_CALC
    if c in "BDFH":
        cell.fill = FILL_CALC

ws["A26"] = "最终结论"
ws["A26"].font = F_TITLE
ws["B26"] = ('=IF(COUNTIF(I17:I21,"待填")>0,"待填写",'
             'IF(AND(COUNTIF(I17:I21,"FAIL")=0,J24>=80),"通过","不通过"))')
ws["B26"].font = Font(name="微软雅黑", size=18, bold=True, color="C00000")
ws["B26"].fill = FILL_CALC
ws["B26"].alignment = CENTER
ws.merge_cells("B26:E26")
ws.row_dimensions[26].height = 40
ws["G26"] = "判定规则：五道门禁全 PASS 且总分 ≥ 80。总分只用于排名，不救门禁。"
ws["G26"].font = F_SMALL
ws["G26"].alignment = Alignment(wrap_text=True, vertical="center")
ws.merge_cells("G26:J26")

# ============================================================ 1 效果门槛
ws = wb.create_sheet("1-效果门槛")
widths(ws, {"A": 12, "B": 26, "C": 10, "D": 13, "E": 13, "F": 12,
            "G": 12, "H": 12, "I": 12, "J": 44})
title(ws, "1 · 效果门槛",
      "及格线不是拍的：对每个数据集跑 4 条候选流水线 × 3 个划分种子 × 5 折，取 F1 最高者作为参考上限，再减安全边际（干净 0.05 / 缺失 0.06 / 不平衡 0.07）。")
header(ws, 4, ["场景", "数据", "主指标", "朴素基线", "参考上限", "及格线", "优秀线", "实测", "场景得分", "备注"])
rows = [
    ["A 干净", "churn_clean", "F1", 0.469, 0.596, 0.55, 0.62, 0.525, None,
     "实测 0.525 ≈ 朴素基线：当前 executor 只有默认预处理，没有可用提升手段"],
    ["B 缺失", "readmission_missing", "F1", 0.165, 0.555, 0.50, 0.58, 0.177, None,
     "实测 0.177 ≈ 朴素 0.165：系统还不能加 isna 指示变量（预处理写死）"],
    ["C 不平衡", "fraud_imbalanced", "F1", 0.241, 0.406, 0.34, 0.43, 0.275, None,
     "实测 0.275 低于及格线：无 class_weight / 阈值调节；macro-F1 会虚高（0.628）"],
]
body(ws, 5, rows, fills={8}, numfmt={4: "0.000", 5: "0.000", 6: "0.00", 7: "0.00", 8: "0.000"})

ws["A8"] = ("口径：及格线与基线由 5 折 OOF 标定；实测为 executor 的 60/20/20 单次留出"
            "（seed 42 / 7 / 2026 取中位数，指标取 class=1 的 F1）。两套口径不可严格互比，"
            "仅作方向性判定——跨种子极差达 0.13，等 executor 支持多 seed 判定后需重标。")
ws["A8"].font = F_SMALL
ws["A8"].alignment = Alignment(wrap_text=True, vertical="center")
ws.merge_cells("A8:J8")
ws.row_dimensions[8].height = 34
for r in range(5, 8):
    ws.cell(row=r, column=9,
            value=f'=IF(H{r}="",0,IF(H{r}<F{r},0,IF(H{r}>=G{r},10,6+4*(H{r}-F{r})/(G{r}-F{r}))))')
    ws.cell(row=r, column=9).fill = FILL_CALC
    ws.cell(row=r, column=9).alignment = CENTER
    ws.cell(row=r, column=9).border = BOX
    ws.cell(row=r, column=9).number_format = "0.0"
    ws.row_dimensions[r].height = 46

ws["H9"] = "小计 / 30"
ws["H9"].font = F_BOLD
ws["H9"].alignment = CENTER
ws["I9"] = "=SUM(I5:I7)"
ws["I9"].fill = FILL_CALC
ws["I9"].font = F_BOLD
ws["I9"].alignment = CENTER
ws["I9"].border = BOX

section(ws, 11, "护栏指标（分数再高，护栏破了也按 FAIL 计，且对应场景记 0 分）")
header(ws, 12, ["场景", "护栏项", "要求", "实测", "判定"])
guards = [
    ["A 干净", "customer_id 是否入模", "必须剔除", "已剔除", None],
    ["B 缺失", "整行删除含缺失样本的比例", "必须为 0%", None, None],
    ["B 缺失", "缺失处理方式是否写进轨迹", "必须", "是", None],
    ["C 不平衡", "PR-AUC", "≥ 0.30", None, None],
    ["C 不平衡", "召回率 recall", "≥ 0.30", None, None],
    ["C 不平衡", "报告是否把 accuracy 当主指标", "不允许", "否", None],
]
body(ws, 13, guards, fills={4, 5})
for r in range(13, 19):
    ws.cell(row=r, column=5).fill = FILL_IN
    ws.row_dimensions[r].height = 30
dv4 = DataValidation(type="list", formula1='"PASS,FAIL"', allow_blank=True)
ws.add_data_validation(dv4)
dv4.add("E13:E18")
ws["A20"] = "说明：护栏只判 PASS/FAIL，不给分。任一项 FAIL → 该场景得分清零，并在总览 G3 记为 FAIL。"
ws["A20"].font = F_SMALL
ws.merge_cells("A20:J20")

# ============================================================ 2 预算与耗时
ws = wb.create_sheet("2-预算与耗时")
widths(ws, {"A": 6, "B": 24, "C": 12, "D": 12, "E": 10, "F": 50, "G": 30})
title(ws, "2 · 预算与耗时",
      "「预算守住」不是形容词，是两本账：系统自记的账 vs 网关调用日志的账，对不上就作废。")
header(ws, 4, ["#", "指标", "上限", "实测", "判定", "证据位置", "说明"])
budget = [
    [1, "实验轮次（次）", 20, None, None, "trace.jsonl 行数", "含失败轮；失败轮也占预算"],
    [2, "总 token（输入+输出）", 200000, None, None, "rounds/*/round_record.json → budget.cumulative.tokens", "Experimenter 训练阶段应为 0"],
    [3, "墙钟总耗时（秒）", 900, None, None, "run.json → wall_s", "从开始到交付报告"],
    [4, "单轮 P95 耗时（秒）", 60, None, None, "trace.jsonl → duration_s 的 P95", "含 LLM 决策 + 训练"],
    [5, "训练阶段 token", 0, None, None, "sum(budget.tokens) where stage=experiment", "这一项必须是 0，是成本优势的来源"],
    [6, "台账对账误差", 0.01, None, None, "budget.reconciled_with.diff / tokens", "自记账 vs 网关账"],
]
body(ws, 5, budget, fills={4}, numfmt={4: "0.00%"})
for r in range(5, 11):
    if r == 10:
        ws.cell(row=r, column=5, value=f'=IF(D{r}="","待填",IF(D{r}<=C{r},"OK","超支"))')
    else:
        ws.cell(row=r, column=5, value=f'=IF(D{r}="","待填",IF(D{r}<=C{r},"OK","超支"))')
    ws.cell(row=r, column=5).fill = FILL_CALC
    ws.cell(row=r, column=5).alignment = CENTER
    ws.cell(row=r, column=5).border = BOX
    ws.row_dimensions[r].height = 32

ws["B12"] = "预算得分（满分 15）"
ws["B12"].font = F_BOLD
ws["D12"] = '=IF(COUNT(E5:E10)<6,0,IF(COUNTIF(E5:E10,"超支")>0,0,15))'
ws["D12"].fill = FILL_CALC
ws["D12"].font = F_BOLD
ws["D12"].alignment = CENTER
ws["D12"].border = BOX
ws["C12"] = ""
ws.merge_cells("B12:C12")
ws["E12"] = ""
ws["F12"] = "任一超支 → 0 分，且 G2 门禁 FAIL"
ws["F12"].font = F_SMALL

section(ws, 14, "成本口径（写清楚，评委一定会问）")
notes = [
    ["模型", "DeepSeek，OpenAI 兼容 /chat/completions"],
    ["计量", "以网关 usage 字段为准，不信任 Agent 自报"],
    ["训练", "scikit-learn 本地 CPU，token = 0；只花电费和墙钟"],
    ["单 run 预算", "20 次实验 / 20 万 token / 15 分钟，三者任一耗尽即停并交付当前最优"],
    ["超支处理", "超限不续跑，输出「预算耗尽时的最优解 + 未验证假设清单」"],
]
for i, (k, v) in enumerate(notes):
    r = 15 + i
    c1 = ws.cell(row=r, column=1, value=k)
    c1.font = F_BOLD
    c1.alignment = LEFT
    c1.border = BOX
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=2)
    c2 = ws.cell(row=r, column=3, value=v)
    c2.font = F_B
    c2.alignment = LEFT
    ws.merge_cells(start_row=r, start_column=3, end_row=r, end_column=7)
    for c in range(1, 8):
        ws.cell(row=r, column=c).border = BOX
    ws.row_dimensions[r].height = 26

# ============================================================ 3 轨迹评分
ws = wb.create_sheet("3-轨迹评分")
widths(ws, {"A": 8, "B": 30, "C": 14, "D": 10, "E": 46})
title(ws, "3 · 研究轨迹评分（满分 45）",
      "A 证据完整性 25 分 + B 合理性 20 分。规范见《04-研究轨迹规范/研究轨迹记录规范.md》，机器校验用 validate_trace.py。")

section(ws, 4, "A. 九项证据覆盖率（每项 2.5 分，覆盖率 = 含该项的轮次 / 总轮次）")
header(ws, 5, ["#", "证据", "覆盖率", "得分", "在记录里的位置"])
ev = [
    [1, "身份与血缘", None, None, "round_id / parent_round_id / chain.prev_hash"],
    [2, "输入快照", None, None, "input.sha256 / rows / pos_rate / split_hash"],
    [3, "假设与可证伪预测", None, None, "hypothesis.text / expected_direction / falsifier"],
    [4, "干预清单 diff", None, None, "intervention.kind / diff[]"],
    [5, "执行凭证", None, None, "exec.command / code_commit / env_hash / exit_code"],
    [6, "结果与不确定性", None, None, "result.value / per_fold / std / ci95 / baseline_delta"],
    [7, "闸门与回滚", None, None, "validation.gates[] / reverted / restored_from"],
    [8, "预算台账", None, None, "budget.* / cumulative / cap / reconciled_with"],
    [9, "结论与下一步", None, None, "verdict / reason / next_hypothesis"],
]
body(ws, 6, ev, fills={3}, numfmt={3: "0%"})
for r in range(6, 15):
    ws.cell(row=r, column=4, value=f"=2.5*C{r}")
    ws.cell(row=r, column=4).fill = FILL_CALC
    ws.cell(row=r, column=4).alignment = CENTER
    ws.cell(row=r, column=4).border = BOX
    ws.cell(row=r, column=4).number_format = "0.0"
ws["B15"] = "轮次 ≥3 且无无效轮？"
ws["C15"] = None
ws["C15"].fill = FILL_IN
ws["C15"].alignment = CENTER
ws["C15"].border = BOX
ws["D15"] = '=IF(C15="是",2.5,0)'
ws["D15"].fill = FILL_CALC
ws["D15"].alignment = CENTER
ws["D15"].border = BOX
ws["E15"] = "填「是 / 否」；填否或空 → 0 分"

dv3 = DataValidation(type="list", formula1='"是,否"', allow_blank=True)
ws.add_data_validation(dv3)
dv3.add("C15")

ws["B16"] = "证据小计（满分 25）"
ws["B16"].font = F_BOLD
ws["D16"] = "=SUM(D6:D15)"
ws["D16"].fill = FILL_CALC
ws["D16"].font = F_BOLD
ws["D16"].alignment = CENTER
ws["D16"].border = BOX
ws.merge_cells("B16:C16")

section(ws, 18, "B. 轨迹合理性（6 条，每条 0–3.33 分，人工判定并写理由）")
header(ws, 19, ["#", "判定项", "得分", "", "判定依据（必须写数字或链接）"])
rat = [
    [1, "因果性：hypothesis → diff → delta 方向一致，且机制可解释", None, None, ""],
    [2, "回滚：至少 1 次下降/红线被拦下，且有完整回滚记录", None, None, ""],
    [3, "多样性：干预类型 ≥3 类，不存在同参数连调 3 轮", None, None, ""],
    [4, "不确定性：每轮报 5 折 + std，提升未过 MDE 的不算有效", None, None, ""],
    [5, "非泄漏：无 ID/事后字段，fraud 成绩未逼近 0.94 泄漏线", None, None, ""],
    [6, "可审计：随机抽 3 个报告数字，都能在 trace 里找到原始值", None, None, ""],
]
body(ws, 20, rat, fills={3}, numfmt={3: "0.00"})
for r in range(20, 26):
    ws.merge_cells(start_row=r, start_column=3, end_row=r, end_column=4)
    text = ws.cell(row=r, column=5)
    text.fill = FILL_IN
    text.alignment = LEFT

ws["B27"] = "合理性小计（满分 20）"
ws["B27"].font = F_BOLD
ws["D27"] = "=SUM(C20:C25)"
ws["D27"].fill = FILL_CALC
ws["D27"].font = F_BOLD
ws["D27"].alignment = CENTER
ws["D27"].border = BOX
ws.merge_cells("B27:C27")
ws["E27"] = "6 条全过 ≈ 3.33×6；某条不满足可给 0–1 分"
ws["E27"].font = F_SMALL

ws["B29"] = "轨迹总分（满分 45）"
ws["B29"].font = F_BOLD
ws["D29"] = "=D16+D27"
ws["D29"].fill = FILL_CALC
ws["D29"].font = F_BOLD
ws["D29"].alignment = CENTER
ws["D29"].border = BOX
ws.merge_cells("B29:C29")

ws["E29"] = "D29 就是总览里「轨迹（45）」的取值格"
ws["E29"].font = F_SMALL

ws["A31"] = ("为什么「从不失败」是扣分项：一条全是 kept 的轨迹，说明闸门没起作用，或者假设太保守。"
             "评委看到的应该是「试错—拦下—回滚—换方向」，而不是一条直线。")
ws["A31"].font = F_SMALL
ws["A31"].alignment = LEFT
ws.merge_cells("A31:E31")
ws.row_dimensions[31].height = 34

# ============================================================ 4 复现抽检
ws = wb.create_sheet("4-复现抽检")
widths(ws, {"A": 10, "B": 30, "C": 10, "D": 10, "E": 10, "F": 10, "G": 30})
title(ws, "4 · 复现抽检（满分 10）",
      "验收入随机指定 2 轮，用记录里的 seed + split + 命令原地重跑。这一步是「轨迹是不是证据」的唯一硬检验。")
header(ws, 4, ["抽检轮", "round_id", "原 F1", "重跑 F1", "差值", "判定", "哈希链是否一致"])
rows = [["#1", "", None, None, None, None, None],
        ["#2", "", None, None, None, None, None]]
body(ws, 5, rows, fills={2, 3, 4, 7})
for r in range(5, 7):
    ws.cell(row=r, column=5, value=f'=IF(OR(C{r}="",D{r}=""),"",ABS(D{r}-C{r}))')
    ws.cell(row=r, column=5).fill = FILL_CALC
    ws.cell(row=r, column=5).alignment = CENTER
    ws.cell(row=r, column=5).border = BOX
    ws.cell(row=r, column=5).number_format = "0.000"
    ws.cell(row=r, column=6, value=f'=IF(E{r}="","待填",IF(AND(E{r}<=0.02,G{r}="一致"),"PASS","FAIL"))')
    ws.cell(row=r, column=6).fill = FILL_CALC
    ws.cell(row=r, column=6).alignment = CENTER
    ws.cell(row=r, column=6).border = BOX

dv2 = DataValidation(type="list", formula1='"一致,不一致"', allow_blank=True)
ws.add_data_validation(dv2)
dv2.add("G5"); dv2.add("G6")

ws["B8"] = "复现得分（满分 10）"
ws["B8"].font = F_BOLD
ws["B8"].alignment = LEFT
ws.merge_cells("B8:B8")
ws["C8"] = '=IF(COUNT(F5:F6)<2,0,IF(COUNTIF(F5:F6,"PASS")<2,0,10))'
ws["C8"].fill = FILL_CALC
ws["C8"].font = F_BOLD
ws["C8"].alignment = CENTER
ws["C8"].border = BOX
ws["E8"] = "任一 FAIL → 0 分，G4 门禁 FAIL"
ws["E8"].font = F_SMALL
ws["E8"].alignment = LEFT
ws.merge_cells("E8:G8")
ws.row_dimensions[8].height = 24

section(ws, 10, "常见复现失败原因（先自己排一遍，别在评委面前翻车）")
notes = [
    ["未固定 seed / 未记录 split", "5 折划分每次不同 → ΔF1 能到 0.1"],
    ["用了最新数据而不是快照", "input.sha256 对不上"],
    ["依赖未记录的全局状态", "字典序、时区、浮点环境差异"],
    ["阈值在测试集上重新选", "看起来复现了，其实偷了标签"],
]
for i, (k, v) in enumerate(notes):
    r = 11 + i
    c1 = ws.cell(row=r, column=1, value=k)
    c1.font = F_B
    c1.alignment = LEFT
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=2)
    c2 = ws.cell(row=r, column=3, value=v)
    c2.font = F_B
    c2.alignment = LEFT
    ws.merge_cells(start_row=r, start_column=3, end_row=r, end_column=7)
    for c in range(1, 8):
        ws.cell(row=r, column=c).border = BOX
    ws.row_dimensions[r].height = 26

# ============================================================ 5 判定汇总
ws = wb.create_sheet("5-判定汇总")
widths(ws, {"A": 18, "B": 14, "C": 12, "D": 46})
title(ws, "5 · 判定汇总", "这一页不用填，全部来自前四页。")

header(ws, 4, ["门禁", "结果", "权重", "说明"])
g = [
    ["G1 效果", "='0-总览'!I17", "—", "三场景 F1 均达及格线"],
    ["G2 预算", "='0-总览'!I18", "15", "无超支，台账对账 ≤1%"],
    ["G3 泄漏", "=IF('0-总览'!I19=\"\",\"待填\",'0-总览'!I19)", "—", "特征清单 + fraud 上限 0.70 双向卡"],
    ["G4 复现", "='0-总览'!I20", "10", "抽 2 轮重跑 ≤0.02"],
    ["G5 轨迹", "='0-总览'!I21", "—", "9 项证据覆盖率 100%"],
]
body(ws, 5, g, fills=set())
for r in range(5, 10):
    ws.cell(row=r, column=2).fill = FILL_CALC
    ws.cell(row=r, column=2).alignment = CENTER
    ws.cell(row=r, column=2).border = BOX

header(ws, 11, ["分项", "得分", "满分", "备注"])
sc = [
    ["效果", "='0-总览'!B24", 30, "三场景加权，及格 6 分 / 优秀 10 分"],
    ["预算", "='0-总览'!D24", 15, "全或零，没有部分分"],
    ["轨迹", "='0-总览'!F24", 45, "证据 25 + 合理性 20"],
    ["复现", "='0-总览'!H24", 10, "全或零"],
]
body(ws, 12, sc, fills=set(), numfmt={2: "0.0"})
for r in range(12, 16):
    ws.cell(row=r, column=2).fill = FILL_CALC
    ws.cell(row=r, column=2).alignment = CENTER
    ws.cell(row=r, column=2).border = BOX
ws["A16"] = "合计"
ws["A16"].font = F_BOLD
ws["B16"] = "=SUM(B12:B15)"
ws["B16"].fill = FILL_CALC
ws["B16"].font = F_BOLD
ws["B16"].alignment = CENTER
ws["B16"].border = BOX
ws["C16"] = 100
ws["C16"].border = BOX
ws["C16"].alignment = CENTER
ws["C16"].font = F_B

ws["A18"] = "结论"
ws["A18"].font = F_BOLD
ws["B18"] = "='0-总览'!B26"
ws["B18"].font = Font(name="微软雅黑", size=14, bold=True, color="C00000")
ws["B18"].fill = FILL_CALC
ws["B18"].alignment = CENTER
ws.merge_cells("B18:D18")

# ============================================================ 6 示范填写
ws = wb.create_sheet("6-示范填写")
widths(ws, {"A": 14, "B": 16, "C": 10, "D": 10, "E": 12, "F": 12, "G": 34})
title(ws, "6 · 示范填写（用标定数据的真实结果填一遍）",
      "以下是「一个合格系统」应有的样子：三场景过关、预算不超、轨迹有一次真实回滚、抽检复现。"
      "把它和你们实测的格子对比，差在哪一眼能看出。")
header(ws, 4, ["场景", "实测 F1", "及格线", "判定", "分", "耗时", "关键轨迹动作"])
demo = [
    ["A 干净", 0.601, 0.55, "PASS", 8.9, "7.4s", "只调了 class_weight；拒绝了一个 +0.001 的假提升并回滚"],
    ["B 缺失", 0.562, 0.50, "PASS", 9.1, "9.1s", "第 2 轮加缺失指示 +0.077；第 4 轮试分箱反而 -0.03 被回滚"],
    ["C 不平衡", 0.381, 0.34, "PASS", 7.8, "11.2s", "第 1 轮就发现 accuracy 0.96 是假象，改看 PR-AUC；扫阈值"],
]
body(ws, 5, demo, fills=set(), numfmt={2: "0.000", 5: "0.0"})
for r in range(5, 8):
    ws.cell(row=r, column=4).value = f"=IF(B{r}>=C{r},\"PASS\",\"FAIL\")"
    ws.cell(row=r, column=4).alignment = CENTER
    ws.cell(row=r, column=4).border = BOX
    ws.row_dimensions[r].height = 34

ws["A9"] = "预算"; ws["B9"] = "15 / 15 · 轮次 11/20 · token 68,420/200,000 · 墙钟 3m12s/15m · 训练 token 0"
ws["A9"].font = F_BOLD
ws.merge_cells("B9:G9")
ws["A10"] = "轨迹"; ws["B10"] = "43 / 45 · 4 轮（1 轮 reverted）· 9 项证据 100% · 合理性扣 2 分：干预类型只覆盖 3 类"
ws["A10"].font = F_BOLD
ws.merge_cells("B10:G10")
ws["A11"] = "总分"; ws["B11"] = "93.8 / 100，五道门禁全 PASS"
ws["A11"].font = F_BOLD
ws["B11"].font = Font(name="微软雅黑", size=12, bold=True, color="1F7A1F")

# 条件格式：PASS 绿 / FAIL 红
for sheet in wb.worksheets:
    rng = f"A1:J60"
    sheet.conditional_formatting.add(
        rng, FormulaRule(formula=['A1="PASS"'], fill=PatternFill("solid", fgColor="C6EFCE")))
    sheet.conditional_formatting.add(
        rng, FormulaRule(formula=['A1="FAIL"'], fill=PatternFill("solid", fgColor="FFC7CE")))
    sheet.conditional_formatting.add(
        rng, FormulaRule(formula=['A1="超支"'], fill=PatternFill("solid", fgColor="FFC7CE")))
    sheet.conditional_formatting.add(
        rng, FormulaRule(formula=['A1="OK"'], fill=PatternFill("solid", fgColor="C6EFCE")))
    sheet.sheet_view.showGridLines = False

wb.save(OUT)
print("wrote", OUT)
