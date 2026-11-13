# EvoResearch · 测试 / 验收 / 路演 交付包

> 给二德子（测试、验收、路演）的一套东西：三份数据、一张验收表、一份路演稿、一套轨迹规范。
> 所有对外引用的数字，都能在这个包里找到出处。

---

## 目录

```
EvoResearch-交付/
├── 01-测试数据/
│   ├── churn_clean.csv           2000×13  干净数据
│   ├── readmission_missing.csv   2400×15  缺失数据（含 MNAR）
│   ├── fraud_imbalanced.csv      8000×13  不平衡数据（含泄漏字段）
│   ├── gen_data.py               生成器，固定种子，逐字节可复现
│   ├── baseline.py               标定及格线用（4 条流水线 × 3 种子 × 5 折）
│   ├── summarize.py              打印 results.json
│   ├── results.json              标定结果（参考上限 / 泄漏线的出处）
│   └── dataset_card.md           数据卡：字段、陷阱、正解、数字来源
├── 02-验收/
│   ├── 验收表.xlsx               6+1 个 sheet：总览、效果、预算、轨迹、复现、汇总、示范
│   ├── 验收说明.md               判定规则、及格线来源、争议裁决
│   └── build_xlsx.py             生成器（改数字后重跑即可）
├── 03-路演/
│   ├── 路演脚本.md               8 分钟逐字稿 + 5 分钟删减表 + 12 问 Q&A + 翻车预案
│   └── 知乎稿.md                 长文 + 短答 + 发布清单
└── 04-研究轨迹规范/
    ├── 研究轨迹记录规范.md        九项证据 + 三条硬约束 + 反例清单
    ├── round_record.schema.json  JSON Schema（draft-07）
    ├── validate_trace.py         零依赖校验器（结构 + 语义 + 哈希链）
    ├── make_demo_run.py          生成示范 run（真跑 sklearn）
    ├── example_round.json        一份真实记录
    └── runs/demo_readmission/    5 轮真实实验，含 2 次回滚，可直接校验
```

---

## 10 分钟上手

```powershell
# 1. 看数据长什么样、为什么难
notepad 01-测试数据\dataset_card.md

# 2. 验一遍轨迹（不需要装任何东西）
cd 04-研究轨迹规范
python validate_trace.py runs/demo_readmission --strict

# 3. 打开验收表，对照「6-示范填写」那一页
start 02-验收\验收表.xlsx
```

想复现全部数字：

```powershell
cd 01-测试数据
python gen_data.py
$env:OMP_NUM_THREADS='1'
C:\Users\Lenovo\miniconda3\envs\lerobot310\python.exe baseline.py
python summarize.py
```

> 为什么用这个 python：本机 `lerobot310` 环境里有 scikit-learn 1.7.2。
> `OMP_NUM_THREADS=1` 必须设——沙箱里 joblib 开线程池会去建命名管道，被拒绝（`PermissionError`）。

---

## 四份东西分别解决什么

| 交付物 | 解决的问题 | 一句话 |
|---|---|---|
| 三份 CSV | 系统在不同"病灶"下靠不靠谱 | 干净考不乱改、缺失考机制理解、不平衡考指标选择 |
| 验收表 | 怎么算过关 | 五道门禁一票否决 + 100 分排名，分数不救门禁 |
| 路演脚本 | 8 分钟怎么讲 | 用"两次失败"证明系统可靠，而不是用 F1 曲线 |
| 轨迹规范 | 每一轮留什么证据 | 九项证据 + 哈希链，第三方只读记录就能复现 |

---

## 三个可以直接拿去讲的事实

1. **同一个人，两个数字**：欺诈数据上 accuracy 0.962，F1 0.241，召回 0.150。全判"正常"也有 0.960。
2. **缺失本身是信号**：加 5 个 `isna` 指示变量，F1 0.165 → 0.443，5 折全部同向。
3. **拒绝比提升更难**：扫阈值 +0.002 < MDE 0.020，系统自己判为噪声并回滚。

---

## 数字的唯一来源

对外说的每个数，必须能在下面三处之一找到：

| 数字 | 出处 |
|---|---|
| 参考上限 0.596 / 0.555 / 0.406 | `01-测试数据/results.json` → `reference_f1` |
| 及格线 0.55 / 0.50 / 0.34 | 同上 → `pass_line`；规则见 `02-验收/验收说明.md` |
| 泄漏线 0.943 | `results.json` → `fraud_imbalanced.csv.leaky.f1` |
| 0.165 → 0.443 → 0.554 → 0.343 → 0.555 | `04-研究轨迹规范/runs/demo_readmission/rounds/*/round_record.json` |
| 五轮 / 零 token / 六秒 | 同上 → `budget.cumulative` |

改了任何数据或代码，**先重跑，再改稿**。路演稿里的数字和 `results.json` 对不上，比讲错一个概念更致命。

---

## 边界（别对外吹过）

- 数据是合成的。好处是可控、可复现；坏处是真实世界的脏法比这复杂。**被问到就直说。**
- demo run 只跑执行器，未接 LLM，因此 token = 0；真实链路的 LLM 记账见 `round_record.json` 的 `budget` 字段设计。
- F1 0.55（再入院）不是天花板，是"不许泄漏、不许挑测试集"约束下的真数。追求 0.9 很容易，代价是作弊。
- 验收表是模板 + 示范，实测列需要被测系统跑完后填。

---

## 下一步（建议顺序）

1. Research Agent 本体（Manager / Analyst / Scientist / Reviewer 编排）——现在是空壳。
2. 把 `round_record.json` 的写入接到 Manager / Experimenter 上，跑出**第一份真实轨迹**。
3. 用 `validate_trace.py` 卡住它，直到 5 轮全绿。
4. 再回来改路演稿：把 demo 的数字换成真实链路的数字。
