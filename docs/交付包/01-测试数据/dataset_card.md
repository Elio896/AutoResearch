# 测试数据卡（dataset card）

三份 CSV，三种「病灶」，对应系统必须自己发现的三个研究问题。
全部由 `gen_data.py` 生成，固定种子 `20261113`，**重跑得到逐字节一致的文件**。

| 文件 | 行 × 列 | 正例率 | 缺失 | 考什么 |
|---|---|---|---|---|
| `churn_clean.csv` | 2000 × 13 | 30.0% | 无 | 没有明显病灶时，能不能**不乱改**（防自进化退化） |
| `readmission_missing.csv` | 2400 × 15 | 28.0% | 5 列，最高 19.3% | 能不能发现**缺失本身携带信息**（MNAR） |
| `fraud_imbalanced.csv` | 8000 × 13 | 4.0% | 无 | 能不能**不用 accuracy**，改用 PR-AUC / F1 |

数据哈希（进轨迹的 `input.sha256` 就用这个）：

```
churn_clean.csv          sha256:7b5a9b82e59b3fb…
readmission_missing.csv  sha256:fc0d237dcf8a0d9…
fraud_imbalanced.csv     sha256:e8d3ffafc0bff92…
```

完整值跑 `python -c "import hashlib;print(hashlib.sha256(open('x.csv','rb').read()).hexdigest())"`。

---

## A. `churn_clean.csv` —— 客户流失（干净）

**字段**：`customer_id`（ID，必须丢弃）、`tenure_months`、`contract_type`（3 类）、`internet_service`（3 类）、`payment_method`（4 类）、`monthly_charges`、`total_charges`、`num_support_tickets`、`avg_monthly_gb`、`is_senior`、`has_partner`、`autopay_enabled`、`churn`（目标）。

**生成机制**：logistic，主效应为月付合同（+1.5）、光纤（+0.7）、入网时长（−0.045/月）、工单数（+0.38）、自动扣款（−0.55），外加"新客户 + 月付"的非线性加成。

**这题的正解**：把 `customer_id` 丢掉，做基本编码，逻辑回归就有 F1 ≈ 0.47；加类别权重 ≈ 0.58。
**提升空间很小（参考上限 0.596）**——这正是考点：好系统在这里应该**停手**，而不是硬凑 5 轮调参。

**陷阱**：
- `customer_id` 是唯一值 ID。放进模型不会立刻爆分（树模型会忽略），但特征清单里出现它就是扣分项。
- `total_charges ≈ tenure × monthly`，与两个原特征强相关，共线。加上它不掉分，但也没用——好的轨迹应该记录"试过、无显著变化、撤掉"。

---

## B. `readmission_missing.csv` —— 30 天再入院（缺失）

**字段**：`patient_id`（ID）、`age`、`gender`、`bmi`、`systolic_bp`、`hba1c`、`creatinine`、`hemoglobin`、`num_medications`、`prior_admissions`、`length_of_stay`、`discharge_disposition`（4 类）、`insurance_type`（4 类）、`smoker`、`readmitted_30d`（目标）。

**缺失情况**：

| 列 | 缺失率 | 机制 |
|---|---|---|
| `bmi` | 11.4% | MCAR（纯随机） |
| `hemoglobin` | 14.4% | MCAR |
| `hba1c` | 19.3% | **MNAR**：越可能再入院，越会被安排查糖化血红蛋白 |
| `creatinine` | 12.4% | **MNAR**：同上，强度略弱 |
| `discharge_disposition` | 6.7% | MAR：SNF 转出的病例记录更全 |

52% 的行至少有一个缺失。

**这题的正解**：加缺失指示变量（`isna`）。实测 **F1 0.165 → 0.443（Δ+0.277）**。
**常见错解**：整行删除 → 样本掉到 48%，F1 只有 0.343，比基线高但远低于正解——这个"看起来还行其实是错"的坑是故意留的。

**陷阱**：
- 中位数填补会把"没查这项检查"当成"指标正常"，直接抹掉信号。
- `patient_id` 是 ID，必须丢。
- 缺失率 19% 的列如果用"删列"处理，恰好删掉了最强的信号列。

---

## C. `fraud_imbalanced.csv` —— 信用卡欺诈（不平衡）

**字段**：`txn_id`（ID）、`amount`、`hour_of_day`、`merchant_category`（8 类）、`card_age_days`、`txn_count_24h`、`avg_amount_7d`、`is_foreign`、`device_risk_score`、`distance_from_home_km`、`cvv_retry_count`、`chargeback_filed`（**事后字段，泄漏陷阱**）、`is_fraud`（目标）。

**生成机制**：logistic + 交互项（`device_risk × is_foreign`、`txn_24h ≥4 × 大额`、金额两端非单调），所以线性模型吃不全，树的模型有优势。

**这题的正解**：识别 accuracy 是假象（0.962，但全预测"正常"也能到 0.96），主指标换 F1 / PR-AUC，加类别权重或扫阈值。
参考上限 **F1 0.406**（PR-AUC 0.354，是随机基线 0.04 的 8.8 倍）。

**陷阱（两个，都致命）**：
1. **`chargeback_filed` 是事后字段**——只有欺诈被确认后才立案。放进去 F1 直接 0.943。这是验收表里"泄漏线"的由来：正式成绩一旦逼近 0.94，判为泄漏，该轮作废。
2. **accuracy**。报告里把 0.962 当主指标，等于宣称"我们检测出了欺诈"，其实只抓回 15%（朴素模型 F1 0.241、召回 0.150）。

---

## 怎么用

```bash
python gen_data.py                 # 重新生成（结果逐字节一致）
python baseline.py                 # 重跑标定，产出 results.json
python summarize.py                # 打印易读摘要
```

`baseline.py` 需要 scikit-learn。本机已有一个可用环境：

```powershell
$env:OMP_NUM_THREADS='1'
C:\Users\Lenovo\miniconda3\envs\lerobot310\python.exe baseline.py
```

> `OMP_NUM_THREADS=1` 是必需的：沙箱里 joblib 开线程池会去建命名管道，被拒绝后报 `PermissionError`。

**及格线怎么来的**：4 条候选流水线（LR / LR+权重 / GBDT / GBDT+权重）× 3 个划分种子 × 5 折，取 F1 最高者作参考上限，再减安全边际：

| 数据集 | 参考上限 | 安全边际 | 及格线 | 优秀线 |
|---|---|---|---|---|
| churn_clean | 0.596 | 0.05 | **0.55** | 0.62 |
| readmission_missing | 0.555 | 0.06 | **0.50** | 0.58 |
| fraud_imbalanced | 0.406 | 0.07 | **0.34** | 0.43 |

安全边际按各数据集 5 折 F1 的标准差（0.02–0.04）取 1.5–2 倍，保证**换一个划分种子不会把及格线翻掉**。
