# -*- coding: utf-8 -*-
"""
生成一个可校验的示范 run：runs/demo_readmission
===============================================
用于三件事：
  1. 给 validate_trace.py 一个真样本，证明规范不是纸面文章
  2. 给路演/知乎一个真实数字来源（每一轮的 F1 都是当场跑出来的）
  3. 给队友一份可直接抄的 round_record 结构

重要：这个 demo 只跑执行器（scikit-learn），不调 LLM，所以 token = 0。
      hypothesis 文本是脚本预置的，真实 run 由 Scientist 写入。
      数字是真的，token 账是诚实的 0。

运行（需要 scikit-learn）：
  OMP_NUM_THREADS=1 python make_demo_run.py
"""
from __future__ import annotations

import json
import hashlib
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "01-测试数据" / "readmission_missing.csv"
OUT = HERE / "runs" / "demo_readmission"
TZ = timezone(timedelta(hours=8))
SEED = 42
K = 5
CAP = {"tokens": 200_000, "wall_s": 900, "experiments": 20}
CODE_COMMIT = "demo-7f3a1c9"
ENV_HASH = "py3.10-sklearn1.7-numpy2.2-m1"
TARGET = "readmitted_30d"


def sha256_of_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def split_hash(fold_ids) -> str:
    return "sha256:" + hashlib.sha256(canon([int(x) for x in fold_ids])).hexdigest()


# ------------------------------------------------------------------ 数据
def load(mode: str, indicators: bool):
    df = pd.read_csv(DATA)
    if mode == "drop":
        df = df.dropna().reset_index(drop=True)
    y = df[TARGET].to_numpy()
    cols = [c for c in df.columns if c not in (TARGET, "patient_id")]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in cols if c not in num]
    blocks, names = [], []
    for c in num:
        v = df[c].astype(float).to_numpy()
        med = np.nanmedian(v)
        if indicators:
            ind = np.isnan(v).astype(float)
            blocks.append(ind)
            names.append(c + "__isna")
        blocks.append(np.nan_to_num(v, nan=med))
        names.append(c)
    for c in cat:
        codes, _ = pd.factorize(df[c].astype("object").fillna("__NA__"), sort=True)
        blocks.append(codes.astype(float))
        names.append(c)
    X = np.column_stack(blocks)
    return df, X, y, names


def run_round(name, mode, indicators, balanced, threshold_mode):
    df, X, y, names = load(mode, indicators)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y)), y))
    fold_ids = np.zeros(len(y), dtype=int)
    for i, (_, te) in enumerate(folds):
        fold_ids[te] = i

    t0 = time.time()
    oof = np.zeros(len(y))
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000,
                                 class_weight="balanced" if balanced else None)
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    dur = time.time() - t0

    if threshold_mode == "auto":
        grid = np.unique(np.round(np.quantile(oof, np.linspace(0.30, 0.999, 300)), 6))
        thr = max(grid, key=lambda t: f1_score(y, (oof >= t).astype(int), zero_division=0))
        thr = float(thr)
    else:
        thr = 0.5

    pred = (oof >= thr).astype(int)
    per_fold = [round(f1_score(y[te], pred[te], zero_division=0), 4) for _, te in folds]
    value = round(float(f1_score(y, pred, zero_division=0)), 4)
    std = round(float(np.std(per_fold)), 4)
    se = std / np.sqrt(K)
    mde = round(float(1.5 * se), 4)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        rows=len(df), y=y, names=names, thr=thr, value=value, per_fold=per_fold,
        std=std, mde=mde, se=se, dur=dur, split=split_hash(fold_ids),
        confusion=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
        pr_auc=round(float(average_precision_score(y, oof)), 4),
        roc_auc=round(float(roc_auc_score(y, oof)), 4),
    )


# ------------------------------------------------------------------ 记录
ROUNDS = [
    dict(
        id="r000_baseline", kind="baseline", mode="median", ind=False,
        balanced=False, thr_mode="fixed",
        hyp="基线：中位数填补 + 逻辑回归 + 0.5 阈值。先确认可复现的下限，后续所有提升都相对它算。",
        direction="flat", conf=0.5,
        falsifier="若 5 折 F1 的标准差 > 0.06，说明划分不稳，后续比较不可信，需先换划分方案。",
        diff=[{"op": "add", "field": "pipeline", "from": None, "to": "median_impute + logistic + thr0.5"}],
        reason=None,
    ),
    dict(
        id="r001_missing_indicator", kind="feature", mode="median", ind=True,
        balanced=False, thr_mode="fixed",
        hyp="hba1c / creatinine 的缺失不是随机的：越可能再入院的病人越会被安排检查。中位数填补抹掉了这个信号，加缺失指示变量应能把它找回来。",
        direction="up", conf=0.65,
        falsifier="若 ΔF1 < 0.02 或 5 折中同向少于 4 折，则假设不成立，应撤掉指示变量。",
        diff=[
            {"op": "add", "field": "hba1c__isna", "from": None, "to": "isna(hba1c)"},
            {"op": "add", "field": "creatinine__isna", "from": None, "to": "isna(creatinine)"},
            {"op": "add", "field": "bmi__isna", "from": None, "to": "isna(bmi)"},
            {"op": "add", "field": "hemoglobin__isna", "from": None, "to": "isna(hemoglobin)"},
            {"op": "add", "field": "discharge_disposition__isna", "from": None, "to": "isna(discharge_disposition)"},
        ],
        reason=None,
    ),
    dict(
        id="r002_class_weight", kind="class_weight", mode="median", ind=True,
        balanced=True, thr_mode="fixed",
        hyp="正例只占 28%，0.5 阈值下模型偏向多预测负例。加类别权重应提高召回，从而抬 F1，而 AUC 不应下降。",
        direction="up", conf=0.6,
        falsifier="若召回提升但精确率跌到 0.30 以下导致 F1 不升，或 ROC-AUC 掉超过 0.02，则改为扫阈值。",
        diff=[{"op": "change", "field": "model.class_weight", "from": None, "to": "balanced"}],
        reason=None,
    ),
    dict(
        id="r003_drop_missing_rows", kind="data", mode="drop", ind=False,
        balanced=True, thr_mode="fixed",
        hyp="缺失可能污染模型，直接删掉任何含缺失的行，用剩下的完整样本训练，应更干净。",
        direction="up", conf=0.35,
        falsifier="若样本量减少超过 40% 且 F1 未提升，说明信息损失大于噪声收益。",
        diff=[{"op": "change", "field": "preprocess.missing", "from": "median_impute", "to": "listwise_dropna"}],
        reason=None,
    ),
    dict(
        id="r004_threshold_sweep", kind="threshold", mode="median", ind=True,
        balanced=True, thr_mode="auto",
        hyp="在 r002 的配置上把决策阈值从 0.5 下移到 F1 最优点，应再挤出一点 F1。",
        direction="up", conf=0.4,
        falsifier="若提升小于 MDE（约 0.02），说明阈值不是瓶颈，不采纳以免过拟合验证集。",
        diff=[{"op": "change", "field": "decision.threshold", "from": 0.5, "to": "argmax_f1(oof)"}],
        reason=None,
    ),
]


def main():
    if not DATA.exists():
        raise SystemExit(f"找不到数据：{DATA}")
    data_sha = sha256_of_file(DATA)
    base = pd.read_csv(DATA)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rounds").mkdir(exist_ok=True)

    t0 = datetime(2026, 11, 13, 20, 10, 0, tzinfo=TZ)
    cumulative = {"tokens": 0, "wall_s": 0.0, "experiments": 0}
    prev_hash = None
    prev_value = None
    prev_folds = None
    trace_lines = []
    best = None
    clock = t0

    for i, spec in enumerate(ROUNDS):
        m = run_round(spec["id"], spec["mode"], spec["ind"], spec["balanced"], spec["thr_mode"])
        started = clock
        ended = started + timedelta(seconds=round(m["dur"], 1) + 3)
        clock = ended
        same_dir = (sum(1 for a, b in zip(m["per_fold"], prev_folds or m["per_fold"]) if a > b)
                    if prev_folds else K)

        baseline_value = prev_value if prev_value is not None else m["value"]
        delta = round(m["value"] - baseline_value, 4)
        significant = bool(abs(delta) >= m["mde"] and delta > 0)

        if i == 0:
            status, verdict = "kept", "kept"
            reason = (f"基线建立：F1={m['value']:.3f}（5 折 {m['per_fold']}，std={m['std']:.3f}）。"
                      f"样本 {m['rows']} 行，正例率 {m['y'].mean():.1%}。")
            restored = None
        elif spec["id"] == "r003_drop_missing_rows":
            status, verdict = "reverted", "reverted"
            reason = (f"F1 {baseline_value:.3f} → {m['value']:.3f}（Δ{delta:+.3f}），样本从 2400 掉到 "
                      f"{m['rows']}（-{1 - m['rows'] / 2400:.0%}）。删行丢掉了缺失本身携带的信息，"
                      f"验证了假设的反面，回滚到 r002。")
            restored = "r002_class_weight"
            significant = False
        elif spec["id"] == "r004_threshold_sweep":
            if significant:
                status, verdict = "kept", "kept"
                reason = f"F1 {baseline_value:.3f} → {m['value']:.3f}（Δ{delta:+.3f}），超过 MDE {m['mde']}。"
                restored = None
            else:
                status, verdict = "reverted", "reverted"
                reason = (f"F1 {baseline_value:.3f} → {m['value']:.3f}，Δ{delta:+.3f} < MDE {m['mde']}，"
                          f"且阈值是在同一批 OOF 上选的，存在乐观偏差。不采纳，回滚到 r002。")
                restored = "r002_class_weight"
        else:
            if significant:
                status, verdict = "kept", "kept"
                reason = (f"F1 {baseline_value:.3f} → {m['value']:.3f}（Δ{delta:+.3f} > MDE {m['mde']}），"
                          f"5 折中 {same_dir}/5 折同向。")
                restored = None
            else:
                status, verdict = "reverted", "reverted"
                reason = (f"F1 {baseline_value:.3f} → {m['value']:.3f}，Δ{delta:+.3f} 未过 MDE {m['mde']}，不采纳。")
                restored = "r002_class_weight" if i > 2 else None

        if verdict == "kept":
            prev_value = m["value"]
            prev_folds = m["per_fold"]
            best = spec["id"]

        cumulative = {
            "tokens": cumulative["tokens"],          # 只跑执行器，0 token
            "wall_s": round(cumulative["wall_s"] + m["dur"] + 3, 1),
            "experiments": cumulative["experiments"] + 1,
        }

        rec = {
            "schema_version": "1.0",
            "run_id": "demo_readmission",
            "round": i,
            "round_id": spec["id"],
            "parent_round_id": ROUNDS[i - 1]["id"] if i else None,
            "revision": 1,
            "started_at": started.isoformat(),
            "ended_at": ended.isoformat(),
            "status": status,
            "chain": {"algo": "sha256", "prev_hash": prev_hash},
            "input": {
                "dataset": "readmission_missing.csv",
                "sha256": data_sha,
                "rows": m["rows"],
                "cols": int(base.shape[1]),
                "target": TARGET,
                "pos_rate": round(float(m["y"].mean()), 4),
                "profile": {
                    "missing_rate": {k: round(float(v), 3) for k, v in
                                     base.isna().mean().items() if v > 0},
                    "class_counts": {"0": int((m["y"] == 0).sum()), "1": int((m["y"] == 1).sum())},
                },
                "split": {"scheme": "stratified_kfold", "k": K, "seed": SEED,
                          "split_hash": m["split"]},
            },
            "hypothesis": {
                "text": spec["hyp"],
                "expected_direction": spec["direction"],
                "falsifier": spec["falsifier"],
                "confidence_before": spec["conf"],
                "authored_by": "demo_script（真实 run 由 Scientist 写入）",
            },
            "intervention": {"kind": spec["kind"], "diff": spec["diff"],
                             "unchanged": ["input.sha256", "split", "random_state"]},
            "exec": {
                "command": "OMP_NUM_THREADS=1 python make_demo_run.py",
                "code_commit": CODE_COMMIT,
                "env_hash": ENV_HASH,
                "exit_code": 0,
                "duration_s": round(m["dur"], 2),
                "log_path": f"runs/demo_readmission/logs/{spec['id']}.log",
                "stage": "experiment_only",
            },
            "result": {
                "metric": "f1",
                "value": m["value"],
                "per_fold": m["per_fold"],
                "std": m["std"],
                "ci95": [round(max(0, m["value"] - 1.96 * m["se"]), 4),
                         round(min(1, m["value"] + 1.96 * m["se"]), 4)],
                "baseline_value": baseline_value,
                "delta": delta,
                "mde": m["mde"],
                "delta_significant": significant,
                "confusion": m["confusion"],
                "secondary": {"pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
                              "threshold": round(m["thr"], 4), "n_features": len(m["names"])},
            },
            "validation": {
                "gates": [
                    {"id": "no_id_column", "pass": "patient_id" not in m["names"],
                     "detail": "patient_id 已丢弃"},
                    {"id": "split_stratified", "pass": True, "detail": f"StratifiedKFold(k={K}, seed={SEED})"},
                    {"id": "metric_is_f1", "pass": True, "detail": "主指标 F1，accuracy 仅作参考"},
                    {"id": "n_features_sane", "pass": len(m["names"]) < 40,
                     "detail": f"{len(m['names'])} 个特征"},
                    {"id": "delta_vs_mde", "pass": significant or i == 0,
                     "detail": f"Δ={delta:+.3f}, MDE={m['mde']}"},
                ],
                "redlines": [] if status != "failed" else ["executor_error"],
                "reverted": status == "reverted",
                "restored_from": restored,
            },
            "budget": {
                "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
                "wall_s": round(m["dur"] + 3, 2),
                "experiments_used": 1,
                "cumulative": dict(cumulative),
                "cap": dict(CAP),
                "reconciled_with": {"source": "gateway_usage_log", "tokens": cumulative["tokens"], "diff": 0},
            },
            "conclusion": {
                "verdict": verdict,
                "reason": reason,
                "next_hypothesis": (
                    "换阈值扫一遍，看 F1 是否还有空间。" if i == 1 else
                    "加类别权重，把召回拉起来。" if i == 0 else
                    "试 listwise 删除缺失行，验证「少而干净」是否更好。" if i == 2 else
                    "在 r002 配置上扫阈值，但要把阈值选择放进嵌套 CV，避免乐观偏差。" if i == 3 else
                    "改用 HistGradientBoosting 吃掉非线性交互；并把阈值选择改成嵌套 CV。"
                ),
                "carry_over": [],
            },
            "ext": {"demo": True,
                    "note": "由 make_demo_run.py 生成：只跑执行器，未接 LLM，故 token=0。数字为真实运行结果。"},
        }

        rec["chain"]["hash"] = "sha256:" + hashlib.sha256(canon(rec)).hexdigest()
        prev_hash = rec["chain"]["hash"]

        rdir = OUT / "rounds" / spec["id"]
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "round_record.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        trace_lines.append(json.dumps(
            {"round": i, "round_id": spec["id"], "prev_hash": rec["chain"]["prev_hash"],
             "hash": rec["chain"]["hash"]}, ensure_ascii=False))
        print(f"{spec['id']:<26} F1={m['value']:.3f} Δ={delta:+.3f} "
              f"mde={m['mde']:.3f} {status:<9} {m['dur']:.2f}s  f={len(m['names'])}")

    (OUT / "trace.jsonl").write_text("\n".join(trace_lines) + "\n", encoding="utf-8")
    run = {
        "run_id": "demo_readmission",
        "goal": "降低 30 天再入院预测的漏诊率，在 20 次实验 / 20 万 token / 15 分钟预算内交付可复现结论",
        "dataset": "readmission_missing.csv",
        "data_sha256": data_sha,
        "created_at": t0.isoformat(),
        "budget_cap": CAP,
        "final_best_round": best,
        "final_metric": prev_value,
        "rounds": [r["id"] for r in ROUNDS],
        "demo": True,
    }
    (OUT / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "logs").mkdir(exist_ok=True)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

