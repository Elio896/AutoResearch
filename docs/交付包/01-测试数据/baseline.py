# -*- coding: utf-8 -*-
"""
基线标定脚本 v2：为验收表提供「真实数字」，而不是拍脑袋的阈值。
================================================================
每个数据集跑 4 条候选流水线（模型动物园），全部用
  3 个划分种子 × 5 折分层 CV，OOF 选阈值（在折外预测上扫一遍 F1）。
最后取 F1 最高的一条作为「参考上限」，验收及格线 = 参考上限 − 安全边际。

  lr        逻辑回归，0.5 阈值          —— 大多数队伍的第一版
  lr_bal    逻辑回归 + 类别权重 + 调阈值
  gbdt      梯度提升，0.5 阈值
  gbdt_bal  梯度提升 + 类别权重 + 调阈值

另外单独跑一条 leaky：把 ID / 事后字段也塞进去，得到「作弊线」，
用来验收「泄漏检测」——分数接近作弊线 = 该轮作废。

运行环境：需要 scikit-learn。若机器上只有 conda 环境，例如
  C:\\Users\\Lenovo\\miniconda3\\envs\\lerobot310\\python.exe
并需设 OMP_NUM_THREADS=1（沙箱里 joblib 开线程池会撞命名管道限制）。
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "01-测试数据"
SEEDS = (42, 7, 2026)
N_SPLITS = 5
MARGIN = {                       # 及格线 = 参考上限 - margin（按指标波动定）
    "churn_clean.csv": 0.05,
    "readmission_missing.csv": 0.06,
    "fraud_imbalanced.csv": 0.07,
}

SPECS = {
    "churn_clean.csv": dict(target="churn", drop=["customer_id"], leaky=[]),
    "readmission_missing.csv": dict(target="readmitted_30d", drop=["patient_id"], leaky=[]),
    "fraud_imbalanced.csv": dict(target="is_fraud", drop=["txn_id"], leaky=["chargeback_filed"]),
}
TARGET = None  # 由主循环设置


def build_matrix(df, drop, leaky, use_leaky=False, use_indicators=True):
    y = df[TARGET].to_numpy()
    cols = [c for c in df.columns if c not in drop + [TARGET]]
    if not use_leaky:
        cols = [c for c in cols if c not in leaky]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in cols if c not in num]
    blocks, names = [], []
    for c in num:
        v = df[c].astype(float).to_numpy()
        med = np.nanmedian(v)
        ind = np.isnan(v).astype(float)
        blocks.append(np.nan_to_num(v, nan=med))
        names.append(c)
        if use_indicators and ind.mean() > 0:      # 缺失指示变量
            blocks.append(ind)
            names.append(c + "__isna")
    for c in cat:
        codes, _ = pd.factorize(df[c].astype("object").fillna("__NA__"), sort=True)
        blocks.append(codes.astype(float))
        names.append(c)
    X = np.column_stack(blocks) if blocks else np.zeros((len(df), 0))
    return X, y, names


def make_model(name):
    if name == "lr":
        return LogisticRegression(max_iter=2000)
    if name == "lr_bal":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    if name == "gbdt":
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=40,
            l2_regularization=2.0, early_stopping=False, random_state=42)
    if name == "gbdt_bal":
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=40,
            l2_regularization=2.0, early_stopping=False,
            class_weight="balanced", random_state=42)
    raise ValueError(name)


def oof_predict(name, X, y, folds):
    oof = np.zeros(len(y))
    t0 = time.time()
    for tr, te in folds:
        if name.startswith("lr"):
            sc = StandardScaler().fit(X[tr])
            clf = make_model(name).fit(sc.transform(X[tr]), y[tr])
            oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        else:
            clf = make_model(name).fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof, time.time() - t0


def pick_threshold(y, score):
    grid = np.unique(np.round(np.quantile(score, np.linspace(0.30, 0.999, 300)), 6))
    best, best_t = -1.0, 0.5
    for t in grid:
        f = f1_score(y, (score >= t).astype(int), zero_division=0)
        if f > best:
            best, best_t = f, float(t)
    return best_t


def measure(y, score, thr):
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        f1=round(f1_score(y, pred, zero_division=0), 4),
        precision=round(precision_score(y, pred, zero_division=0), 4),
        recall=round(recall_score(y, pred, zero_division=0), 4),
        pr_auc=round(average_precision_score(y, score), 4),
        roc_auc=round(roc_auc_score(y, score), 4),
        accuracy=round((tp + tn) / len(y), 4),
        threshold=round(float(thr), 4),
        confusion=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
    )


def run_candidate(name, X, y, seed):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(len(y)), y))
    oof, sec = oof_predict(name, X, y, folds)
    thr = 0.5 if name in ("lr", "gbdt") else pick_threshold(y, oof)
    m = measure(y, oof, thr)
    m["duration_s"] = round(sec, 2)
    m["per_fold_f1"] = [
        round(f1_score(y[te], (oof[te] >= thr).astype(int), zero_division=0), 4)
        for _, te in folds]
    m["fold_std"] = round(float(np.std(m["per_fold_f1"])), 4)
    return m


def main():
    global TARGET
    out = {}
    for fname, spec in SPECS.items():
        TARGET = spec["target"]
        df = pd.read_csv(DATA / fname)
        y = df[TARGET].to_numpy()
        X, y2, names = build_matrix(df, spec["drop"], spec["leaky"])

        rec = dict(file=fname, rows=int(len(df)), n_features=len(names),
                   positives=int(y.sum()), pos_rate=round(float(y.mean()), 4),
                   seeds=list(SEEDS), candidates={})
        print(f"\n=== {fname}  n={len(df)}  pos={y.sum()} ({y.mean():.1%})  f={len(names)} ===")

        for cand in ("lr", "lr_bal", "gbdt", "gbdt_bal"):
            runs = [run_candidate(cand, X, y, s) for s in SEEDS]
            agg = {}
            for k in ("f1", "pr_auc", "roc_auc", "precision", "recall", "accuracy"):
                vals = [r[k] for r in runs]
                agg[k] = round(float(np.median(vals)), 4)
                agg[k + "_min"] = round(float(np.min(vals)), 4)
                agg[k + "_max"] = round(float(np.max(vals)), 4)
            agg["threshold"] = round(float(np.median([r["threshold"] for r in runs])), 4)
            agg["fold_std"] = round(float(np.median([r["fold_std"] for r in runs])), 4)
            agg["duration_s"] = round(float(np.median([r["duration_s"] for r in runs])), 2)
            agg["confusion"] = runs[0]["confusion"]
            rec["candidates"][cand] = agg
            print(f"  {cand:9s} F1={agg['f1']:.3f} [{agg['f1_min']:.3f},{agg['f1_max']:.3f}] "
                  f"P={agg['precision']:.3f} R={agg['recall']:.3f} "
                  f"PR-AUC={agg['pr_auc']:.3f} ROC={agg['roc_auc']:.3f} "
                  f"Acc={agg['accuracy']:.3f} thr={agg['threshold']} {agg['duration_s']}s")

        best = max(rec["candidates"].items(), key=lambda kv: kv[1]["f1"])
        rec["best"] = best[0]
        rec["reference_f1"] = best[1]["f1"]
        rec["pass_line"] = round(rec["reference_f1"] - MARGIN[fname], 3)
        rec["excellent_line"] = round(min(0.99, rec["reference_f1"] + 0.02), 3)
        print(f"  -> best={best[0]}  ref_F1={rec['reference_f1']:.3f}  "
              f"pass>={rec['pass_line']:.3f}  excellent>={rec['excellent_line']:.3f}")

        # 缺失数据的对照组：去掉缺失指示变量，量化「加指示变量」这一干预的效果
        if df.isna().any().any():
            X0, y0, _ = build_matrix(df, spec["drop"], spec["leaky"], use_indicators=False)
            runs0 = [run_candidate("lr_bal", X0, y0, s) for s in SEEDS]
            f1_0 = round(float(np.median([r["f1"] for r in runs0])), 4)
            rec["no_indicator"] = dict(f1=f1_0, note="逻辑回归+类别权重+调阈值，但只用中位数填补，不加指示变量")
            rec["indicator_gain"] = round(rec["candidates"]["lr_bal"]["f1"] - f1_0, 3)
            print(f"  -> 无指示变量 lr_bal F1={f1_0:.3f}  加指示变量增益="
                  f"{rec['indicator_gain']:+.3f}")

        # 作弊线：把 ID / 事后字段放回去
        Xl, yl, names_l = build_matrix(df, [], spec["leaky"], use_leaky=True)
        lruns = [run_candidate("lr", Xl, yl, s) for s in SEEDS]
        rec["leaky"] = dict(
            f1=round(float(np.median([r["f1"] for r in lruns])), 4),
            pr_auc=round(float(np.median([r["pr_auc"] for r in lruns])), 4),
            n_features=len(names_l),
            note="把 ID / 事后字段放回去后的分数。正式成绩接近这条线 = 判泄漏。")
        print(f"  -> leaky F1={rec['leaky']['f1']:.3f} (f={len(names_l)})")
        out[fname] = rec

    (HERE / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
