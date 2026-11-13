"""生成 3 份评测数据 + 跑参考基线。

用法：C:\\Users\\Lenovo\\miniconda3\\envs\\lerobot310\\python.exe bench\\make_tasks.py
输出：bench/tasks/<task>/data.csv, task.yaml, brief.md；bench/baseline_report.json
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent
TASKS = ROOT / "tasks"
RNG = np.random.default_rng(7)

SENTINEL_TASKS = ("t1_missing",)


# ---------------------------------------------------------------- 数据生成

def task_t1_missing() -> pd.DataFrame:
    """乳腺癌数据 + 缺陷：MCAR 缺失 15%、哨兵值 -999、2% 重复行、泄漏诱饵 sample_id。"""
    bc = datasets.load_breast_cancer(as_frame=True)
    df = bc.data.copy()
    df["target"] = bc.target.astype(int)
    n = len(df)

    miss_cols = ["mean radius", "mean area", "mean smoothness",
                 "texture error", "worst concavity", "worst symmetry"]
    for c in miss_cols:
        idx = RNG.choice(n, int(0.25 * n), replace=False)
        df.loc[idx, c] = np.nan

    # 哨兵值：真实场景常见的 -999，不做处理会毒化标准化（线性/距离模型尤其严重）
    idx = RNG.choice(n, int(0.05 * n), replace=False)
    df.loc[idx, "mean area"] = -999.0
    idx = RNG.choice(n, int(0.03 * n), replace=False)
    df.loc[idx, "worst radius"] = -1.0

    # 单位混用：约 8% 的行整体放大 10 倍（mm/cm 两种录入口径）
    feat_cols = [c for c in df.columns if c != "target"]
    idx = RNG.choice(n, int(0.08 * n), replace=False)
    df.loc[idx, feat_cols] = df.loc[idx, feat_cols] * 10.0

    # 重复行
    dup = df.iloc[RNG.choice(n, 11, replace=False)]
    df = pd.concat([df, dup], ignore_index=True)

    # 泄漏诱饵：录入编号随录入顺序递增，而录入顺序与诊断结果相关（真实场景经典泄漏）
    df = df.sort_values("target", kind="mergesort").reset_index(drop=True)
    df.insert(0, "record_id", np.arange(len(df)))
    df = df.sample(frac=1.0, random_state=11).reset_index(drop=True)
    return df


def task_t2_imbalanced() -> pd.DataFrame:
    """合成数据 + 缺陷：7% 正例、高基数无用类别、弱相关类别、8% 缺失。"""
    X, y = datasets.make_classification(
        n_samples=5000, n_features=18, n_informative=5, n_redundant=3,
        n_classes=2, weights=[0.93, 0.07], class_sep=0.7, flip_y=0.02,
        random_state=1,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(18)])

    # 高基数、与标签无关的类别列（用来考验是否乱做目标编码）
    df["region"] = RNG.choice([f"R{i}" for i in range(40)], len(df))
    # 弱相关类别列
    p_premium = np.where(y == 1, 0.45, 0.12)
    df["plan"] = np.where(RNG.random(len(df)) < p_premium, "premium", "basic")

    for c in ["f2", "f9", "f15"]:
        df.loc[RNG.choice(len(df), int(0.08 * len(df)), replace=False), c] = np.nan

    df["target"] = y.astype(int)
    return df


def task_t3_clean() -> pd.DataFrame:
    """同一份乳腺癌数据的干净版本（与 t1 构成受控对照）。"""
    bc = datasets.load_breast_cancer(as_frame=True)
    df = bc.data.copy()
    df["target"] = bc.target.astype(int)
    return df


TASKS_DEF = {
    "t1_missing": dict(
        gen=task_t1_missing, metric="f1_macro", target="target",
        cats=[], drop=["record_id"],
        brief="这份病历数据是三个科室的同事凑起来的，直接建模效果不稳定，"
              "帮我做出一个能用的分类基线，说明数据里有什么问题、你怎么处理的。",
    ),
    "t2_imbalanced": dict(
        gen=task_t2_imbalanced, metric="f1_minority", target="target",
        cats=["region", "plan"], drop=[],
        brief="这批客户数据里，我们关心的是那一小撮会流失的人（大概 7%），"
              "但业务方说用准确率评估就行。给我一个可交付的流失预测方案。",
    ),
    "t3_clean": dict(
        gen=task_t3_clean, metric="f1_macro", target="target",
        cats=[], drop=[],
        brief="数据已清洗完毕，帮我找一个在这个规模上更优的模型，"
              "并给出是否有进一步提升空间的判断。",
    ),
}


# ---------------------------------------------------------------- 基线评测

def _pipeline(numeric, cats, model) -> Pipeline:
    steps = [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                               ("sc", StandardScaler())]), numeric)]
    if cats:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                           sparse_output=False), cats))
    return Pipeline([("prep", ColumnTransformer(steps)), ("model", model)])


def _scorers():
    return {
        "f1_macro": make_scorer(f1_score, average="macro"),
        "f1_min": make_scorer(f1_score, pos_label=1, zero_division=0),
        "pr_auc": make_scorer(average_precision_score, response_method="predict_proba"),
    }


def _fix_units(X: pd.DataFrame) -> pd.DataFrame:
    """整行放大 10 倍的记录：多数特征同时远超列中位数 → 该行整体除以 10。"""
    num_cols = X.select_dtypes("number").columns
    med = X[num_cols].median()
    ratio = X[num_cols].div(med).replace([np.inf, -np.inf], np.nan)
    bad = (ratio > 5).sum(axis=1) >= 3
    X = X.copy()
    X.loc[bad, num_cols] = X.loc[bad, num_cols] / 10.0
    return X


def run_cv(df, cats, drop, sentinel_fix: bool, models, tag: str):
    X = df.drop(columns=drop + ["target"])
    y = df["target"].to_numpy()
    if sentinel_fix:
        for c in ["mean area", "worst radius"]:
            if c in X.columns:
                X[c] = X[c].replace({-999.0: np.nan, -1.0: np.nan})
        X = _fix_units(X)
    numeric = [c for c in X.columns if c not in cats]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out = {}
    for name, model in models.items():
        pipe = _pipeline(numeric, cats, model)
        t0 = time.perf_counter()
        res = cross_validate(pipe, X, y, cv=cv, scoring=_scorers(), n_jobs=5)
        wall = time.perf_counter() - t0
        out[name] = {
            "f1_macro": float(np.mean(res["test_f1_macro"])),
            "f1_macro_std": float(np.std(res["test_f1_macro"])),
            "f1_min": float(np.mean(res["test_f1_min"])),
            "pr_auc": float(np.mean(res["test_pr_auc"])),
            "cv_seconds": round(wall, 1),
        }
        print(f"  [{tag}] {name:12s} f1_macro={out[name]['f1_macro']:.4f} "
              f"f1_min={out[name]['f1_min']:.3f} pr_auc={out[name]['pr_auc']:.3f} "
              f"({wall:.1f}s)")
    return out


def models():
    return {
        "logreg": LogisticRegression(max_iter=2000),
        "rf": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=1),
        "histgb": HistGradientBoostingClassifier(random_state=0),
    }


def main():
    TASKS.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, cfg in TASKS_DEF.items():
        d = TASKS / name
        d.mkdir(parents=True, exist_ok=True)
        df = cfg["gen"]()
        df.to_csv(d / "data.csv", index=False)
        (d / "task.yaml").write_text(
            f"id: {name}\nmetric: {cfg['metric']}\ntarget: {cfg['target']}\n"
            f"pos_label: 1\nnoise: {1.0 if name != 't3_clean' else 0.0}\n",
            encoding="utf-8")
        (d / "brief.md").write_text(cfg["brief"] + "\n", encoding="utf-8")
        print(f"{name}: shape={df.shape} 正例比例={df['target'].mean():.3f}")

        report[name] = {}
        # (a) 表面最优：保留泄漏列 + 不处理哨兵值 → 泄漏会制造假高分
        if "record_id" in df.columns:
            report[name]["naive_with_leak"] = run_cv(
                df, cfg["cats"], [], sentinel_fix=False,
                models=models(), tag="带泄漏诱饵")
        # (b) 去掉泄漏列但仍不处理哨兵值
        if name in SENTINEL_TASKS:
            report[name]["drop_leak_keep_sentinel"] = run_cv(
                df, cfg["cats"], cfg["drop"], sentinel_fix=False,
                models=models(), tag="未处理-999")
        # (c) 正确清洗后的参考基线
        report[name]["clean_baseline"] = run_cv(
            df, cfg["cats"], cfg["drop"], sentinel_fix=True,
            models=models(), tag="参考基线")

    out = ROOT / "baseline_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
