#!/usr/bin/env python3
"""reproduce_baseline.py - re-run the reference baselines on the 3 benchmark CSVs.

This mirrors bench/make_tasks.py exactly (same splits, same pipelines, same models)
so the numbers are comparable to bench/baseline_report.json:

    5-fold StratifiedKFold(shuffle=True, random_state=42)
    logreg  = LogisticRegression(max_iter=2000)
    rf      = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=1)
    histgb  = HistGradientBoostingClassifier(random_state=0)
    preprocessing = median impute + StandardScaler (+ OneHot for categoricals)
    variant (a) naive_with_leak   : keep the leak column, leave sentinels alone
    variant (b) drop_leak_keep_sentinel : drop leak column, sentinels still raw
    variant (c) clean_baseline    : drop leak column, -999/-1 -> NaN, undo x10 unit rows

Usage
    .venv\\Scripts\\python.exe tools\\reproduce_baseline.py
    .venv\\Scripts\\python.exe tools\\reproduce_baseline.py --tasks-dir incoming\\research_agent_bundle\\bench\\tasks
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT = Path(__file__).resolve().parent.parent

TASKS = {
    "t1_missing": {"cats": [], "drop": ["record_id"], "sentinel": True, "has_leak": True},
    "t2_imbalanced": {"cats": ["region", "plan"], "drop": [], "sentinel": True, "has_leak": False},
    "t3_clean": {"cats": [], "drop": [], "sentinel": True, "has_leak": False},
}


def pipeline(numeric, cats, model) -> Pipeline:
    steps = [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                               ("sc", StandardScaler())]), numeric)]
    if cats:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats))
    return Pipeline([("prep", ColumnTransformer(steps)), ("model", model)])


def scorers():
    return {
        "f1_macro": make_scorer(f1_score, average="macro"),
        "f1_min": make_scorer(f1_score, pos_label=1, zero_division=0),
        "pr_auc": make_scorer(average_precision_score, response_method="predict_proba"),
    }


def fix_units(X: pd.DataFrame) -> pd.DataFrame:
    num_cols = X.select_dtypes("number").columns
    med = X[num_cols].median()
    ratio = X[num_cols].div(med).replace([np.inf, -np.inf], np.nan)
    bad = (ratio > 5).sum(axis=1) >= 3
    X = X.copy()
    X.loc[bad, num_cols] = X.loc[bad, num_cols] / 10.0
    return X


def models() -> dict:
    return {
        "logreg": LogisticRegression(max_iter=2000),
        "rf": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=1),
        "histgb": HistGradientBoostingClassifier(random_state=0),
    }


def run_cv(df, cats, drop, sentinel_fix, tag, jobs):
    X = df.drop(columns=drop + ["target"])
    y = df["target"].to_numpy()
    if sentinel_fix:
        for c in ["mean area", "worst radius"]:
            if c in X.columns:
                X[c] = X[c].replace({-999.0: np.nan, -1.0: np.nan})
        X = fix_units(X)
    numeric = [c for c in X.columns if c not in cats]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out = {}
    for name, model in models().items():
        pipe = pipeline(numeric, cats, model)
        t0 = time.perf_counter()
        res = cross_validate(pipe, X, y, cv=cv, scoring=scorers(), n_jobs=jobs)
        wall = time.perf_counter() - t0
        out[name] = {
            "f1_macro": float(np.mean(res["test_f1_macro"])),
            "f1_macro_std": float(np.std(res["test_f1_macro"])),
            "f1_min": float(np.mean(res["test_f1_min"])),
            "pr_auc": float(np.mean(res["test_pr_auc"])),
            "cv_seconds": round(wall, 1),
        }
        o = out[name]
        print(f"    [{tag}] {name:7s} f1_macro={o['f1_macro']:.4f} "
              f"f1_min={o['f1_min']:.4f} pr_auc={o['pr_auc']:.4f}  ({wall:.1f}s)")
    return out


def main() -> int:
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default=str(PROJECT / "bench" / "tasks"))
    ap.add_argument("--report", default=str(PROJECT / "bench" / "baseline_report.json"))
    ap.add_argument("--out", default=str(PROJECT / "artifacts" / "baseline_reproduction.json"))
    ap.add_argument("--jobs", type=int, default=5)
    args = ap.parse_args()

    tasks_dir = Path(args.tasks_dir)
    report_path = Path(args.report)
    reference = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    print(f"tasks-dir : {tasks_dir}")
    print(f"reference : {report_path}" + ("" if reference else "  (missing - nothing to compare)"))
    print()

    produced = {}
    for name, cfg in TASKS.items():
        csv = tasks_dir / name / "data.csv"
        if not csv.exists():
            print(f"{name}: data.csv missing, skipped")
            continue
        df = pd.read_csv(csv)
        print(f"{name}  shape={df.shape}")
        produced[name] = {}

        if cfg["has_leak"]:
            produced[name]["naive_with_leak"] = run_cv(
                df, cfg["cats"], [], False, "带泄漏诱饵", args.jobs)
            produced[name]["drop_leak_keep_sentinel"] = run_cv(
                df, cfg["cats"], cfg["drop"], False, "未处理-999", args.jobs)
        produced[name]["clean_baseline"] = run_cv(
            df, cfg["cats"], cfg["drop"], cfg["sentinel"], "参考基线", args.jobs)
        print()

    # ---- compare
    print("=" * 96)
    print("比 对（本次实测 vs baseline_report.json，容差 0.002）")
    print("=" * 96)
    rows, mismatches = [], 0
    for task, variants in produced.items():
        for variant, per_model in variants.items():
            for model, metrics in per_model.items():
                ref = reference.get(task, {}).get(variant, {}).get(model, {})
                for key in ("f1_macro", "f1_min", "pr_auc"):
                    got = metrics[key]
                    want = ref.get(key)
                    if want is None:
                        continue
                    delta = got - want
                    ok = abs(delta) <= 0.002
                    if not ok:
                        mismatches += 1
                    rows.append((task, variant, model, key, want, got, delta, ok))
    for task, variant, model, key, want, got, delta, ok in rows:
        flag = "OK  " if ok else "DIFF"
        print(f"  {flag} {task:14s} {variant:22s} {model:7s} {key:9s} "
              f"ref={want:.4f} got={got:.4f} d={delta:+.4f}")
    print(f"\n不一致: {mismatches} / {len(rows)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "produced": produced,
        "reference": reference,
        "mismatch_count": mismatches,
        "tolerance": 0.002,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
