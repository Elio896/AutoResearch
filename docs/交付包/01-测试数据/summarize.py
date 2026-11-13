# -*- coding: utf-8 -*-
"""把 results.json 打印成易读摘要（也可直接抄进验收表）。"""
import json
from pathlib import Path

d = json.loads((Path(__file__).parent / "results.json").read_text(encoding="utf-8"))
for f, r in d.items():
    print(f"\n{f}  n={r['rows']} pos={r['positives']} ({r['pos_rate']:.1%})")
    print(f"  best={r['best']}  ref_F1={r['reference_f1']:.3f}  "
          f"pass>={r['pass_line']:.3f}  excellent>={r['excellent_line']:.3f}")
    for k, v in r["candidates"].items():
        print("   {:9s} F1={:.3f} PR-AUC={:.3f} ROC={:.3f} Acc={:.3f} "
              "thr={:.3f} foldstd={:.3f} {}s".format(
                  k, v["f1"], v["pr_auc"], v["roc_auc"], v["accuracy"],
                  v["threshold"], v["fold_std"], v["duration_s"]))
    if "indicator_gain" in r:
        print("   无指示变量 F1={:.3f}  增益={:+.3f}".format(
            r["no_indicator"]["f1"], r["indicator_gain"]))
    print("   泄漏线 F1={:.3f} ({} 特征)".format(r["leaky"]["f1"], r["leaky"]["n_features"]))
