"""生成三份适合演示和验收的测试数据。

原则（按妈妈定的范围）：
    - 表格分类，样本 200-1000 行（跑得快，但足够看出差异）
    - 每份都有**真实信号**，不是纯噪声——否则分数恒为 0 看不出东西
    - 三个场景各打一个点：干净 / 缺失 / 不平衡

设计要点：
    * 干净场景：类别均衡，线性可分性中等，好模型能到 0.9+
    * 缺失场景：缺失**本身有信息**（缺失比例与标签相关），
      这样 add_missing_indicator 才有用武之地
    * 不平衡场景：正例 ~5%，且正例有可学的模式，
      class_weight / 正例 F1 才有意义
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_clean(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """干净三分类：6 个数值特征，3 个类别，中等重叠度。"""
    rng = np.random.default_rng(seed)
    per_class = n // 3
    rows = []
    # 三个类别的中心，故意留些重叠，不然 F1 会直接 1.0 显得假
    centers = {
        'A': [0.0, 0.0, 0.0, 1.0, -1.0, 0.5],
        'B': [2.0, 1.0, -0.5, 0.0, 1.0, -1.0],
        'C': [-1.5, 2.0, 1.0, -1.0, 0.0, 1.2],
    }
    for label, center in centers.items():
        for _ in range(per_class):
            point = np.array(center) + rng.normal(0, 1.1, 6)
            rows.append({'f1': point[0], 'f2': point[1], 'f3': point[2],
                         'f4': point[3], 'f5': point[4], 'f6': point[5],
                         'species': label})
    frame = pd.DataFrame(rows).round(3)
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_missing(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """缺失场景：缺失比例和标签相关，isna 指示变量应该有信息量。

    信号强度刻意压低——上一次太强，F1 直接 0.98，演示时看不出任何开关的差别。
    现在均值差减半、噪声加大，目标是「能学但学不满」，参考线 0.65~0.85。
    """
    rng = np.random.default_rng(seed)
    labels = rng.choice([0, 1], n, p=[0.55, 0.45])
    rows = []
    for y in labels:
        base = 0.9 if y == 1 else -0.9          # 原来 1.8，砍一半
        f1 = base + rng.normal(0, 1.5)          # 噪声从 1.2 加到 1.5
        f2 = base * 0.5 + rng.normal(0, 1.8)    # 噪声从 1.4 加到 1.8
        f3 = rng.normal(0, 1.3) + (0.4 if y == 1 else -0.2)
        f4 = rng.integers(0, 6) + (1 if y == 1 else 0)
        # 关键设计：正例更容易缺 f2 —— 缺失本身携带标签信息
        missing_rate = 0.40 if y == 1 else 0.15
        if rng.random() < missing_rate:
            f2 = np.nan
        if rng.random() < 0.12:
            f3 = np.nan
        rows.append({'age': round(float(f1), 2), 'income': round(float(f2), 2),
                     'score': round(float(f3), 3), 'visits': int(f4),
                     'segment': rng.choice(['basic', 'plus', 'vip']),
                     'churned': int(y)})
    return pd.DataFrame(rows)


def make_imbalanced(n: int = 1000, seed: int = 2026) -> pd.DataFrame:
    """不平衡场景：正例约 5%，正例有可学模式但不平凡。

    同样压低信号：上一版 class_weight 反而把分数拉低，说明模式太容易学、
    加权只是在放大噪声。目标是参考线 0.55~0.75。
    """
    rng = np.random.default_rng(seed)
    labels = rng.choice([0, 1], n, p=[0.95, 0.05])
    rows = []
    for y in labels:
        if y == 1:
            amount = rng.lognormal(4.0, 0.9)
            hour = rng.choice([1, 2, 3, 4, 23, 12, 15])   # 掺入白天的，削弱信号
            foreign = rng.choice([0, 1], p=[0.55, 0.45])
            retry = rng.integers(0, 4)
            risk = rng.normal(0.60, 0.20)
        else:
            amount = rng.lognormal(3.6, 0.9)
            hour = rng.integers(6, 23)
            foreign = rng.choice([0, 1], p=[0.92, 0.08])
            retry = rng.choice([0, 0, 0, 1])
            risk = rng.normal(0.30, 0.20)
        rows.append({
            'amount': round(float(amount), 2),
            'hour_of_day': int(hour),
            'is_foreign': int(foreign),
            'cvv_retry': int(retry),
            'device_risk': round(float(np.clip(risk, 0.01, 0.99)), 3),
            'txn_count_24h': int(max(0, rng.poisson(2.8 if y else 3.6))),
            'card_age_days': int(rng.integers(30, 3000)),
            'channel': rng.choice(['app', 'web', 'pos'], p=[0.6, 0.3, 0.1]),
            'is_fraud': int(y),
        })
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def describe(frame: pd.DataFrame, target: str, name: str) -> str:
    counts = frame[target].value_counts(normalize=True).round(4).to_dict()
    missing = round(float(frame.isna().mean().mean()), 4)
    return (f'{name:12s} rows={len(frame):5d} features={len(frame.columns) - 1:3d} '
            f'classes={frame[target].nunique()} missing={missing:.2%} dist={counts}')


def main() -> int:
    out = Path('data/demo')
    out.mkdir(parents=True, exist_ok=True)

    clean = make_clean()
    missing = make_missing()
    imbalanced = make_imbalanced()

    clean.to_csv(out / 'clean_balanced.csv', index=False)
    missing.to_csv(out / 'missing_informative.csv', index=False)
    imbalanced.to_csv(out / 'imbalanced_fraud.csv', index=False)

    print(describe(clean, 'species', 'clean'))
    print(describe(missing, 'churned', 'missing'))
    print(describe(imbalanced, 'is_fraud', 'imbalanced'))
    print(f'\nwritten to {out.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
