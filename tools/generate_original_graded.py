"""生成一组难度递进的数据，用来看本地 3B 到底能做到哪一档。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]/'fixtures/graded'
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261115)


def build(name, rows, classes, features, missing, imbalance, noise, categorical=True):
    x = rng.normal(0, 1, (rows, features))
    weights = rng.normal(0, 1, features)
    signal = x @ weights + rng.normal(0, noise, rows)
    if classes == 2:
        cut = np.quantile(signal, 1 - imbalance)
        y = (signal > cut).astype(int)
        labels = np.array(['no', 'yes'])
    else:
        edges = np.quantile(signal, np.linspace(0, 1, classes + 1)[1:-1])
        y = np.digitize(signal, edges)
        labels = np.array(['c%d' % i for i in range(classes)])
    frame = pd.DataFrame(x, columns=['f%02d' % (i + 1) for i in range(features)]).round(4)
    if categorical:
        frame['region'] = rng.choice(['north', 'south', 'east', 'west'], rows)
        frame['channel'] = rng.choice(['a', 'b', 'c'], rows, p=[.6, .3, .1])
    frame['label'] = labels[y]
    for col in frame.columns[:max(1, features // 3)]:
        frame.loc[rng.random(rows) < missing, col] = np.nan
    frame.to_csv(OUT / (name + '.csv'), index=False)
    return frame, labels[y]


SPECS = [
    ('g1_easy', 1200, 2, 8, 0.00, 0.30, 1.0),
    ('g2_mild', 2000, 2, 12, 0.05, 0.18, 1.4),
    ('g3_medium', 3000, 3, 18, 0.10, 0.12, 1.9),
    ('g4_hard', 4000, 4, 28, 0.18, 0.08, 2.4),
    ('g5_harder', 5000, 6, 40, 0.25, 0.04, 3.0),
    ('g6_extreme', 6000, 8, 60, 0.35, 0.02, 3.6),
]

for name, rows, classes, features, missing, imbalance, noise in SPECS:
    frame, y = build(name, rows, classes, features, missing, imbalance, noise)
    counts = pd.Series(y).value_counts(normalize=True).round(3).to_dict()
    print('%-11s 行%5d 列%3d 类别%2d 缺失%4.0f%% 分布%s' % (
        name, len(frame), frame.shape[1], classes, missing * 100, counts))
print('写入', OUT)
