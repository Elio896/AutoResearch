"""复现二德子报的缺陷：fraud 数据里的事后字段 chargeback_filed。

她跑出来的结果：
    保留 chargeback_filed → validation F1 = 0.955
    显式剔除            → 0.635
    差 0.32，而老规则三条全漏。

这里用合成数据复现同样的形状：
    chargeback_filed：0/1 整数，名字不含 id，与标签相关约 0.87（不到老的 0.98 阈值）
期望：新规则能把它标成 candidate。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np           # noqa: E402
import pandas as pd          # noqa: E402

from integrity import inspect_features   # noqa: E402


def build() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    n = 2000
    # 标签：5% 是欺诈（和 fraud_imbalanced 的极端不平衡接近）
    labels = pd.Series(rng.choice(['0', '1'], size=n, p=[0.95, 0.05]))
    fraud = (labels == '1').to_numpy()

    frame = pd.DataFrame({
        'amount': rng.normal(100, 40, n).round(2),
        'hour': rng.integers(0, 24, n),
        'merchant_risk': rng.integers(0, 5, n),
    })
    # 事后字段：欺诈确认立案后才为 1，但有一定噪声，相关约 0.87
    noise = rng.random(n) < 0.10
    chargeback = fraud.astype(int) ^ noise.astype(int)
    frame['chargeback_filed'] = chargeback
    return frame, labels


def main():
    frame, labels = build()
    corr = frame['chargeback_filed'].corr(pd.Series(pd.factorize(labels)[0]))
    print(f'chargeback_filed 与标签相关: {corr:.3f}')
    findings = inspect_features(frame, labels, 'synthetic-fraud', 'target')
    hit = [f for f in findings if f['column'] == 'chargeback_filed']
    print(f'findings: {[(f["column"], f["signals"]) for f in findings]}')
    if hit:
        print('RESULT: 新规则命中了 ->', hit[0]['level'], hit[0]['signals'])
        return 0
    print('RESULT: 仍然漏检（缺陷未修好）')
    return 1


if __name__ == '__main__':
    sys.exit(main())
