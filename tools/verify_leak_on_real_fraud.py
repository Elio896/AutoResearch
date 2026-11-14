"""用二德子的**真数据**验证泄漏规则。

fraud_imbalanced.csv 里有两个可疑列：
    chargeback_filed  —— 事后字段（立案后才是 1），应该抓
    is_foreign        —— 真实特征（和标签基本无关），不该抓

她给的统计：
    chargeback_filed  一致率 0.994 / kappa 0.925 / phi 0.926
    is_foreign        一致率 0.924 / kappa 0.214 / phi 0.219
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                      # noqa: E402
from sklearn.metrics import cohen_kappa_score   # noqa: E402

from integrity import POST_HOC_PATTERN, inspect_features   # noqa: E402

CSV = Path('docs/交付包/01-测试数据/fraud_imbalanced.csv')


def main():
    if not CSV.is_file():
        print('missing', CSV)
        return 2
    frame = pd.read_csv(CSV)
    labels = frame['is_fraud'].astype(str)
    features = frame.drop(columns=['is_fraud'])
    print(f'rows={len(frame)} cols={len(features.columns)}')
    print(f'正例率={(labels == "1").mean():.4f}')

    encoded = pd.Series(pd.factorize(labels)[0])
    for name in ('chargeback_filed', 'is_foreign'):
        if name not in features.columns:
            print(f'{name}: 不在数据里')
            continue
        col = pd.to_numeric(features[name], errors='coerce')
        corr = col.corr(encoded)
        kappa = cohen_kappa_score(col.round().astype(int), encoded.astype(int))
        print(f'{name}: corr={corr:.3f} kappa={kappa:.3f} '
              f'name_match={bool(POST_HOC_PATTERN.search(name.lower()))}')

    findings = inspect_features(features, labels, 'real-fraud', 'is_fraud')
    print('\nfindings:')
    for f in findings:
        print(f'  {f["column"]:24s} {f["level"]:10s} {f["signals"]}')

    flagged = {f['column'] for f in findings}
    ok_chargeback = 'chargeback_filed' in flagged
    bad_foreign = 'is_foreign' in flagged
    print()
    print('chargeback_filed 被抓到:', ok_chargeback)
    print('is_foreign 被误报:', bad_foreign)
    return 0 if (ok_chargeback and not bad_foreign) else 1


if __name__ == '__main__':
    sys.exit(main())
