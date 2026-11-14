"""跑一遍三份演示数据，看分数是否在合理区间。

合理区间（我自己定的，跑完对照）：
    clean       macro-F1 应该 > 0.85（三类有信号）
    missing     正例 F1 应该 > 0.65（缺失本身有信息）
    imbalanced  正例 F1 应该 > 0.55（5% 正例，能学到模式）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from executor import ResearchSession

CASES = [
    ('data/demo/clean_balanced.csv', 'species', 'f1_macro', '1',
     {'baseline': 0.85, 'label': 'clean'}),
    ('data/demo/missing_informative.csv', 'churned', 'f1_minority', '1',
     {'baseline': 0.65, 'label': 'missing'}),
    ('data/demo/imbalanced_fraud.csv', 'is_fraud', 'f1_minority', '1',
     {'baseline': 0.55, 'label': 'imbalanced'}),
]

PREPS = [
    ('默认', {}),
    ('+isna', {'add_missing_indicator': True}),
    ('+weight', {'imbalance_strategy': 'class_weight'}),
    ('+both', {'add_missing_indicator': True, 'imbalance_strategy': 'class_weight'}),
    ('+ordinal', {'add_missing_indicator': True, 'imbalance_strategy': 'class_weight',
                  'encode': 'ordinal'}),
]


def main():
    ok = True
    for path, target, metric, pos, meta in CASES:
        if not Path(path).is_file():
            print(f'缺文件: {path}')
            ok = False
            continue
        print(f"\n=== {meta['label']} ({path}) ===")
        best = 0.0
        for name, prep in PREPS:
            session = ResearchSession(path, target, metric=metric, pos_label=pos)
            result = session.run('logistic_regression', {}, preprocessing=prep)
            score = result['validation_score']
            best = max(best, score)
            print(f'  {name:10s} {score:.4f}')
        passed = best >= meta['baseline']
        print(f"  最好 {best:.4f}  "
              f"{'达标' if passed else '偏低了（可能信号不够）'}（参考线 {meta['baseline']}）")
        if not passed:
            ok = False
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
