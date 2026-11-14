"""验证预处理开关的效果。

二德子给的实测：readmission 加上 isna 指示变量后，正例 F1 从 0.165 到 0.443。
这里在 t1（缺失场景）上把四种组合跑一遍，看数字。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from executor import ResearchSession


def run(csv, metric, pos_label, prep, label):
    session = ResearchSession(csv, 'target', metric=metric, pos_label=pos_label)
    result = session.run('logistic_regression', {}, preprocessing=prep)
    score = result['validation_score']
    print(f'{label:34s} {score:.4f}')
    return score


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else 'bench/tasks/t1_missing/data.csv'
    metric = sys.argv[2] if len(sys.argv) > 2 else 'f1_minority'
    pos = sys.argv[3] if len(sys.argv) > 3 else '1'

    print(f'data: {csv}  metric: {metric}\n')
    run(csv, metric, pos, {'imbalance_strategy': 'none'}, 'baseline (median, none)')
    run(csv, metric, pos, {'imbalance_strategy': 'none',
                           'add_missing_indicator': True}, '+ missing indicator')
    run(csv, metric, pos, {'imbalance_strategy': 'class_weight'}, '+ class_weight')
    run(csv, metric, pos, {'imbalance_strategy': 'class_weight',
                           'add_missing_indicator': True}, '+ both')
    run(csv, metric, pos, {'imbalance_strategy': 'class_weight',
                           'add_missing_indicator': True,
                           'impute': 'mean'}, '+ both, mean impute')
    run(csv, metric, pos, {'imbalance_strategy': 'class_weight',
                           'add_missing_indicator': True,
                           'encode': 'ordinal'}, '+ both, ordinal encode')


if __name__ == '__main__':
    main()
