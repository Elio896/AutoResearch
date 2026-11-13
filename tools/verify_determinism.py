"""E2 验证：同 seed 重跑，指标差必须 ≤ 1e-6。

轨迹规范第 06 项要求 E3（第三方能重建），但 E2 是**验收门槛**——
先证明"同一个人、同一份代码、同一个 seed，跑两遍数字一样"，
才谈得上"第三方也能复现"。

用法：
  .venv\\Scripts\\python.exe tools\\verify_determinism.py bench\\tasks\\t2_imbalanced\\data.csv ^
      --target target --metric f1_minority --pos-label 1
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from executor import ResearchSession       # noqa: E402

TOLERANCE = 1e-6


def run_once(csv, target, model, params, metric, pos_label, seed=42):
    session = ResearchSession(csv, target, seed=seed, metric=metric, pos_label=pos_label)
    result = session.run(model=model, parameters=params)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv')
    p.add_argument('--target', required=True)
    p.add_argument('--metric', default='f1_macro')
    p.add_argument('--pos-label', default='1')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output')
    args = p.parse_args()

    cases = [
        ('logistic_regression', {'class_weight': 'balanced'}),
        ('random_forest', {'n_estimators': 50, 'max_depth': 5, 'class_weight': 'balanced'}),
        ('hist_gradient_boosting', {'max_depth': 4, 'class_weight': 'balanced'}),
    ]

    rows = []
    all_ok = True
    for model, params in cases:
        first = run_once(args.csv, args.target, model, params, args.metric, args.pos_label, args.seed)
        second = run_once(args.csv, args.target, model, params, args.metric, args.pos_label, args.seed)
        a = float(first.get('validation_score', first.get('validation_f1', 0)))
        b = float(second.get('validation_score', second.get('validation_f1', 0)))
        diff = abs(a - b)
        ok = diff <= TOLERANCE
        all_ok = all_ok and ok
        rows.append({'model': model, 'run1': a, 'run2': b, 'abs_diff': diff, 'ok': ok,
                     'dataset_sha256_match': first.get('dataset_sha256') == second.get('dataset_sha256'),
                     'split_match': first.get('split_sizes') == second.get('split_sizes')})
        print(f'{model:26s} run1={a:.10f} run2={b:.10f} diff={diff:.2e} {"OK" if ok else "FAIL"}')

    summary = {'all_deterministic': all_ok, 'tolerance': TOLERANCE, 'cases': rows,
               'level': 'E2' if all_ok else 'below_E2'}
    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
