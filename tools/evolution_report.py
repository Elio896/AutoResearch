"""跨任务进化总账：把所有任务的版本链和经验池串起来看。

用法：
  .venv\\Scripts\\python.exe tools\\evolution_report.py
  .venv\\Scripts\\python.exe tools\\evolution_report.py --json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experience import ExperienceStore   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def scan_runs() -> list[dict]:
    runs = []
    base = ROOT / 'artifacts' / 'evoundo'
    if not base.exists():
        return runs
    for run_dir in sorted(base.iterdir()):
        versions_file = run_dir / 'versions.json'
        if not versions_file.is_file():
            continue
        try:
            data = json.loads(versions_file.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            continue
        runs.append({
            'run_id': data.get('run_id') or run_dir.name,
            'versions': len(data.get('versions') or []),
            'rollbacks': data.get('rollbacks', 0),
            'best_score': data.get('best_score'),
            'current': data.get('current_id'),
            'best': data.get('best_id'),
        })
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experience', default=str(ROOT / 'artifacts' / 'experience.jsonl'))
    p.add_argument('--json', action='store_true')
    args = p.parse_args()

    runs = scan_runs()
    store = ExperienceStore(args.experience)
    lineage = store.lineage()

    report = {
        'tasks': len(runs),
        'total_versions': sum(r['versions'] for r in runs),
        'total_rollbacks': sum(r['rollbacks'] for r in runs),
        'best_scores': [r['best_score'] for r in runs if r['best_score'] is not None],
        'experience': lineage,
        'runs': runs,
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print('=== 跨任务进化总账 ===')
    print(f"任务数: {report['tasks']}")
    print(f"版本总数: {report['total_versions']}")
    print(f"回滚总数: {report['total_rollbacks']}")
    if report['best_scores']:
        print(f"最好成绩: {max(report['best_scores']):.4f}")
    print()
    print('=== 经验池 ===')
    for key in ('observed', 'ever_used', 'total_uses', 'successes',
                'contradictions', 'success_rate', 'active', 'retired'):
        print(f"  {key}: {lineage.get(key)}")
    print()
    if runs:
        print('=== 各任务 ===')
        for r in runs:
            print(f"  {r['run_id'][:18]:20s} versions={r['versions']:<3} "
                  f"rollbacks={r['rollbacks']:<2} best={r['best_score']}")


if __name__ == '__main__':
    main()
