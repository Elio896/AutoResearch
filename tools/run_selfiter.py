"""自迭代对照实验：证明"越用越好"，或者诚实地证明"没观察到"。

两组：
  实验组：同一个经验库累积，跑 N 轮
  对照组：每轮用全新的空库（研究本身不带记忆）

三条判据（任一条成立即算改进，但要说明是哪条）：
  1. experiments_used 下降
  2. local_tokens 下降
  3. elapsed_seconds 下降

如果都不成立，就照实写"未观察到改进"——不许改口径凑。

用法：
  .venv\\Scripts\\python.exe tools\\run_selfiter.py bench\\tasks\\t2_imbalanced\\data.csv ^
      --target target --metric f1_minority --pos-label 1 --rounds 3 --output artifacts\\selfiter.json
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import Budget                     # noqa: E402
from executor import ResearchSession          # noqa: E402
from experience import ExperienceStore        # noqa: E402
from orchestrator import LocalQwenClient      # noqa: E402
from research_loop import ResearchLoop        # noqa: E402


def run_round(args, experience_path, round_index):
    session = ResearchSession(args.csv, args.target, metric=args.metric, pos_label=args.pos_label)
    proposer = LocalQwenClient(base_url=args.base_url, model=args.model)
    budget = Budget(max_experiments=args.max_experiments,
                    max_seconds=args.max_seconds,
                    max_local_calls=args.max_local_calls)
    store = ExperienceStore(experience_path) if experience_path else None
    loop = ResearchLoop(session, proposer=proposer, budget=budget,
                        patience=args.patience, experience=store,
                        task_name=args.task_name or Path(args.csv).stem)
    result = loop.run()
    summary = result['summary']
    return {
        'round': round_index,
        'best_score': summary.get('best_score'),
        'best_model': summary.get('best_model'),
        'experiments_used': summary.get('experiments'),
        'local_tokens': summary.get('local_tokens'),
        'local_calls': summary.get('local_calls'),
        'elapsed_seconds': summary.get('elapsed_seconds'),
        'experience_count': len(store.all()) if store else 0,
        'stop_reason': summary.get('stop_reason'),
    }


def trend(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    if len(vals) < 2:
        return None
    return round(vals[-1] - vals[0], 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv')
    p.add_argument('--target', required=True)
    p.add_argument('--metric', default='f1_macro')
    p.add_argument('--pos-label', default='1')
    p.add_argument('--model', default='qwen2.5:7b-instruct-q4_K_M')
    p.add_argument('--base-url', default='http://127.0.0.1:11434')
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--max-experiments', type=int, default=3)
    p.add_argument('--max-local-calls', type=int, default=3)
    p.add_argument('--max-seconds', type=float, default=180.0)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--task-name', default='')
    p.add_argument('--output', default='')
    args = p.parse_args()

    started = time.time()
    results = {}

    # 实验组：同一库累积
    with tempfile.TemporaryDirectory() as tmp:
        shared = str(Path(tmp) / 'shared.jsonl')
        rows = [run_round(args, shared, i + 1) for i in range(args.rounds)]
        results['treatment'] = rows
        print('=== treatment (累积经验) ===')
        for r in rows:
            print(f"  r{r['round']}: score={r['best_score']} exp={r['experiments_used']} "
                  f"tokens={r['local_tokens']} t={r['elapsed_seconds']}s "
                  f"exp_pool={r['experience_count']}")

    # 对照组：每轮新库
    rows = []
    for i in range(args.rounds):
        with tempfile.TemporaryDirectory() as tmp:
            fresh = str(Path(tmp) / 'fresh.jsonl')
            rows.append(run_round(args, fresh, i + 1))
    results['control'] = rows
    print('=== control (每轮清空经验) ===')
    for r in rows:
        print(f"  r{r['round']}: score={r['best_score']} exp={r['experiments_used']} "
              f"tokens={r['local_tokens']} t={r['elapsed_seconds']}s "
              f"exp_pool={r['experience_count']}")

    # 三条判据逐条判
    verdicts = {}
    for key in ('experiments_used', 'local_tokens', 'elapsed_seconds'):
        t = trend(results['treatment'], key)
        c = trend(results['control'], key)
        if t is None or c is None:
            verdicts[key] = {'improved': None, 'reason': 'not_enough_rounds'}
        else:
            # 实验组下降得比对照组多，才算"记忆带来了改进"
            verdicts[key] = {
                'treatment_delta': t,
                'control_delta': c,
                'improved': t < c,
                'reason': 'treatment_dropped_more' if t < c else 'no_observed_gain',
            }
    results['verdicts'] = verdicts
    results['any_improved'] = any(v.get('improved') for v in verdicts.values())
    results['elapsed_total'] = round(time.time() - started, 1)

    print('\n=== 判据 ===')
    for key, v in verdicts.items():
        print(f"  {key}: treatment={v.get('treatment_delta')} control={v.get('control_delta')} "
              f"improved={v.get('improved')}")
    print('any_improved:', results['any_improved'])

    if args.output:
        Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print('saved:', args.output)


if __name__ == '__main__':
    main()
