"""跑一次五角色研究闭环（真实数据 + 本地模型）。

用法：
  .venv\\Scripts\\python.exe tools\\run_five_roles.py bench\\tasks\\t2_imbalanced\\data.csv ^
      --target target --metric f1_minority --pos-label 1 ^
      --model qwen2.5:7b-instruct-q4_K_M --max-experiments 5 --output artifacts\\five_roles_t2.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import Budget                      # noqa: E402
from executor import ResearchSession           # noqa: E402
from experience import ExperienceStore         # noqa: E402
from orchestrator import LocalQwenClient       # noqa: E402
from research_loop import ResearchLoop         # noqa: E402


def main():
    p = argparse.ArgumentParser(description='Run the five-role research loop.')
    p.add_argument('csv')
    p.add_argument('--target', required=True)
    p.add_argument('--metric', default='f1_macro', choices=['f1_macro', 'f1_minority'])
    p.add_argument('--pos-label', default='1')
    p.add_argument('--base-url', default='http://127.0.0.1:11434')
    p.add_argument('--model', default='qwen2.5:7b-instruct-q4_K_M')
    p.add_argument('--max-experiments', type=int, default=5)
    p.add_argument('--max-seconds', type=float, default=300.0)
    p.add_argument('--max-local-calls', type=int, default=6)
    p.add_argument('--max-local-tokens', type=int, default=20000)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--experience', default='', help='经验库 jsonl 路径（留空则不启用）')
    p.add_argument('--task-name', default='')
    p.add_argument('--output')
    args = p.parse_args()

    session = ResearchSession(args.csv, args.target, metric=args.metric, pos_label=args.pos_label)
    proposer = LocalQwenClient(base_url=args.base_url, model=args.model)
    budget = Budget(max_experiments=args.max_experiments, max_seconds=args.max_seconds,
                    max_local_calls=args.max_local_calls, max_local_tokens=args.max_local_tokens)

    def emit(event):
        if event['event'] in ('analyst_profile', 'manager_plan', 'scientist_proposal',
                              'experiment', 'reviewer_verdict', 'finish'):
            print(json.dumps(event, ensure_ascii=False)[:300], flush=True)

    loop = ResearchLoop(session, proposer=proposer, budget=budget,
                        patience=args.patience, emit=emit,
                        experience=ExperienceStore(args.experience) if args.experience else None,
                        task_name=args.task_name or Path(args.csv).stem)
    result = loop.run()

    print('\n=== summary ===')
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print('saved:', args.output)


if __name__ == '__main__':
    main()
