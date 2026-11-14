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
    p.add_argument('--small-model', default='qwen2.5:3b-instruct-q4_K_M',
                   help='红队用的小模型；留空则与 --model 相同')
    p.add_argument('--max-experiments', type=int, default=5)
    p.add_argument('--max-seconds', type=float, default=300.0)
    p.add_argument('--max-local-calls', type=int, default=6)
    p.add_argument('--max-local-tokens', type=int, default=20000)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--enable-cloud', action='store_true', help='本地卡住时允许调云端（需 DEEPSEEK_API_KEY）')
    p.add_argument('--max-cloud-calls', type=int, default=1)
    p.add_argument('--enable-jev', action='store_true', help='云端升级前先问 JEV 值不值得（需 TYPESAFE_API_KEY）')
    p.add_argument('--enable-qwen-router', action='store_true', help='让本地千问决定要不要调云端、调哪个档')
    p.add_argument('--enable-jev-harness', action='store_true', help='五道零 token 语义门（省 token 的那层）')
    p.add_argument('--enable-signal-router', action='store_true',
                   help='用可计算信号（首次合法率/重复率/停滞/同类改动Δ）判要不要升级')
    p.add_argument('--enable-task-router', action='store_true',
                   help='任务判别器：千问接不了的活分给 JEV（决策）/ DeepSeek（思考）')
    p.add_argument('--experience', default='', help='经验库 jsonl 路径（留空则不启用）')
    p.add_argument('--task-name', default='')
    p.add_argument('--output')
    args = p.parse_args()

    session = ResearchSession(args.csv, args.target, metric=args.metric, pos_label=args.pos_label)
    proposer = LocalQwenClient(base_url=args.base_url, model=args.model)
    # 模型分层：Scientist 用大模型，RedTeam 用小模型（找茬比提方案简单）
    small_proposer = None
    if args.small_model and args.small_model != args.model:
        small_proposer = LocalQwenClient(base_url=args.base_url, model=args.small_model)
    budget = Budget(max_experiments=args.max_experiments, max_seconds=args.max_seconds,
                    max_local_calls=args.max_local_calls, max_local_tokens=args.max_local_tokens)

    def emit(event):
        if event['event'] in ('analyst_profile', 'manager_plan', 'scientist_proposal',
                              'experiment', 'reviewer_verdict', 'finish'):
            print(json.dumps(event, ensure_ascii=False)[:300], flush=True)

    # 云端兜底：本地卡住时才用，而且先过 JEV
    cloud_proposer = None
    jev_gate = None
    if args.enable_cloud:
        import os
        from cloud_provider import DeepSeekClient
        key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
        if key:
            cloud_proposer = DeepSeekClient(key)
        else:
            print('warning: --enable-cloud 但环境里没有 DEEPSEEK_API_KEY，跳过')
    if args.enable_jev:
        from jev_router import JevGate
        jev_gate = JevGate.auto()

    # 千问路由：让本地 3B 自己判要不要调云端
    qwen_router = None
    if args.enable_qwen_router:
        from qwen_router import decide_escalation
        qwen_router = lambda state: decide_escalation(  # noqa: E731
            state, base_url=args.base_url, model=args.small_model)

    # Jev Harness：五道零 token 的门（省 token 的第一道防线）
    jev_harness = None
    if args.enable_jev_harness:
        from jev_harness import JevHarness
        jev_harness = JevHarness()

    loop = ResearchLoop(session, proposer=proposer, budget=budget,
                        patience=args.patience, emit=emit,
                        experience=ExperienceStore(args.experience) if args.experience else None,
                        task_name=args.task_name or Path(args.csv).stem,
                        small_proposer=small_proposer,
                        cloud_proposer=cloud_proposer,
                        jev_gate=jev_gate,
                        max_cloud_calls=args.max_cloud_calls,
                        qwen_router=qwen_router,
                        jev_harness=jev_harness,
                        use_signal_router=args.enable_signal_router,
                        use_task_router=args.enable_task_router)
    result = loop.run()

    print('\n=== summary ===')
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print('saved:', args.output)


if __name__ == '__main__':
    main()
