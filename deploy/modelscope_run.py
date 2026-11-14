"""ModelScope 免费 GPU Notebook 里一键跑 EvoResearch。

用法（在 ModelScope 的 Jupyter 里）：
    1. 把整个项目传上去（或 git clone）
    2. 打开这个文件，或直接在一个 cell 里执行：
           !bash deploy/modelscope_setup.sh
    3. 然后：
           python deploy/modelscope_run.py --csv data/demo/clean_balanced.csv --target species

ModelScope 免费 GPU 一般是 16GB（如 V100 16G 或 T4 16G），
跑 Qwen2.5-7B-Instruct 的 FP16 没问题，不用量化。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 项目根目录加进 path（保证在 Notebook 任意位置都能 import）
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def pick_qwen_model() -> str:
    """ModelScope 上直接跑官方 Qwen2.5-7B-Instruct，不量化。"""
    # 优先用 ModelScope 官方推理 API；没 key 就回退到 transformers 本地加载
    import os
    key = os.environ.get('DASHSCOPE_API_KEY') or os.environ.get('MODELSCOPE_API_KEY')
    if key:
        return 'dashscope:qwen2.5-7b-instruct'
    # 本地 transformers 加载（ModelScope 镜像里可能已预装）
    try:
        import torch
        if torch.cuda.is_available():
            return 'transformers:qwen/Qwen2.5-7B-Instruct'
    except Exception:                       # noqa: BLE001
        pass
    return 'qwen2.5:7b-instruct-q4_K_M'     # 最后退路：本地 ollama


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--metric', default='f1_macro')
    parser.add_argument('--pos-label', default='1')
    parser.add_argument('--max-experiments', type=int, default=5)
    parser.add_argument('--use-mde', action='store_true',
                        help='多种子复核（慢 5 倍，但能过滤噪声提升）')
    args = parser.parse_args()

    from executor import ResearchSession
    from research_loop import ResearchLoop
    from agents import Budget

    session = ResearchSession(args.csv, args.target,
                              metric=args.metric, pos_label=args.pos_label)
    model = pick_qwen_model()

    # 在 ModelScope 上，千问走 transformers 或 dashscope，不依赖本地 ollama
    proposer = None
    if model.startswith('transformers:'):
        from orchestrator import HFClient
        proposer = HFClient(model.split(':', 1)[1])
    elif model.startswith('dashscope:'):
        from orchestrator import DashScopeClient
        proposer = DashScopeClient(model.split(':', 1)[1])

    budget = Budget(max_experiments=args.max_experiments,
                    max_seconds=600.0,
                    max_local_calls=args.max_experiments + 2)
    loop = ResearchLoop(session, proposer=proposer, budget=budget,
                        use_mde=args.use_mde)
    result = loop.run()

    print('\n=== 结果 ===')
    summary = result['summary']
    for key in ('best_model', 'best_score', 'evoundo_versions',
                'evoundo_rollbacks', 'stop_reason'):
        if key in summary:
            print(f'{key}: {summary[key]}')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
