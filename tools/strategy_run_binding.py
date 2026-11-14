"""第一步验收（二）：真实研究运行绑定策略版本；运行中库里换版本，这次运行不受影响。

跑法：
    .venv\\Scripts\\python.exe tools\\strategy_run_binding.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agents import Budget  # noqa: E402
from executor import ResearchSession  # noqa: E402
from research_loop import ResearchLoop  # noqa: E402
from strategy_store import StrategyStore  # noqa: E402

BASE = {
    'classifier_prompt': '按类别分布选择指标。',
    'planning_prompt': '先看缺失与唯一值，再选基线模型。',
    'candidate_selection_rules': [{'when': 'imbalanced', 'prefer': 'class_weight'}],
}


def make_csv(path: Path) -> None:
    rng = np.random.default_rng(7)
    n = 300
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.integers(0, 3, n)
    logit = 1.2 * a - 0.8 * b + 0.5 * c
    y = (1 / (1 + np.exp(-logit)) > rng.random(n)).astype(int)
    pd.DataFrame({'a': a, 'b': b, 'c': c, 'target': y}).to_csv(path, index=False)


def stub_proposer(messages, timeout=None):
    return {
        'message': {'content': json.dumps({
            'model': 'logistic_regression',
            'parameters': {'max_iter': 500},
            'hypothesis': '基线先跑通，确认链路',
            'expected_direction': 'flat',
            'falsifier': '分数没有变化',
        })},
        'prompt_eval_count': 0,
        'eval_count': 0,
    }


def main() -> int:
    work = Path('artifacts') / 'strategy-run-binding'
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    csv_path = work / 'toy.csv'
    make_csv(csv_path)

    store = StrategyStore(work / 'strategy_store.json')
    base = store.bootstrap(BASE, reason='基线策略')
    print(f'初始正式策略：{base["strategy_id"]}')

    session = ResearchSession(csv_path, 'target', metric='f1_macro', pos_label='1')
    loop = ResearchLoop(session, proposer=stub_proposer,
                        budget=Budget(max_experiments=2, max_seconds=120.0, max_local_calls=6),
                        strategy_store=store, run_id='run_pin_demo')
    pinned_id = loop.strategy['strategy_id']
    print(f'构造 ResearchLoop 时钉住：{pinned_id}')

    # 运行还没开始，库里就被换了版本（模拟别人在中间升级）
    candidate = store.propose({'planning_prompt': '换了新规划：先查事后字段。'},
                              reason='有人在中间升级', evidence_run_ids=['run_x'])
    store.activate(candidate['strategy_id'], checks={'ok': True},
                   jev={'decision': 'upgrade'}, by='intruder')
    print(f'运行开始前库里已被切成：{store.active()["strategy_id"]}')

    result = loop.run()
    summary = result['summary']
    pinned_events = [e for e in result['events'] if e['event'] == 'strategy_pinned']
    print('这次运行的结果：')
    print(f'  status={summary["status"]}  experiments={summary["experiments"]}  '
          f'best={summary["best_score"]}')
    print(f'  它用的策略：{summary["strategy_id"]}（运行前库里已经是 {store.active()["strategy_id"]}）')
    print(f'  pin 自查 intact={summary["strategy_pin_intact"]}')
    print(f'  strategy_pinned 事件：{pinned_events[0] if pinned_events else "缺失"}')

    # 新开一个 loop 才会用新策略
    session2 = ResearchSession(csv_path, 'target', metric='f1_macro', pos_label='1')
    loop2 = ResearchLoop(session2, proposer=stub_proposer,
                         budget=Budget(max_experiments=1, max_seconds=60.0, max_local_calls=3),
                         strategy_store=store, run_id='run_pin_demo_2')
    print(f'新构造的 loop 钉住：{loop2.strategy["strategy_id"]}（才是切换后的版本）')

    ok = (summary['strategy_id'] == pinned_id
          and summary['strategy_pin_intact'] is True
          and loop2.strategy['strategy_id'] == store.active()['strategy_id'])
    print('\n判定：' + ('通过——运行按钉住的那一版跑完，新运行才用新版本'
                       if ok else '失败'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
