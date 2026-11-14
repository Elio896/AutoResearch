"""第三步验收：在**真实跑出来的轨迹**上找失败模式，并要求每条记录都能指回事件。

做两件事：
1. 用一个"只会重复同一个方案"的提案器真跑一次 ResearchLoop，然后检测——
   验证失败模式来自实际事件流，不是编的；
2. 用只有一次实验的运行检测——验证证据不足时会说"不足以判断"。
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import failure_modes  # noqa: E402
from agents import Budget  # noqa: E402
from executor import ResearchSession  # noqa: E402
from research_loop import ResearchLoop  # noqa: E402
from strategy_store import StrategyStore  # noqa: E402

WORK = Path('artifacts') / 'failure-modes-acceptance'

EXPECTED_TYPES = {
    'repeated_proposal': {'scientist_proposal', 'cloud_proposal'},
    'invalid_structure_repeat': {'scientist_error', 'proposal_invalid'},
    'same_family_tuning_no_gain': {'experiment'},
    'repeated_change_no_gain': {'scientist_proposal', 'cloud_proposal'},
}


def make_csv(path: Path) -> None:
    rng = np.random.default_rng(11)
    n = 260
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.integers(0, 4, n)
    logit = 1.1 * a - 0.6 * b + 0.4 * c
    y = (1 / (1 + np.exp(-logit)) > rng.random(n)).astype(int)
    pd.DataFrame({'a': a, 'b': b, 'c': c, 'target': y}).to_csv(path, index=False)


def stuck_proposer(messages, timeout=None):
    """永远提同一个方案——真实运行里就会被记成重复提案。"""
    return {'message': {'content': json.dumps({
        'model': 'logistic_regression',
        'parameters': {'max_iter': 500},
        'hypothesis': '还是同一套',
        'expected_direction': 'up',
        'falsifier': '分数没变',
    })}, 'prompt_eval_count': 0, 'eval_count': 0}


def check_traceability(events: list[dict], result: dict) -> bool:
    ok = True
    for record in result['records']:
        expected = EXPECTED_TYPES.get(record['failure_type'])
        if expected is None:
            continue
        for ref in record['event_refs']:
            if ref.startswith('worklog:'):
                continue
            index = int(ref.split('_')[1])
            if index >= len(events):
                print(f'    !! {ref} 越界')
                ok = False
                continue
            actual = events[index].get('event')
            flag = 'ok' if actual in expected else '不匹配'
            print(f'    {ref} → {actual}  {flag}')
            ok &= actual in expected
    return ok


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)
    csv_path = WORK / 'toy.csv'
    make_csv(csv_path)

    store = StrategyStore(WORK / 'strategy_store.json')
    strategy = store.bootstrap({'planning_prompt': '基线规划', 'classifier_prompt': '基线分类',
                                'candidate_selection_rules': []}, reason='验收基线')

    print('== 1. 真跑一次（提案器只会重复同一个方案）==')
    session = ResearchSession(csv_path, 'target', metric='f1_macro', pos_label='1')
    loop = ResearchLoop(session, proposer=stuck_proposer,
                        budget=Budget(max_experiments=4, max_seconds=180.0, max_local_calls=8),
                        strategy_store=store, run_id='run_stuck')
    outcome = loop.run()
    events = outcome['events']
    summary = outcome['summary']
    print(f'  实验 {summary["experiments"]} 次，best={summary["best_score"]}，'
          f'策略={summary["strategy_id"]}')
    print(f'  事件 {len(events)} 条，其中提案事件 '
          f'{sum(1 for e in events if e["event"] == "scientist_proposal")} 条')

    print('== 2. 检测 ==')
    found = failure_modes.detect(events, run_id=summary['trajectory_run_id'],
                                 strategy_id=summary['strategy_id'])
    print(f'  verdict={found["verdict"]}  failure_count={found["failure_count"]}  '
          f'insufficient={found["insufficient"]}')
    for record in found['records']:
        print(f'  - {record["failure_type"]}: refs={record["event_refs"]} '
              f'measurements={json.dumps(record["measurements"], ensure_ascii=False)}')

    print('== 3. 逐条核对引用是否指向真实事件 ==')
    traceable = check_traceability(events, found)
    print(f'  可追溯：{traceable}')

    print('== 4. 证据不足的运行 ==')
    session2 = ResearchSession(csv_path, 'target', metric='f1_macro', pos_label='1')
    loop2 = ResearchLoop(session2, proposer=stuck_proposer,
                         budget=Budget(max_experiments=1, max_seconds=60.0, max_local_calls=3),
                         strategy_store=store, run_id='run_tiny')
    outcome2 = loop2.run()
    thin = failure_modes.detect(outcome2['events'], run_id='run_tiny',
                               strategy_id=strategy_id(outcome2))
    print(f'  verdict={thin["verdict"]}  insufficient={thin["insufficient"]}')
    print(f'  原因：{thin["insufficient_reasons"]}')
    print(f'  失败记录条数：{thin["failure_count"]}（证据不够就不写经验）')

    ok = (traceable and found['failure_count'] >= 1
          and thin['insufficient'] is True and thin['failure_count'] == 0)

    report = WORK / 'report.json'
    report.write_text(json.dumps({'stuck_run': found, 'thin_run': thin},
                                 ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n报告：{report.resolve()}')
    print('判定：' + ('通过——失败模式来自真实事件、可逐条追溯、证据不足时拒绝下结论'
                     if ok else '失败'))
    return 0 if ok else 1


def strategy_id(outcome: dict) -> str | None:
    return (outcome.get('summary') or {}).get('strategy_id')


if __name__ == '__main__':
    raise SystemExit(main())
