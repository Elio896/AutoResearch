"""给演示页面准备一份**真实跑出来**的策略库（不是手写 JSON）。

做法：真跑两组对照评估 → 真的走 decide() → 真的 activate。
唯一"模拟"的是 Jev 的回答（没有 key），所以事件里 `by` 写 `seed-demo-simulated`、
`jev.model` 写 `simulated-jev`，页面上看得见。
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from strategy_decision import decide  # noqa: E402
from strategy_eval import compare, freeze_criteria, run_arm  # noqa: E402
from strategy_store import StrategyStore  # noqa: E402

WORK = Path('artifacts') / 'seed-demo'


def make_task(path: Path, seed: int, n: int = 160) -> None:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.integers(0, 3, n)
    y = (1 / (1 + np.exp(-(1.0 * a - 0.6 * b + 0.4 * c))) > rng.random(n)).astype(int)
    pd.DataFrame({'a': a, 'b': b, 'c': c, 'target': y}).to_csv(path, index=False)


def repeating(changes):
    def proposer(messages, timeout=None):
        return {'message': {'content': json.dumps({
            'model': 'logistic_regression', 'parameters': {'max_iter': 400},
            'hypothesis': '同一套再试一次', 'expected_direction': 'up',
            'falsifier': '分数没变'})}, 'prompt_eval_count': 0, 'eval_count': 0}
    return proposer


def varying(changes):
    state = {'i': 0}

    def proposer(messages, timeout=None):
        state['i'] += 1
        return {'message': {'content': json.dumps({
            'model': 'random_forest', 'parameters': {'n_estimators': 30 + 15 * state['i']},
            'hypothesis': f'第 {state["i"]} 个方案', 'expected_direction': 'up',
            'falsifier': '分数没变'})}, 'prompt_eval_count': 0, 'eval_count': 0}
    return proposer


def simulated_jev(answer: str):
    def _gate(state, questions):
        return {'ok': True, 'model': 'simulated-jev',
                'answers': {'strategy_decision': answer}}
    return _gate


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)
    tasks = [WORK / 'eval_a.csv', WORK / 'eval_b.csv']
    make_task(tasks[0], 2026)
    make_task(tasks[1], 77)

    store = StrategyStore()
    if store.active() is not None:
        print(f"策略库已有 active={store.active()['strategy_id']}，不覆盖")
        return 0

    baseline = store.bootstrap({
        'classifier_prompt': '按类别分布选择指标；不平衡时用少数类 F1。',
        'planning_prompt': '先看缺失与唯一值，再选基线模型。',
        'candidate_selection_rules': [{'when': 'imbalanced', 'prefer': 'class_weight'}],
    }, reason='当前系统硬编码行为')

    good = store.propose(
        {'planning_prompt': '先查重：上一轮提过的方案不要再提；再看事后字段与 ID 列。'},
        reason='失败证据显示重复提案集中在同一模型族，规划里没有查重步骤',
        evidence_run_ids=['run_stuck'],
        hypothesis={'what': '规划里加查重与排雷', 'why': '重复提案都发生在没有查重的轮次',
                    'expected_improvement': '重复率下降，合法输出率不降',
                    'falsifier': '独立任务上重复率不降反升',
                    'side_effects': '单次运行多一次扫描，耗时略增'})

    bad = store.propose(
        {'planning_prompt': '随便再试一次就好。'},
        reason='一个不解决重复问题的候选（用来展示拒绝路径）',
        evidence_run_ids=['run_stuck'],
        hypothesis={'what': '换个说法', 'why': '没想清楚', 'expected_improvement': '未知',
                    'falsifier': '重复率不降', 'side_effects': '无'})

    criteria = freeze_criteria()
    print(f'条件冻结 digest={criteria["digest"]}')

    old_arm = run_arm(store, baseline['strategy_id'], tasks, repeating,
                      WORK / 'runs', max_experiments=3, label='seed_old')
    good_arm = run_arm(store, good['strategy_id'], tasks, varying,
                       WORK / 'runs', max_experiments=3, label='seed_good')
    bad_arm = run_arm(store, bad['strategy_id'], tasks, repeating,
                      WORK / 'runs', max_experiments=3, label='seed_bad')

    report_good = compare(old_arm, good_arm, criteria)
    report_bad = compare(old_arm, bad_arm, criteria)
    print(f'好候选：{report_good["verdict"]}；坏候选：{report_bad["verdict"]}')

    print('真实 Jev（没 key）会怎样：')
    blocked = decide(store, good['strategy_id'], report_good, None)
    print(f'  {blocked["decision"]} / {blocked["source"]}')

    upgraded = decide(store, good['strategy_id'], report_good, simulated_jev('upgrade'),
                      by='seed-demo-simulated')
    print(f'模拟 Jev：applied={upgraded["applied"]} → active={store.active()["strategy_id"]}')

    decide(store, bad['strategy_id'], report_bad, simulated_jev('upgrade'), by='seed-demo')

    print(f'\n最终 active={store.active()["strategy_id"]}，版本 {len(store.versions())} 个')
    for event in store.events():
        print(f"  {event['type']:15} {event.get('strategy_id','')}  "
              f"{event.get('source') or event.get('by') or ''}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
