"""第六步验收：Jev 给选择、代码给约束，两关都过才切换，切换后新运行真的用新版。

真实 Jev 没 key 就如实报"不可用、保持旧策略"；模拟决定只用来验证流程，会标明。
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import jev_client  # noqa: E402
from agents import Budget  # noqa: E402
from executor import ResearchSession  # noqa: E402
from research_loop import ResearchLoop  # noqa: E402
from strategy_decision import decide  # noqa: E402
from strategy_eval import compare, freeze_criteria, run_arm  # noqa: E402
from strategy_store import StrategyStore  # noqa: E402

WORK = Path('artifacts') / 'strategy-upgrade-acceptance'


def make_task(path: Path, seed: int, n: int = 220) -> None:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.integers(0, 3, n)
    y = (1 / (1 + np.exp(-(1.0 * a - 0.7 * b + 0.5 * c))) > rng.random(n)).astype(int)
    pd.DataFrame({'a': a, 'b': b, 'c': c, 'target': y}).to_csv(path, index=False)


def stuck_factory(changes: dict):
    def proposer(messages, timeout=None):
        return {'message': {'content': json.dumps({
            'model': 'logistic_regression', 'parameters': {'max_iter': 500},
            'hypothesis': '同一套再试一次', 'expected_direction': 'up', 'falsifier': '分数没变'})},
            'prompt_eval_count': 0, 'eval_count': 0}
    return proposer


def obedient_factory(changes: dict):
    prompt = str(changes.get('planning_prompt') or '')
    obeys = '查重' in prompt
    state = {'i': 0}

    def proposer(messages, timeout=None):
        if obeys:
            state['i'] += 1
            return {'message': {'content': json.dumps({
                'model': 'random_forest', 'parameters': {'n_estimators': 40 + 20 * state['i']},
                'hypothesis': f'第 {state["i"]} 个不同方案',
                'expected_direction': 'up', 'falsifier': '分数没变'})},
                'prompt_eval_count': 0, 'eval_count': 0}
        return stuck_factory(changes)(messages, timeout)
    return proposer


def real_jev(state: str, questions: dict) -> dict:
    """真调 Jev（没 key 会返回 ok=False，不抛异常）。"""
    return jev_client.ask_jev(state, questions, timeout=20)


def simulated_jev(answer: str):
    calls = []

    def _gate(state, questions):
        calls.append(1)
        return {'ok': True, 'model': 'simulated-jev', 'answers': {'strategy_decision': answer}}

    _gate.calls = calls
    return _gate


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)
    eval_a, eval_b = WORK / 'eval_a.csv', WORK / 'eval_b.csv'
    make_task(eval_a, 2026)
    make_task(eval_b, 77)

    store = StrategyStore(WORK / 'strategy_store.json')
    old = store.bootstrap({
        'classifier_prompt': '按类别分布选择指标。',
        'planning_prompt': '先看缺失与唯一值，再选基线模型。',
        'candidate_selection_rules': [],
    }, reason='基线')
    good = store.propose({'planning_prompt': '先查重：上一轮提过的方案不要再提，换模型族。'},
                         reason='重复提案率高', evidence_run_ids=['run_dev'],
                         hypothesis={'what': '规划加查重', 'why': '重复提案集中在同一族',
                                     'expected_improvement': '重复率下降',
                                     'falsifier': '重复率不降', 'side_effects': '耗时略增'})
    bad = store.propose({'planning_prompt': '随便再试一次就好。'},
                        reason='不解决重复问题的候选', evidence_run_ids=['run_dev'])

    criteria = freeze_criteria()
    print(f"升级条件（冻结）：digest={criteria['digest']}")

    print('\n== 1. 对照评估 ==')
    arm_old = run_arm(store, old['strategy_id'], [eval_a, eval_b], stuck_factory,
                      WORK / 'runs', max_experiments=4, label='old')
    arm_good = run_arm(store, good['strategy_id'], [eval_a, eval_b], obedient_factory,
                       WORK / 'runs', max_experiments=4, label='good')
    report = compare(arm_old, arm_good, criteria)
    print(f"  旧：重复率={arm_old['duplicate_rate']}  新：重复率={arm_good['duplicate_rate']}")
    print(f"  结论：{report['verdict']}（{report['reason']}）")

    print('\n== 2. 拿真实 Jev 判 ==')
    real = decide(store, good['strategy_id'], report, real_jev)
    print(f"  decision={real['decision']}  applied={real['applied']}  source={real['source']}")
    print(f"  原因：{real['reason']}")
    print(f"  正式策略：{store.active()['strategy_id']}（没被切）")

    print('\n== 3. 模拟 Jev 选 upgrade（仅验证流程）==')
    sim = simulated_jev('upgrade')
    decided = decide(store, good['strategy_id'], report, sim)
    print(f"  decision={decided['decision']}  applied={decided['applied']}  source={decided['source']}")
    print(f"  正式策略变成：{store.active()['strategy_id']}")
    print(f"  Jev 收到的选项：{['upgrade', 'keep', 'defer']}")

    print('\n== 4. 切换后，下一次研究真的用新版 ==')
    session = ResearchSession(eval_a, 'target', metric='f1_macro', pos_label='1')
    loop = ResearchLoop(session, proposer=obedient_factory(store.active()['changes']),
                        budget=Budget(max_experiments=1, max_seconds=60.0, max_local_calls=3),
                        strategy_store=store, run_id='after_upgrade',
                        trajectory_dir=WORK / 'after_upgrade')
    outcome = loop.run()
    print(f"  这次运行 pin 到：{outcome['summary']['strategy_id']}"
          f"（库里 active={store.active()['strategy_id']}）")

    print('\n== 5. 没通过评估的候选，代码先拦，不问 Jev ==')
    arm_bad = run_arm(store, bad['strategy_id'], [eval_a, eval_b], obedient_factory,
                      WORK / 'runs', max_experiments=4, label='bad')
    report_bad = compare(arm_old, arm_bad, criteria)
    spy = simulated_jev('upgrade')
    blocked = decide(store, bad['strategy_id'], report_bad, spy)
    print(f"  评估结论：{report_bad['verdict']}  decision={blocked['decision']} "
          f"source={blocked['source']}  Jev 被调用次数={len(spy.calls)}")

    print('\n== 6. 切换留痕 ==')
    for event in store.events():
        if event['type'] in ('activate', 'decision_keep', 'decision_defer'):
            detail = event.get('reason') or event.get('previous_id')
            print(f"  {event['type']}  strategy={event['strategy_id']}  {detail}")

    ok = (real['applied'] is False and real['source'] == 'jev_unavailable'
          and decided['applied'] is True
          and outcome['summary']['strategy_id'] == decided['strategy_id']
          and blocked['applied'] is False and len(spy.calls) == 0)
    print('\n判定：' + ('通过——真 Jev 不可用时保持旧策略；两关都过才切换；切换后新运行真的用新版；'
                       '未过评估的候选连 Jev 都不会被问' if ok else '失败'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
