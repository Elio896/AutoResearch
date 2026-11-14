"""第七步验收：完整走一遍 旧策略 → 候选 → 评估通过 → 启用 → 触发回滚 → 恢复旧版。

回滚规则在**上线前**冻结；上线后只看实测值。提案器是脚本（模拟模型行为），流程本身是真的。
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agents import Budget  # noqa: E402
from executor import ResearchSession  # noqa: E402
from research_loop import ResearchLoop  # noqa: E402
from strategy_decision import decide  # noqa: E402
from strategy_eval import compare, freeze_criteria, run_arm  # noqa: E402
from strategy_rollback import freeze_rules, rollback_if_needed  # noqa: E402
from strategy_store import StrategyStore  # noqa: E402

WORK = Path('artifacts') / 'strategy-rollback-acceptance'


def make_task(path: Path, seed: int, n: int = 200) -> None:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.integers(0, 3, n)
    y = (1 / (1 + np.exp(-(1.0 * a - 0.6 * b + 0.4 * c))) > rng.random(n)).astype(int)
    pd.DataFrame({'a': a, 'b': b, 'c': c, 'target': y}).to_csv(path, index=False)


def varying_factory(changes: dict):
    state = {'i': 0}

    def proposer(messages, timeout=None):
        state['i'] += 1
        return {'message': {'content': json.dumps({
            'model': 'random_forest', 'parameters': {'n_estimators': 30 + 15 * state['i']},
            'hypothesis': f'第 {state["i"]} 个方案', 'expected_direction': 'up',
            'falsifier': '分数没变'})}, 'prompt_eval_count': 0, 'eval_count': 0}
    return proposer


def repeating_factory(changes: dict):
    def proposer(messages, timeout=None):
        return {'message': {'content': json.dumps({
            'model': 'logistic_regression', 'parameters': {'max_iter': 500},
            'hypothesis': '还是同一套', 'expected_direction': 'up',
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
    post = [WORK / 'post_c.csv', WORK / 'post_d.csv']
    for path, seed in zip(tasks + post, (2026, 77, 5, 19)):
        make_task(path, seed)

    store = StrategyStore(WORK / 'strategy_store.json')
    old = store.bootstrap({
        'classifier_prompt': '按类别分布选择指标。',
        'planning_prompt': '先看缺失与唯一值。',
        'candidate_selection_rules': [],
    }, reason='基线')
    candidate = store.propose({'planning_prompt': '先查重：上一轮提过的方案不要再提。'},
                              reason='重复提案率高', evidence_run_ids=['run_dev'],
                              hypothesis={'what': '规划加查重', 'why': '重复集中在同一族',
                                          'expected_improvement': '重复率下降',
                                          'falsifier': '重复率不降', 'side_effects': '耗时略增'})

    criteria = freeze_criteria()
    rules = freeze_rules()
    print(f"升级条件冻结 digest={criteria['digest']}；回滚规则冻结 digest={rules['digest']}")
    print(f"回滚规则：{json.dumps(rules['rules'], ensure_ascii=False)}")

    print('\n== 1. 评估：候选能不能升 ==')
    arm_old = run_arm(store, old['strategy_id'], tasks, repeating_factory,
                      WORK / 'runs', max_experiments=4, label='old')
    arm_new = run_arm(store, candidate['strategy_id'], tasks, varying_factory,
                      WORK / 'runs', max_experiments=4, label='new')
    report = compare(arm_old, arm_new, criteria)
    print(f"  旧重复率={arm_old['duplicate_rate']}  新重复率={arm_new['duplicate_rate']}  "
          f"→ {report['verdict']}")

    print('\n== 2. 启用（模拟 Jev 选 upgrade，仅验流程）==')
    decided = decide(store, candidate['strategy_id'], report, simulated_jev('upgrade'))
    print(f"  applied={decided['applied']}  active={store.active()['strategy_id']}")

    print('\n== 3. 上线后监控：新策略在后续任务上退化了 ==')
    base_arm = run_arm(store, old['strategy_id'], post, varying_factory,
                       WORK / 'runs', max_experiments=4, label='monitor_old')
    live_arm = run_arm(store, candidate['strategy_id'], post, repeating_factory,
                       WORK / 'runs', max_experiments=4, label='monitor_new')
    print(f"  基线重复率={base_arm['duplicate_rate']}  上线后重复率={live_arm['duplicate_rate']}")

    print('\n== 4. 按冻结规则判 ==')
    outcome = rollback_if_needed(
        store, current_id=candidate['strategy_id'], fallback_id=old['strategy_id'],
        baseline=base_arm, current=live_arm, rules=rules['rules'])
    print(f"  verdict={outcome['verdict']}  triggers={outcome['triggers']}  "
          f"applied={outcome['applied']}")
    for check in outcome['checks']:
        print(f"    {check['rule']}: 实测={check['value']} 阈值={check['threshold']} "
              f"触发={check['fired']}（{check['note']}）")

    print('\n== 5. 后续运行恢复旧版本 ==')
    print(f"  active={store.active()['strategy_id']}")
    session = ResearchSession(post[0], 'target', metric='f1_macro', pos_label='1')
    loop = ResearchLoop(session, proposer=varying_factory(store.active()['changes']),
                        budget=Budget(max_experiments=1, max_seconds=60.0, max_local_calls=3),
                        strategy_store=store, run_id='after_rollback',
                        trajectory_dir=WORK / 'after_rollback')
    outcome_run = loop.run()
    print(f"  新开一次研究 pin 到：{outcome_run['summary']['strategy_id']}")

    print('\n== 6. 失败的那个版本还在吗 ==')
    failed = store.get(candidate['strategy_id'])
    print(f"  {failed['strategy_id']}  status={failed['status']}  "
          f"planning_prompt={failed['changes']['planning_prompt']!r}")

    print('\n== 7. 事件时间线 ==')
    for event in store.events():
        if event['type'] in ('activate', 'rollback', 'decision_keep', 'decision_defer'):
            print(f"  {event['type']:15} {event['strategy_id']}  "
                  f"{event.get('triggers') or event.get('reason', '')}")

    event = store.events()[-1]
    ok = (outcome['applied'] is True
          and store.active()['strategy_id'] == old['strategy_id']
          and outcome_run['summary']['strategy_id'] == old['strategy_id']
          and failed is not None and failed['status'] == 'retired'
          and event['type'] == 'rollback' and 'checks' in event['evidence'])
    print('\n判定：' + ('通过——上线→退化→按预定规则回滚→后续运行用回旧版，失败版本保留作证据'
                       if ok else '失败'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
