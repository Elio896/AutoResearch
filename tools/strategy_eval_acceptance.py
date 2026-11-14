"""第五步验收：新旧策略对照评估（开发任务 / 评估任务分开，条件跑前冻结）。

注意口径：这里的**提案器是确定性脚本**，用来看清评估与判定流程是否可靠；
它不代表模型能力，也不能算"真实自进化效果"。真实模型那一层在第八步的界面里
会标 `simulated`。脚本会把这件事写在输出里，不藏。
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from strategy_eval import compare, freeze_criteria, run_arm  # noqa: E402
from strategy_store import StrategyStore  # noqa: E402

WORK = Path('artifacts') / 'strategy-eval-acceptance'


def make_task(path: Path, seed: int, n: int = 240) -> None:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.integers(0, 3, n)
    logit = 1.0 * a - 0.7 * b + 0.5 * c
    y = (1 / (1 + np.exp(-logit)) > rng.random(n)).astype(int)
    pd.DataFrame({'a': a, 'b': b, 'c': c, 'target': y}).to_csv(path, index=False)


def stuck_factory(changes: dict):
    """不读提示词，永远提同一个方案。"""
    def proposer(messages, timeout=None):
        return {'message': {'content': json.dumps({
            'model': 'logistic_regression', 'parameters': {'max_iter': 500},
            'hypothesis': '同一套再试一次', 'expected_direction': 'up', 'falsifier': '分数没变'})},
            'prompt_eval_count': 0, 'eval_count': 0}
    return proposer


def obedient_factory(changes: dict):
    """读提示词：里面写了"查重"就不重复提，否则和 stuck 一样。

    这是脚本化的"模型行为"，只为验证评估流程；不是真模型。
    """
    prompt = str(changes.get('planning_prompt') or '')
    obeys = '查重' in prompt
    state = {'i': 0}

    def proposer(messages, timeout=None):
        if obeys:
            state['i'] += 1
            return {'message': {'content': json.dumps({
                'model': 'random_forest', 'parameters': {'n_estimators': 40 + 20 * state['i']},
                'hypothesis': f'第 {state["i"]} 个不同方案，避免重复',
                'expected_direction': 'up', 'falsifier': '分数没变'})},
                'prompt_eval_count': 0, 'eval_count': 0}
        return {'message': {'content': json.dumps({
            'model': 'logistic_regression', 'parameters': {'max_iter': 500},
            'hypothesis': '同一套再试一次', 'expected_direction': 'up', 'falsifier': '分数没变'})},
            'prompt_eval_count': 0, 'eval_count': 0}
    return proposer


def show_arm(arm: dict) -> None:
    print(f"  {arm['label']}  ({arm['strategy_id']} digest={arm['strategy_digest']})")
    print(f"    任务 {arm['completed']}/{arm['task_count']} 完成  "
          f"mean_best={arm['mean_best_score']}")
    print(f"    合法输出率={arm['legal_output_rate']}  "
          f"重复率={arm['duplicate_rate']}  （提案 {arm['proposals']}，"
          f"不合法 {arm['invalid_outputs']}，重复 {arm['duplicate_proposals']}）")
    print(f"    本地调用={arm['local_calls']}  云端调用={arm['cloud_calls']}  "
          f"成本={arm['cost_cny']}（{arm['cost_basis']}）")


def show_report(report: dict) -> None:
    print(f"  verdict={report['verdict']}   reason={report['reason']}")
    for check in report['checks']:
        flag = {True: '通过', False: '未通过', None: '算不出'}[check['passed']]
        print(f"    {check['check']}: {flag}  {json.dumps(check['detail'], ensure_ascii=False)}")


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)

    dev = WORK / 'dev_task.csv'
    eval_a = WORK / 'eval_task_a.csv'
    eval_b = WORK / 'eval_task_b.csv'
    make_task(dev, seed=11)
    make_task(eval_a, seed=2026)
    make_task(eval_b, seed=77)
    print(f'开发任务：{dev.name}（用来发现失败，不参与判定）')
    print(f'评估任务：{eval_a.name}、{eval_b.name}（判定只看这两个）\n')

    store = StrategyStore(WORK / 'strategy_store.json')
    old = store.bootstrap({
        'classifier_prompt': '按类别分布选择指标。',
        'planning_prompt': '先看缺失与唯一值，再选基线模型。',
        'candidate_selection_rules': [],
    }, reason='基线策略')

    good = store.propose({'planning_prompt': '先查重：上一轮提过的方案不要再提，换模型族。'},
                         reason='重复提案率高', evidence_run_ids=['run_dev'])
    bad = store.propose({'planning_prompt': '随便再试一次就好。'},
                        reason='一个不解决问题的候选（用来验证拒绝路径）',
                        evidence_run_ids=['run_dev'])

    criteria = freeze_criteria()
    print(f"升级条件（跑前冻结）：digest={criteria['digest']}")
    print(f"  条件：{json.dumps(criteria['criteria'], ensure_ascii=False)}\n")

    tasks = [eval_a, eval_b]
    print('== 策略 A：旧策略（基线） ==')
    arm_old = run_arm(store, old['strategy_id'], tasks, stuck_factory, WORK / 'runs',
                      max_experiments=4, label='old')
    show_arm(arm_old)

    print('\n== 策略 B：候选（规划里加了查重） ==')
    arm_good = run_arm(store, good['strategy_id'], tasks, obedient_factory, WORK / 'runs',
                       max_experiments=4, label='good')
    show_arm(arm_good)

    print('\n== 对照 A vs B ==')
    report = compare(arm_old, arm_good, criteria)
    show_report(report)

    print('\n== 反例：候选 C 不解决重复问题 ==')
    arm_bad = run_arm(store, bad['strategy_id'], tasks, obedient_factory, WORK / 'runs',
                      max_experiments=4, label='bad')
    show_arm(arm_bad)
    print()
    report_bad = compare(arm_old, arm_bad, criteria)
    show_report(report_bad)

    print('\n== 正式策略有没有被这次评估改动？ ==')
    print(f"  active={store.active()['strategy_id']}（仍是 {old['strategy_id']}："
          f"{store.active()['strategy_id'] == old['strategy_id']}）")

    print('\n== 样本不足时 ==')
    thin = run_arm(store, old['strategy_id'], [eval_a], stuck_factory, WORK / 'runs',
                   max_experiments=1, label='thin')
    report_thin = compare(thin, thin, criteria)
    show_report(report_thin)

    ok = (report['verdict'] == 'upgrade'
          and report_bad['verdict'] == 'keep'
          and report_thin['verdict'] == 'insufficient_evidence'
          and store.active()['strategy_id'] == old['strategy_id'])
    (WORK / 'report.json').write_text(json.dumps(
        {'good': report, 'bad': report_bad, 'thin': report_thin,
         'arms': {'old': arm_old, 'good': arm_good, 'bad': arm_bad}},
        ensure_ascii=False, indent=2), encoding='utf-8')

    print('\n口径：提案器是确定性脚本（模拟模型行为），本步验证的是评估与判定流程；'
          '不代表模型能力。')
    print('判定：' + ('通过——好候选能升、坏候选被拒、样本不足不下结论、正式策略未被评估改动'
                     if ok else '失败'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
