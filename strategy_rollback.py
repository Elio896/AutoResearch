"""上线后的回滚：条件事先写死，触发就自动切回，失败案例一条不删。

规则在**上线前**冻结（带 digest），上线后只负责比较实测值，不负责解释。
触发就回滚、并留下证据；没触发就保持——不许"看情况"。

未知指标绝不当作正常：算不出就是 `insufficient_evidence`，保持现状并写明原因。
"""
from __future__ import annotations

import hashlib
import json
import time

DEFAULT_RULES: dict = {
    'min_tasks': 1,                 # 至少几个任务的数据才判
    'legal_output_rate_drop': 0.20,  # 合法输出率掉超过 20 个百分点 → 回滚
    'duplicate_rate_rise': 0.30,     # 重复率涨超过 30 个百分点 → 回滚
    'mean_best_score_drop': 0.05,    # 平均最佳分数掉超过 0.05 → 回滚
    'budget_violations': 0,          # 硬预算/执行约束违规次数上限（超过就回滚）
}


def freeze_rules(rules: dict | None = None) -> dict:
    frozen = json.loads(json.dumps(rules or DEFAULT_RULES))
    body = json.dumps(frozen, sort_keys=True, ensure_ascii=False)
    return {'rules': frozen, 'frozen_at': time.time(),
            'digest': hashlib.sha256(body.encode('utf-8')).hexdigest()[:16]}


def _delta(a, b):
    if a is None or b is None:
        return None
    return round(b - a, 4)


def evaluate(baseline: dict, current: dict, rules: dict) -> dict:
    """按冻结的规则判要不要回滚。返回每条规则的实际值和是否触发。"""
    checks: list[dict] = []
    triggers: list[str] = []

    def add(name: str, value, threshold, fired: bool | None, note: str):
        checks.append({'rule': name, 'value': value, 'threshold': threshold,
                       'fired': fired, 'note': note})
        if fired:
            triggers.append(name)

    enough = min(baseline.get('task_count') or 0, current.get('task_count') or 0) >= \
        int(rules['min_tasks'])
    checks.append({'rule': 'min_tasks', 'value': current.get('task_count'),
                   'threshold': rules['min_tasks'], 'fired': not enough and None,
                   'note': '样本是否够判'})

    drop = _delta(baseline.get('legal_output_rate'), current.get('legal_output_rate'))
    if drop is None:
        add('legal_output_rate_drop', None, rules['legal_output_rate_drop'], None, '算不出')
    else:
        add('legal_output_rate_drop', round(-drop, 4), rules['legal_output_rate_drop'],
            (-drop) >= rules['legal_output_rate_drop'], '合法输出率下降幅度')

    rise = _delta(baseline.get('duplicate_rate'), current.get('duplicate_rate'))
    if rise is None:
        add('duplicate_rate_rise', None, rules['duplicate_rate_rise'], None, '算不出')
    else:
        add('duplicate_rate_rise', rise, rules['duplicate_rate_rise'],
            rise >= rules['duplicate_rate_rise'], '重复提案率上升幅度')

    score_drop = _delta(baseline.get('mean_best_score'), current.get('mean_best_score'))
    if score_drop is None:
        add('mean_best_score_drop', None, rules['mean_best_score_drop'], None, '算不出')
    else:
        add('mean_best_score_drop', round(-score_drop, 4), rules['mean_best_score_drop'],
            (-score_drop) >= rules['mean_best_score_drop'], '平均最佳分数下降幅度')

    violations = int(current.get('budget_violations') or 0)
    add('budget_violations', violations, rules['budget_violations'],
        violations > int(rules['budget_violations']), '硬预算/执行约束违规次数')

    unknown = [check['rule'] for check in checks if check['fired'] is None
               and check['rule'] != 'min_tasks']
    if triggers:
        verdict, reason = 'rollback', f'触发：{triggers}'
    elif not enough or unknown:
        verdict, reason = 'insufficient_evidence', (
            f'数据不足或算不出：样本够={enough}，算不出={unknown}')
    else:
        verdict, reason = 'hold', '没有触发任何回滚条件'
    return {'verdict': verdict, 'reason': reason, 'triggers': triggers,
            'checks': checks, 'evaluated_at': time.time()}


def rollback_if_needed(store, *, current_id: str, fallback_id: str,
                       baseline: dict, current: dict, rules: dict,
                       by: str = 'auto-rollback') -> dict:
    """判 → 触发就切回旧版本，并把证据写进事件。"""
    report = evaluate(baseline, current, rules)
    report.update({'current_id': current_id, 'fallback_id': fallback_id,
                   'rules': rules})

    active = store.active()
    if active is None or active['strategy_id'] != current_id:
        # 要监控的那版根本不在线上（或者已经被换掉了），就不要瞎切
        return {**report, 'applied': False, 'verdict': 'hold',
                'reason': f'当前正式策略是 {active["strategy_id"] if active else None}，'
                          f'不是要监控的 {current_id}'}

    if report['verdict'] != 'rollback':
        return {**report, 'applied': False}

    version = store.rollback(fallback_id, triggers=report['triggers'],
                             evidence=report, by=by,
                             reason='；'.join(report['triggers'])[:200])
    return {**report, 'applied': True, 'strategy_id': version['strategy_id']}
