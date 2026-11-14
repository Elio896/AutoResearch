"""新旧策略对照评估：两套隔离，条件先写死，没有结果不许升级。

**两层隔离**（别混）：

1. 机器学习层：训练 / 验证 / 测试——由 `ResearchSession` 保证，测试集只在最终评估用一次；
2. 策略评估层：**开发任务**（用来发现失败、提出候选）与**评估任务**（用来判断能不能升级）。
   候选策略不能靠"产生它的那几个任务"验收，评估任务必须另给。

另外两条硬规矩：

- 升级条件在跑之前写定并冻结（带时间戳与摘要），跑完不许改；
- 样本不够、指标算不出来时，结论是 `insufficient_evidence`，不是"通过"。
- 未知账单不记成零成本：本地调用记 `local_no_charge`，云端调用没对账单时记 `unknown`。
"""
from __future__ import annotations

import hashlib
import json
import time
import math
from pathlib import Path

import failure_modes
from agents import Budget
from executor import ResearchSession
from research_loop import ResearchLoop
from strategy_store import StrategyStore

SCHEMA = 1

DEFAULT_CRITERIA: dict = {
    'min_tasks': 2,                      # 至少几个评估任务才算数
    'legal_output_rate': {'min_delta': 0.0},     # 合法输出率不得下降
    'duplicate_rate': {'max_delta': 0.0},        # 重复提案率不得上升
    'mean_best_score': {'min_delta': -0.02},     # 分数允许轻微波动，掉超过 0.02 不行
    'calls': {'max_ratio': 1.5},                 # 调用量不能超过旧策略的 1.5 倍
    # 光不退化不算升级：至少得有一项超过这个幅度，否则就是白改
    'min_improvement': {'legal_output_rate': 0.0, 'duplicate_rate': 0.0,
                        'mean_best_score': 0.0},
}


def freeze_criteria(criteria: dict | None = None) -> dict:
    """把条件在跑之前钉死。跑完再改就等于作弊，这里留摘要给人查。"""
    frozen = json.loads(json.dumps(criteria or DEFAULT_CRITERIA))
    body = json.dumps(frozen, sort_keys=True, ensure_ascii=False)
    return {
        'schema': SCHEMA,
        'criteria': frozen,
        'frozen_at': time.time(),
        'digest': hashlib.sha256(body.encode('utf-8')).hexdigest()[:16],
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def run_arm(store: StrategyStore, strategy_id: str, tasks: list[Path],
            proposer_factory, workdir: Path, target: str = 'target',
            metric: str = 'f1_macro', pos_label: str = '1',
            max_experiments: int = 2, max_seconds: float = 180.0,
            label: str = 'arm') -> dict:
    """用某一版策略，在给定的评估任务上真跑一遍，收集指标。"""
    version = store.get(strategy_id)
    if version is None:
        raise KeyError(f'策略不存在：{strategy_id}')
    workdir.mkdir(parents=True, exist_ok=True)

    per_task: list[dict] = []
    for index, csv_path in enumerate(tasks):
        session = ResearchSession(Path(csv_path), target, metric=metric, pos_label=pos_label)
        proposer = proposer_factory(version['changes'])
        loop = ResearchLoop(
            session, proposer=proposer,
            budget=Budget(max_experiments=max_experiments, max_seconds=max_seconds,
                          max_local_calls=max_experiments + 4),
            strategy_store=store, strategy_id=strategy_id,
            run_id=f'{label}_task{index}',
            trajectory_dir=workdir / f'{label}_task{index}',
        )
        outcome = loop.run()
        events = outcome['events']
        summary = outcome['summary']
        proposals = [e for e in events if e['event'] in ('scientist_proposal', 'cloud_proposal')]
        invalid = [e for e in events if e['event'] in ('scientist_error', 'scientist_rejected', 'proposal_invalid')]
        attempts = len(proposals) + len(invalid)
        detected = failure_modes.detect(events, run_id=summary.get('trajectory_run_id', ''),
                                        strategy_id=strategy_id)
        duplicates = sum(max(0, record['measurements'].get('duplicate_count', 1) - 1)
                         for record in detected['records']
                         if record['failure_type'] == 'repeated_proposal')

        per_task.append({
            'task': Path(csv_path).name,
            'status': summary.get('status'),
            'best_score': summary.get('best_score'),
            'experiments': summary.get('experiments'),
            'proposals': attempts,
            'invalid_outputs': len(invalid),
            'legal_output_rate': _ratio(len(proposals), attempts),
            'duplicate_proposals': duplicates,
            'duplicate_rate': _ratio(duplicates, attempts),
            'local_calls': summary.get('local_calls', 0),
            'cloud_calls': summary.get('cloud_calls', 0),
            'elapsed_seconds': summary.get('elapsed_seconds'),
            'strategy_pin_intact': summary.get('strategy_pin_intact'),
            'insufficient': detected['insufficient'],
        })

    completed = [task for task in per_task if task['status'] == 'completed']
    scores = [task['best_score'] for task in completed if task['best_score'] is not None]
    proposals_total = sum(task['proposals'] for task in per_task)
    invalid_total = sum(task['invalid_outputs'] for task in per_task)
    duplicates_total = sum(task['duplicate_proposals'] for task in per_task)
    cloud_calls = sum(task['cloud_calls'] for task in per_task)

    return {
        'schema': SCHEMA,
        'label': label,
        'strategy_id': strategy_id,
        'strategy_digest': version['config_digest'],
        'task_count': len(per_task),
        'completed': len(completed),
        'completion_rate': _ratio(len(completed), len(per_task)),
        'mean_best_score': round(sum(scores) / len(scores), 4) if scores else None,
        'proposals': proposals_total,
        'invalid_outputs': invalid_total,
        'legal_output_rate': _ratio(proposals_total - invalid_total, proposals_total),
        'duplicate_proposals': duplicates_total,
        'duplicate_rate': _ratio(duplicates_total, proposals_total),
        'local_calls': sum(task['local_calls'] for task in per_task),
        'cloud_calls': cloud_calls,
        'cost_basis': 'local_no_charge' if cloud_calls == 0 else 'unknown',
        'cost_cny': 0.0 if cloud_calls == 0 else None,
        'tasks': per_task,
    }


def _check(name: str, passed: bool | None, detail: dict) -> dict:
    return {'check': name, 'passed': passed, 'detail': detail}


def compare(old: dict, new: dict, criteria: dict) -> dict:
    """按冻结的条件判：能不能升。样本不够或指标缺失一律 insufficient。"""
    rules = criteria['criteria']
    checks: list[dict] = []
    digest = hashlib.sha256(json.dumps(rules, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:16]
    checks.append(_check('criteria_integrity', digest == criteria.get('digest'), {}))
    checks.append(_check('completion', old.get('completed') == old['task_count'] and
                         new.get('completed') == new['task_count'],
                         {'old': old.get('completed'), 'new': new.get('completed')}))
    if 'task_ids' in old or 'task_ids' in new:
        checks.append(_check('matched_tasks', old.get('task_ids') == new.get('task_ids'), {}))

    enough = min(old['task_count'], new['task_count']) >= int(rules['min_tasks'])
    checks.append(_check('min_tasks', enough,
                         {'old': old['task_count'], 'new': new['task_count'],
                          'required': rules['min_tasks']}))

    missing = [name for name in ('legal_output_rate', 'duplicate_rate', 'mean_best_score')
               if any(not isinstance(arm.get(name), (int, float)) or
                      not math.isfinite(arm.get(name, float('nan'))) for arm in (old, new))]
    if missing:
        # NaN/inf are unknown evidence, never a passing comparison.
        old, new = dict(old), dict(new)
        for name in missing:
            if not isinstance(old.get(name), (int, float)) or not math.isfinite(old[name]):
                old[name] = None
            if not isinstance(new.get(name), (int, float)) or not math.isfinite(new[name]):
                new[name] = None
    checks.append(_check('metrics_available', not missing, {'missing': missing}))

    if old.get('legal_output_rate') is not None and new.get('legal_output_rate') is not None:
        delta = new['legal_output_rate'] - old['legal_output_rate']
        checks.append(_check('legal_output_rate',
                             delta >= rules['legal_output_rate']['min_delta'],
                             {'old': old['legal_output_rate'], 'new': new['legal_output_rate'],
                              'delta': round(delta, 4),
                              'min_delta': rules['legal_output_rate']['min_delta']}))
    else:
        checks.append(_check('legal_output_rate', None, {'reason': '算不出来'}))

    if old.get('duplicate_rate') is not None and new.get('duplicate_rate') is not None:
        delta = new['duplicate_rate'] - old['duplicate_rate']
        checks.append(_check('duplicate_rate',
                             delta <= rules['duplicate_rate']['max_delta'],
                             {'old': old['duplicate_rate'], 'new': new['duplicate_rate'],
                              'delta': round(delta, 4),
                              'max_delta': rules['duplicate_rate']['max_delta']}))
    else:
        checks.append(_check('duplicate_rate', None, {'reason': '算不出来'}))

    if old.get('mean_best_score') is not None and new.get('mean_best_score') is not None:
        delta = new['mean_best_score'] - old['mean_best_score']
        checks.append(_check('mean_best_score',
                             delta >= rules['mean_best_score']['min_delta'],
                             {'old': old['mean_best_score'], 'new': new['mean_best_score'],
                              'delta': round(delta, 4),
                              'min_delta': rules['mean_best_score']['min_delta']}))
    else:
        checks.append(_check('mean_best_score', None, {'reason': '算不出来'}))

    old_calls = (old.get('local_calls') or 0) + (old.get('cloud_calls') or 0)
    new_calls = (new.get('local_calls') or 0) + (new.get('cloud_calls') or 0)
    ratio = new_calls / old_calls if old_calls else (0 if new_calls == 0 else float('inf'))
    checks.append(_check('calls', ratio <= rules['calls']['max_ratio'],
                         {'old': old.get('local_calls'), 'new': new.get('local_calls'),
                          'ratio': round(ratio, 3) if math.isfinite(ratio) else None,
                          'max_ratio': rules['calls']['max_ratio']}))
    for metric in ('model_tokens', 'elapsed_seconds'):
        if metric in old or metric in new:
            a, b = old.get(metric), new.get(metric)
            limit = rules.get(metric, {}).get('max_ratio', 1.5 if metric == 'model_tokens' else 2.0)
            available = a is not None and b is not None and math.isfinite(a) and math.isfinite(b)
            floor = 1.0 if metric == 'elapsed_seconds' else 0
            checks.append(_check(metric, b <= max(floor, a) * limit if available else None,
                                 {'old': a, 'new': b, 'max_ratio': limit}))

    # 只"不比旧策略差"不够——必须有一项可测的改善，否则这次修改就是白改
    thresholds = rules.get('min_improvement') or {}
    improved: list[str] = []
    if old.get('legal_output_rate') is not None and new.get('legal_output_rate') is not None:
        if new['legal_output_rate'] - old['legal_output_rate'] > thresholds.get('legal_output_rate', 0.0):
            improved.append('legal_output_rate')
    if old.get('duplicate_rate') is not None and new.get('duplicate_rate') is not None:
        if old['duplicate_rate'] - new['duplicate_rate'] > thresholds.get('duplicate_rate', 0.0):
            improved.append('duplicate_rate')
    if old.get('mean_best_score') is not None and new.get('mean_best_score') is not None:
        if new['mean_best_score'] - old['mean_best_score'] > thresholds.get('mean_best_score', 0.0):
            improved.append('mean_best_score')
    if (old.get('model_tokens') is not None and new.get('model_tokens') is not None and
            old['model_tokens'] - new['model_tokens'] > thresholds.get('model_tokens', 0)):
        improved.append('model_tokens')
    checks.append(_check('demonstrated_improvement',
                         None if missing else bool(improved),
                         {'improved': improved, 'thresholds': thresholds}))

    unknown = [check['check'] for check in checks if check['passed'] is None]
    failed = [check['check'] for check in checks if check['passed'] is False]
    if unknown or not enough or missing:
        verdict, reason = 'insufficient_evidence', (
            f'不满足判断条件：缺指标 {missing or "-"}；样本不足 {not enough}；算不出 {unknown or "-"}')
    elif failed:
        verdict, reason = 'keep', f'未通过：{failed}'
    else:
        verdict, reason = 'upgrade', '全部条件通过'

    return {
        'schema': SCHEMA,
        'criteria': criteria,
        'old': {k: v for k, v in old.items() if k != 'tasks'},
        'new': {k: v for k, v in new.items() if k != 'tasks'},
        'checks': checks,
        'failed_checks': failed,
        'unknown_checks': unknown,
        'verdict': verdict,
        'reason': reason,
        'evaluated_at': time.time(),
    }
