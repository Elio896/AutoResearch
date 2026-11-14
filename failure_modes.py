"""从真实轨迹里找失败模式——只认可以数出来的东西。

输入是一次运行的事件流（`research_loop` 的 `events`）和实验列表，
输出是失败记录，每条都能指回具体事件（`event_refs`，按事件流下标编成 event_007 这种）。

两条底线：

1. **说不清就不说**：证据不够时返回 `insufficient=True` 和原因，不产出一条"经验"。
2. **不解释动机**：这里只报"发生了什么、多少次、哪些事件"，为什么这样、该怎么改，
   是下一步（DeepSeek）的活，不在这里替它写。
"""
from __future__ import annotations

import json
import time

SCHEMA = 1

FAILURE_TYPES = {
    'repeated_proposal': '同一个方案被重复提出',
    'invalid_structure_repeat': '连续输出不合法结构',
    'same_family_tuning_no_gain': '同一模型族反复调参无改善',
    'repeated_change_no_gain': '同类改动持续无收益',
    'operation_check_failed': '基础操作未通过验证',
}

MIN_EXPERIMENTS = 2


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def event_refs(events: list[dict], indices: list[int]) -> list[str]:
    """事件流下标 → 可追溯的引用。下标就是这条事件在本 run 里的位置。"""
    return [f'event_{index:03d}' for index in indices]


def _family(model: str | None) -> str:
    name = (model or '').lower()
    if not name:
        return 'unknown'
    if 'logistic' in name or 'svm' in name or 'ridge' in name:
        return 'linear'
    if 'forest' in name or 'boost' in name or 'tree' in name:
        return 'tree'
    return name.split('_')[0]


def _proposals(events: list[dict]) -> list[tuple[int, dict]]:
    return [(index, event.get('proposal') or {})
            for index, event in enumerate(events)
            if event.get('event') in ('scientist_proposal', 'cloud_proposal')]


def _changes_between(previous: dict | None, current: dict) -> list[str]:
    if previous is None:
        return ['__first__']
    keys = []
    if previous.get('model') != current.get('model'):
        keys.append('model')
    if canonical(previous.get('parameters') or {}) != canonical(current.get('parameters') or {}):
        keys.append('parameters')
    return keys


def _detect_repeated_proposal(events: list[dict]) -> list[dict]:
    groups: dict[str, list[int]] = {}
    payloads: dict[str, dict] = {}
    for index, proposal in _proposals(events):
        key = canonical({'model': proposal.get('model'),
                         'parameters': proposal.get('parameters') or {}})
        groups.setdefault(key, []).append(index)
        payloads.setdefault(key, proposal)

    records = []
    for key, indices in groups.items():
        if len(indices) < 2:
            continue
        proposal = payloads[key]
        records.append({
            'failure_type': 'repeated_proposal',
            'event_refs': event_refs(events, indices),
            'measurements': {'duplicate_count': len(indices),
                             'model': proposal.get('model'),
                             'parameters': proposal.get('parameters') or {}},
        })
    return records


def _detect_invalid_structure(events: list[dict]) -> list[dict]:
    indices = [index for index, event in enumerate(events)
               if event.get('event') in ('scientist_error', 'proposal_invalid')]
    if len(indices) < 2:
        return []
    samples = []
    for index in indices:
        payload = events[index]
        samples.append({'event': payload.get('event'),
                        'error': str(payload.get('error') or payload.get('reason') or '')[:160]})
    return [{
        'failure_type': 'invalid_structure_repeat',
        'event_refs': event_refs(events, indices),
        'measurements': {'invalid_count': len(indices), 'samples': samples[:3]},
    }]


def _detect_same_family_no_gain(events: list[dict]) -> list[dict]:
    experiments = [(index, event) for index, event in enumerate(events)
                   if event.get('event') == 'experiment']
    if len(experiments) < 3:
        return []

    best = None
    gains: list[int] = []          # 记为"没有刷新最优"的实验下标
    for index, event in experiments:
        score = float(event.get('score') or 0.0)
        if best is None or score > best + 1e-12:
            best = score if best is None else max(best, score)
            gains = []
        else:
            gains.append(index)

    if len(gains) < 3:
        return []
    streak = gains[-3:]
    families = [_family(events[index].get('model')) for index in streak]
    if len(set(families)) != 1:
        return []
    return [{
        'failure_type': 'same_family_tuning_no_gain',
        'event_refs': event_refs(events, streak),
        'measurements': {'model_family': families[0], 'no_gain_streak': len(streak),
                         'best_score': round(best, 6)},
    }]


def _detect_repeated_change_no_gain(events: list[dict]) -> list[dict]:
    proposals = _proposals(events)
    if len(proposals) < 3:
        return []

    change_runs: dict[str, list[int]] = {}
    previous = None
    for position, (index, proposal) in enumerate(proposals):
        keys = _changes_between(previous, proposal)
        previous = proposal
        if keys == ['__first__']:
            continue
        change_runs.setdefault(canonical(sorted(keys)), []).append(position)

    repeated = {key: positions for key, positions in change_runs.items() if len(positions) >= 2}
    if not repeated:
        return []

    scores = [float(event.get('score') or 0.0) for index, event in enumerate(events)
              if event.get('event') == 'experiment']
    if len(scores) < 2:
        return []
    improved = any(scores[i] > scores[i - 1] + 1e-12 for i in range(1, len(scores)))
    if improved:
        return []

    key, positions = max(repeated.items(), key=lambda kv: len(kv[1]))
    indices = [proposals[position][0] for position in positions]
    return [{
        'failure_type': 'repeated_change_no_gain',
        'event_refs': event_refs(events, indices),
        'measurements': {'changed_fields': json.loads(key), 'repeat_count': len(positions),
                         'experiment_scores': [round(s, 6) for s in scores]},
    }]


def _detect_operation_check_failed(worklog_records: list[dict]) -> list[dict]:
    records = []
    for item in worklog_records:
        failed = [check for check in (item.get('check_results') or []) if not check.get('passed')]
        if not failed:
            continue
        records.append({
            'failure_type': 'operation_check_failed',
            'event_refs': [f"worklog:{item.get('run_id', '')}:#{item.get('logged_at', 0)}"],
            'measurements': {'failed_checks': [check.get('check') for check in failed],
                             'tool': (item.get('tool_computation') or {}).get('tool')},
        })
    return records


def detect(events: list[dict], run_id: str = '', strategy_id: str | None = None,
           worklog_records: list[dict] | None = None) -> dict:
    """跑一遍检测。证据不足时明确说不足，不硬凑失败记录。"""
    events = list(events or [])
    experiments = [event for event in events if event.get('event') == 'experiment']
    proposals = _proposals(events)

    detectors: dict[str, dict] = {}
    reasons: list[str] = []

    def run(name: str, fn, *args):
        try:
            found = fn(*args)
            detectors[name] = {'evaluated': True, 'records': len(found)}
            return found
        except Exception as exc:                 # noqa: BLE001
            detectors[name] = {'evaluated': False, 'reason': str(exc)[:160]}
            reasons.append(f'{name} 没能算：{str(exc)[:120]}')
            return []

    records: list[dict] = []
    records += run('repeated_proposal', _detect_repeated_proposal, events)
    records += run('invalid_structure_repeat', _detect_invalid_structure, events)
    records += run('same_family_tuning_no_gain', _detect_same_family_no_gain, events)
    records += run('repeated_change_no_gain', _detect_repeated_change_no_gain, events)
    if worklog_records is not None:
        records += run('operation_check_failed', _detect_operation_check_failed, worklog_records)
    else:
        detectors['operation_check_failed'] = {'evaluated': False, 'reason': '没有提供工作记录'}

    # 证据不够就不下结论
    insufficient = False
    if len(experiments) < MIN_EXPERIMENTS:
        insufficient = True
        reasons.append(f'实验只有 {len(experiments)} 次，少于 {MIN_EXPERIMENTS} 次，不足以判断')
    if not proposals:
        insufficient = True
        reasons.append('没有提案事件，无法判断是否重复')

    for record in records:
        record.update({
            'schema': SCHEMA,
            'run_id': run_id,
            'strategy_id': strategy_id,
            'detected_at': round(time.time(), 3),
            'evidence_complete': True,
        })

    return {
        'schema': SCHEMA,
        'run_id': run_id,
        'strategy_id': strategy_id,
        'event_count': len(events),
        'experiment_count': len(experiments),
        'proposal_count': len(proposals),
        'failure_count': len(records),
        'records': records,
        'insufficient': insufficient,
        'insufficient_reasons': reasons,
        'detectors': detectors,
        'verdict': 'insufficient_evidence' if insufficient else
                   ('failures_found' if records else 'clean'),
    }
