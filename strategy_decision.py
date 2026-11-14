"""策略升级的最后一关：Jev 做选择，代码做约束。

分工要说清：
- **Jev**：在有界选项里选一个（upgrade / keep / defer），它不做审批，
  也不决定"要不要调用 DeepSeek"——两个专业分支是并列的；
- **代码**：先检查对照评估的结论，没过就不许问 Jev；Jev 的回复必须是合法选项；
  Jev 不可用就保持旧策略，**不伪造一个决定**。

任何一项失败都不切换正式策略。
"""
from __future__ import annotations

import json

from strategy_store import StrategyStore

DECISIONS = ('upgrade', 'keep', 'defer')

_SYNONYMS = {
    'upgrade': 'upgrade', '升': 'upgrade', '升级': 'upgrade', 'switch': 'upgrade',
    'keep': 'keep', '保留': 'keep', '不换': 'keep', '保持': 'keep', 'stay': 'keep',
    'defer': 'defer', '暂缓': 'defer', '等等': 'defer', 'wait': 'defer', 'later': 'defer',
}


def code_checks(eval_report: dict) -> tuple[dict, list[str]]:
    """把对照评估的结果翻成代码侧检查项。没跑评估就不算通过。"""
    checks: dict = {}
    problems: list[str] = []

    verdict = (eval_report or {}).get('verdict')
    if verdict is None:
        return {}, ['没有对照评估报告，代码侧检查无从谈起']
    checks['eval_verdict_upgrade'] = verdict == 'upgrade'
    if verdict != 'upgrade':
        problems.append(f'对照评估结论是 {verdict}，不是 upgrade')
    if not (eval_report or {}).get('checks'):
        problems.append('评估报告缺少实际检查项')

    for check in (eval_report or {}).get('checks') or []:
        name = check.get('check')
        if not name:
            continue
        checks[name] = check.get('passed') is True
        if check.get('passed') is not True:
            problems.append(f'{name} 未通过')

    if not checks:
        problems.append('评估报告里没有可用的检查项')
    return checks, problems


def build_request(candidate: dict, diff: dict, eval_report: dict) -> dict:
    """给 Jev 的东西：候选差异 + 对照证据 + 三个选项。不给它看的就别让它猜。"""
    old = (eval_report or {}).get('old') or {}
    new = (eval_report or {}).get('new') or {}
    state = (
        f"候选策略 {candidate['strategy_id']} 想替换当前正式策略。\n"
        f"改动：{json.dumps(diff.get('changes'), ensure_ascii=False)}\n"
        f"理由：{candidate.get('reason', '')}\n"
        f"假设：{json.dumps(candidate.get('hypothesis'), ensure_ascii=False)}\n"
        f"对照证据：旧 legal_output_rate={old.get('legal_output_rate')} "
        f"duplicate_rate={old.get('duplicate_rate')} mean_best_score={old.get('mean_best_score')}；"
        f"新 legal_output_rate={new.get('legal_output_rate')} "
        f"duplicate_rate={new.get('duplicate_rate')} mean_best_score={new.get('mean_best_score')}\n"
        f"评估结论：{eval_report.get('verdict')}（{eval_report.get('reason')}）"
    )
    questions = {
        'strategy_decision': {
            'type': 'choice',
            'instructions': '这个候选策略要不要升级为正式策略？只选一个。',
            'criteria': list(DECISIONS),
        }
    }
    return {'state': state, 'questions': questions}


def normalize_decision(answer) -> str | None:
    """Jev 的回复可能有几种写法，认不出来就算无效。"""
    if isinstance(answer, dict):
        for key in ('choice', 'decision', 'value', 'answer', 'label'):
            if key in answer:
                answer = answer[key]
                break
    if not isinstance(answer, str):
        return None
    text = answer.strip().lower()
    if text in DECISIONS:
        return text
    return _SYNONYMS.get(text)


def decide(store: StrategyStore, candidate_id: str, eval_report: dict,
           jev_gate=None, *, by: str = 'jev') -> dict:
    """按顺序判：代码检查 → Jev 选择 → 落库。任何一步不过都不切换。"""
    candidate = store.get(candidate_id)
    if candidate is None:
        raise KeyError(candidate_id)
    if candidate['status'] != 'candidate':
        raise ValueError(f'{candidate_id} 不是候选（当前 {candidate["status"]}）')

    active = store.active()
    if active is None:
        return {'decision': 'keep', 'applied': False, 'source': 'code',
                'reason': '没有正式策略，不能谈升级'}
    diff = store.diff(active['strategy_id'], candidate_id)

    checks, problems = code_checks(eval_report)
    if problems:
        store.record_decision(candidate_id, 'keep', source='code',
                              evidence={'checks': checks, 'problems': problems},
                              reason='；'.join(problems)[:300])
        return {'decision': 'keep', 'applied': False, 'source': 'code',
                'reason': '代码侧检查未通过：' + '；'.join(problems)[:200],
                'checks': checks, 'jev': None}

    request = build_request(candidate, diff, eval_report)
    if jev_gate is None:
        store.record_decision(candidate_id, 'keep', source='jev_unavailable',
                              evidence={'checks': checks, 'request_state': request['state'][:500]},
                              reason='Jev 不可用，保持旧策略（不伪造决定）')
        return {'decision': 'keep', 'applied': False, 'source': 'jev_unavailable',
                'reason': 'Jev 不可用，保持旧策略', 'checks': checks, 'jev': None}

    try:
        raw = jev_gate(request['state'], request['questions'])
    except Exception as exc:                    # noqa: BLE001
        store.record_decision(candidate_id, 'keep', source='jev_error',
                              evidence={'checks': checks, 'error': str(exc)[:200]},
                              reason=f'Jev 调用失败：{str(exc)[:120]}')
        return {'decision': 'keep', 'applied': False, 'source': 'jev_error',
                'reason': f'Jev 调用失败：{str(exc)[:120]}', 'checks': checks, 'jev': None}

    if not isinstance(raw, dict) or not raw.get('ok'):
        detail = (raw or {}).get('error') if isinstance(raw, dict) else 'bad_response'
        store.record_decision(candidate_id, 'keep', source='jev_unavailable',
                              evidence={'checks': checks, 'response': str(raw)[:300]},
                              reason=f'Jev 没有给出可用回答：{detail}')
        return {'decision': 'keep', 'applied': False, 'source': 'jev_unavailable',
                'reason': f'Jev 没有给出可用回答：{detail}', 'checks': checks, 'jev': raw}

    answers = raw.get('answers') or {}
    choice = normalize_decision(answers.get('strategy_decision'))
    if choice is None:
        store.record_decision(candidate_id, 'keep', source='jev_invalid',
                              evidence={'checks': checks, 'answers': str(answers)[:300]},
                              reason='Jev 回答不在允许选项里')
        return {'decision': 'keep', 'applied': False, 'source': 'jev_invalid',
                'reason': 'Jev 回答不在允许选项里', 'checks': checks, 'jev': raw}

    jev_record = {'decision': choice, 'source': 'jev',
                  'model': raw.get('model'), 'answers': answers}
    if choice != 'upgrade':
        store.record_decision(candidate_id, choice, source='jev',
                              evidence={'checks': checks, 'jev': jev_record},
                              reason=f'Jev 选择 {choice}')
        return {'decision': choice, 'applied': False, 'source': 'jev',
                'reason': f'Jev 选择 {choice}', 'checks': checks, 'jev': jev_record}

    if not checks or not all(checks.values()):
        # 理论上到不了这里（上面已经拦过），留着当最后一道保险
        store.record_decision(candidate_id, 'keep', source='code',
                              evidence={'checks': checks}, reason='代码检查未全部通过')
        return {'decision': 'keep', 'applied': False, 'source': 'code',
                'reason': '代码检查未全部通过', 'checks': checks, 'jev': jev_record}

    activated = store.activate(candidate_id, checks=checks, jev=jev_record,
                               by=by, reason='代码检查通过 + Jev 选择升级')
    return {'decision': 'upgrade', 'applied': True, 'source': 'jev',
            'reason': '已切换', 'checks': checks, 'jev': jev_record,
            'strategy_id': activated['strategy_id'],
            'previous_id': store.events()[-1].get('previous_id')}
