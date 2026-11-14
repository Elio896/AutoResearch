"""让 DeepSeek 提一个候选策略：必须有假设、有否证条件，只许动白名单。

它读失败摘要，交回来的东西代码要过三关：

1. **范围**：只能改 `classifier_prompt` / `planning_prompt` / `candidate_selection_rules`；
2. **假设**：what / why / expected_improvement / falsifier / side_effects 五项齐全且非空；
   **没有 falsifier 的直接退**——说不清"什么结果会否定它"，就不是假设，是许愿；
3. **不越权**：候选只能落成 `candidate`，正式策略、数据标签、评分函数、测试集都碰不到。

它一次只许提一个修改方向；提两个以上就退回，让证据说话而不是广撒网。
"""
from __future__ import annotations

import json
import time

import strategy_store
from strategy_store import WHITELIST_FIELDS, StrategyStore

REQUIRED_HYPOTHESIS = ('what', 'why', 'expected_improvement', 'falsifier', 'side_effects')
MAX_FIELD_CHARS = 4000
MAX_CHANGED_FIELDS = 1


class CandidateRejected(ValueError):
    """候选不合规。消息里带具体原因，不许含糊。"""


def build_prompt(failure_report: dict, active: dict) -> list[dict]:
    """给云端模型的输入：只给证据和边界，不给结论。"""
    records = failure_report.get('records') or []
    summary = [{
        'failure_type': record.get('failure_type'),
        'measurements': record.get('measurements'),
        'event_refs': record.get('event_refs'),
        'run_id': record.get('run_id'),
    } for record in records]

    body = {
        '当前正式策略': active.get('changes'),
        '失败证据': summary,
        '证据不足原因': failure_report.get('insufficient_reasons') or [],
        '允许修改的字段': sorted(WHITELIST_FIELDS),
        '要求': {
            '每次只提一个修改方向': True,
            '必须给出假设五项': list(REQUIRED_HYPOTHESIS),
            'falsifier': '必须写成可观测的结果，例如「新策略在独立任务上重复率不降反升」',
            '禁止': ['改数据标签', '改评分函数', '改测试集', '动验收门槛',
                     '删除失败实验', '直接改正式策略'],
        },
        '输出格式': {
            'changes': {'字段名': '新的值'},
            'hypothesis': {key: '...' for key in REQUIRED_HYPOTHESIS},
            'target_failure_types': ['要治的失败类型'],
            'evidence_run_ids': ['依据哪几次运行'],
        },
    }
    content = ('你要针对下面这些失败证据，提出**一个**策略修改。\n'
               + json.dumps(body, ensure_ascii=False, indent=2)
               + '\n只输出 JSON，不要解释。')
    return [{'role': 'user', 'content': content}]


def validate_candidate(payload: dict, active_changes: dict) -> tuple[bool, list[str]]:
    """结构 + 范围 + 假设完整性。全部通过才算一条候选。"""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return False, ['返回值不是 JSON 对象']

    changes = payload.get('changes')
    if not isinstance(changes, dict) or not changes:
        errors.append('changes 必须是非空对象')
    else:
        unknown = sorted(set(changes) - set(WHITELIST_FIELDS))
        if unknown:
            errors.append(f'越权字段：{unknown}（只允许 {sorted(WHITELIST_FIELDS)}）')
        if len(changes) > MAX_CHANGED_FIELDS:
            errors.append(f'一次改了 {len(changes)} 个字段，超过 {MAX_CHANGED_FIELDS} 个'
                          '——一次只许提一个修改方向')
        for key, value in changes.items():
            if key not in WHITELIST_FIELDS:      # 越权字段上面已经报过，这里不再 KeyError
                continue
            expected = WHITELIST_FIELDS[key]
            if not isinstance(value, expected):
                errors.append(f'{key} 必须是 {expected.__name__}')
            elif isinstance(value, str) and len(value) > MAX_FIELD_CHARS:
                errors.append(f'{key} 超过 {MAX_FIELD_CHARS} 字符')

    hypothesis = payload.get('hypothesis')
    if not isinstance(hypothesis, dict):
        errors.append('缺 hypothesis')
    else:
        for key in REQUIRED_HYPOTHESIS:
            value = hypothesis.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f'hypothesis.{key} 不能为空')

    kinds = payload.get('target_failure_types')
    if kinds is not None and (not isinstance(kinds, list) or
                              any(not isinstance(item, str) for item in kinds)):
        errors.append('target_failure_types 必须是字符串数组')

    evidence = payload.get('evidence_run_ids')
    if evidence is not None and (not isinstance(evidence, list) or
                                 any(not isinstance(item, str) for item in evidence)):
        errors.append('evidence_run_ids 必须是字符串数组')

    if isinstance(changes, dict) and changes and not errors:
        merged = dict(active_changes)
        merged.update(changes)
        if json.dumps(merged, sort_keys=True, ensure_ascii=False) == \
                json.dumps(active_changes, sort_keys=True, ensure_ascii=False):
            errors.append('改完和当前正式策略一模一样，不算修改')

    return (not errors), errors


def _extract_json(raw: str) -> dict:
    body = (raw or '').strip()
    start, end = body.find('{'), body.rfind('}')
    if start < 0 or end <= start:
        raise CandidateRejected('模型返回里没有 JSON 对象')
    try:
        return json.loads(body[start:end + 1])
    except Exception as exc:                    # noqa: BLE001
        raise CandidateRejected(f'JSON 解析失败：{str(exc)[:120]}') from None


def propose_from_failures(store: StrategyStore, failure_report: dict, client,
                          *, timeout: float = 90.0, proposed_by: str = 'deepseek',
                          simulated: bool = False) -> dict:
    """调用云端模型 → 校验 → 落成 candidate。任何一步不过都不写库。"""
    active = store.active()
    if active is None:
        raise CandidateRejected('还没有正式策略，先 bootstrap')

    if failure_report.get('insufficient'):
        reasons = '；'.join(failure_report.get('insufficient_reasons') or []) or '证据不足'
        raise CandidateRejected(f'证据不足，不许提候选：{reasons}')
    if not (failure_report.get('records') or []):
        raise CandidateRejected('这次运行没有失败证据，没有可治的东西')

    messages = build_prompt(failure_report, active)
    raw = client(messages, timeout=timeout)
    text = raw if isinstance(raw, str) else \
        ((raw.get('message') or {}).get('content') or '')
    payload = _extract_json(text)

    ok, errors = validate_candidate(payload, active['changes'])
    if not ok:
        raise CandidateRejected('候选不合规：' + '；'.join(errors))

    hypothesis = payload['hypothesis']
    reason = (f"{hypothesis['what']} —— 理由：{hypothesis['why']}"
              f"；预期：{hypothesis['expected_improvement']}"
              f"；否证条件：{hypothesis['falsifier']}")
    version = store.propose(
        payload['changes'],
        reason=reason,
        evidence_run_ids=payload.get('evidence_run_ids') or
        [record.get('run_id') for record in failure_report.get('records') or []
         if record.get('run_id')],
        proposed_by=proposed_by,
        hypothesis=hypothesis,
        target_failure_types=payload.get('target_failure_types') or
        [record.get('failure_type') for record in failure_report.get('records') or []],
    )
    version['simulated'] = bool(simulated)
    return version


def simulated_client(payload_override: dict | None = None):
    """验收用的假客户端：明确标 simulated，不当成 DeepSeek 的真实能力。"""
    def _call(messages, timeout=None):
        if payload_override is not None:
            return {'message': {'content': json.dumps(payload_override, ensure_ascii=False)}}
        return {'message': {'content': json.dumps({
            'changes': {'planning_prompt': '先查事后字段与 ID 列，再定指标与基线。'},
            'hypothesis': {
                'what': '在规划提示词里加入事后字段排查步骤',
                'why': '失败证据显示重复提案集中在同一模型族调参，规划没有先排雷',
                'expected_improvement': '重复提案率下降，合法输出率不下降',
                'falsifier': '独立任务上新策略重复率不降反升，或合法输出率下降超过 5%',
                'side_effects': '多一次列级扫描，单次运行耗时可能增加',
            },
            'target_failure_types': ['repeated_proposal'],
            'evidence_run_ids': ['run_stuck'],
        }, ensure_ascii=False)}}
    return _call
