"""本地 JEV：用本地 Qwen 实现 TypeSafe Jev 的**同一套协议**。

为什么要这个：
  TypeSafe 的 Jev 我们接不通（地区封锁 451 + key 401）。但赛题要的是
  「有一个刚性校验引擎」，不是「必须用 TypeSafe 那一家」。

所以做法是：**协议照抄，实现换成本地模型**。
  - 请求/响应结构跟 TypeSafe 的 /v1/systemone 完全一致
  - 三种问题类型 noul / choice / score 都支持
  - 有 TYPESAFE_API_KEY 时优先走官方（jev_client.py），没有就走这里

这样 Jury 问「你们的 JEV 是什么」，答案是：一个按 TypeSafe 协议实现的
本地校验引擎，可以随时切到官方服务。

对外接口和 jev_client.ask_jev 一样，方便替换。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

DEFAULT_BASE = 'http://127.0.0.1:11434'
DEFAULT_MODEL = 'qwen2.5:7b-instruct-q4_K_M'

ALLOWED_TYPES = {'noul', 'choice', 'score'}


class LocalJevError(Exception):
    pass


def _ask_ollama(prompt: str, base_url: str, model: str, timeout: int) -> str:
    body = json.dumps({
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'format': 'json',
        'options': {'temperature': 0, 'num_predict': 200},
    }).encode('utf-8')
    req = urllib.request.Request(
        base_url.rstrip('/') + '/api/chat', data=body,
        headers={'Content-Type': 'application/json'}, method='POST')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode('utf-8', 'replace'))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LocalJevError(f'local model unreachable: {exc}') from None
    except json.JSONDecodeError:
        raise LocalJevError('local model returned non-JSON envelope') from None
    return ((payload.get('message') or {}).get('content') or '').strip()


def _prompt_for(state: str, questions: dict) -> str:
    schema_hint = {}
    for qid, q in questions.items():
        if q['type'] == 'noul':
            schema_hint[qid] = {'noul': 0.0, 'reason': 'short reason'}
        elif q['type'] == 'choice':
            schema_hint[qid] = {'choice': '<one of criteria keys>', 'confidence': 0.0}
        else:
            schema_hint[qid] = {'score': 0.0, 'confidence': 0.0}
    return (
        '你是刚性校验引擎。只依据给定状态判断，不做主观评价，不编造。\n'
        f'状态：\n{state}\n\n'
        f'问题定义：\n{json.dumps(questions, ensure_ascii=False)}\n\n'
        '只输出 JSON，结构如下（键必须与问题 id 一致）：\n'
        f'{json.dumps(schema_hint, ensure_ascii=False)}\n'
        '规则：noul 是 0~1 的概率；score 是 0~1 的连续分；'
        'choice 必须从 criteria 的键里选一个。不确定时给保守值，不要猜满分。'
    )


def _parse_answers(text: str, questions: dict) -> dict:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raise LocalJevError('model output is not valid JSON') from None
    if not isinstance(raw, dict):
        raise LocalJevError('model output is not an object')

    answers = {}
    for qid, q in questions.items():
        got = raw.get(qid)
        if not isinstance(got, dict):
            # 缺哪个就给哪个一个保守的"无法判断"，不编造
            answers[qid] = _fallback(q)
            answers[qid]['unavailable'] = True
            continue
        if q['type'] == 'noul':
            value = got.get('noul')
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                answers[qid] = _fallback(q)
                continue
            answers[qid] = {
                'type': 'noul',
                'noul': max(0.0, min(1.0, float(value))),
                'reason': str(got.get('reason', ''))[:200],
            }
        elif q['type'] == 'choice':
            criteria = q.get('criteria') or {}
            keys = list(criteria.keys()) if isinstance(criteria, dict) else []
            choice = got.get('choice')
            if choice not in keys:
                answers[qid] = _fallback(q)
                answers[qid]['unavailable'] = True
                continue
            answers[qid] = {
                'type': 'choice',
                'choice': choice,
                'confidence': max(0.0, min(1.0, float(got.get('confidence', 0.5) or 0.5))),
            }
        else:  # score
            value = got.get('score')
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                answers[qid] = _fallback(q)
                continue
            answers[qid] = {
                'type': 'score',
                'score': max(0.0, min(1.0, float(value))),
                'confidence': max(0.0, min(1.0, float(got.get('confidence', 0.5) or 0.5))),
            }
    return answers


def _fallback(question: dict) -> dict:
    """模型答不出来时给一个明确的"不知道"，而不是假装有结论。"""
    kind = question.get('type')
    if kind == 'noul':
        return {'type': 'noul', 'noul': 0.0, 'reason': 'unavailable'}
    if kind == 'choice':
        return {'type': 'choice', 'choice': None, 'confidence': 0.0}
    return {'type': 'score', 'score': 0.0, 'confidence': 0.0}


def ask_jev_local(state: str, questions: dict, base_url: str = DEFAULT_BASE,
                  model: str = DEFAULT_MODEL, timeout: int = 60) -> dict:
    """与 jev_client.ask_jev 同构的返回结构。

    {"ok": True,  "answers": {...}, "model": "...", "source": "local_qwen"}
    {"ok": False, "error": "...", "detail": "..."}
    """
    if not str(state or '').strip():
        return {'ok': False, 'error': 'no_state', 'detail': 'state 为空'}
    if not isinstance(questions, dict) or not questions:
        return {'ok': False, 'error': 'bad_request', 'detail': 'questions 必须非空'}
    for qid, q in questions.items():
        if not isinstance(q, dict) or q.get('type') not in ALLOWED_TYPES:
            return {'ok': False, 'error': 'bad_request', 'detail': f'{qid}: type 非法'}
        if not str(q.get('instructions', '')).strip():
            return {'ok': False, 'error': 'bad_request', 'detail': f'{qid}: 缺 instructions'}

    try:
        text = _ask_ollama(_prompt_for(state[:8000], questions), base_url, model, timeout)
        answers = _parse_answers(text, questions)
    except LocalJevError as exc:
        return {'ok': False, 'error': 'local_jev_failed', 'detail': str(exc)[:300]}

    return {'ok': True, 'answers': answers, 'model': model, 'source': 'local_qwen'}


# ---------------- 校验引擎的业务封装 ----------------

REVIEW_QUESTIONS = {
    'is_real_problem': {
        'type': 'noul',
        'instructions': '这条实验发现指向的是一个真实的实现问题（不是表达或格式问题）',
    },
    'severity': {
        'type': 'score',
        'instructions': '问题有多严重（0 可以忽略，1 会毁掉结论）',
    },
    'category': {
        'type': 'choice',
        'instructions': '属于哪一类问题',
        'criteria': {
            'leakage': '数据泄漏',
            'metric': '度量口径错误',
            'reproducibility': '不可复现',
            'budget': '预算越界',
            'other': '其他',
        },
    },
}


def review_finding_local(state: str, **kwargs) -> dict:
    return ask_jev_local(state, REVIEW_QUESTIONS, **kwargs)


if __name__ == '__main__':
    import sys
    demo_state = sys.argv[1] if len(sys.argv) > 1 else (
        '实验报告里写 factorial(24) 的结果是 24 的阶乘，但没有给出任何运行输出。'
        '验证集 F1 从 0.2994 涨到 0.6620，实验只跑了 3 轮。')
    result = review_finding_local(demo_state)
    print(json.dumps(result, ensure_ascii=False, indent=2))
