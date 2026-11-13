"""千问路由：让本地小模型自己决定「要不要调云端、调哪个」。

赛题 comment1 的原话是「每个 agent 背后的任务复杂程度可能不同，因此可以灵活选择」。
这一层把这个"灵活选择"交给本地千问来做——它见过当前的数据画像、历史分数和剩余预算，
比外部规则更清楚现在是不是真的卡住了。

和 JEV 的分工：
    千问（3B）  —— 便宜、快，做**路由判断**："要不要调云端？调哪个？"
    JEV（官方） —— 按 token 计费，做**质量判断**："这次诊断值不值得花钱？"

两者可以叠加：千问先判要不要，JEV 再判值不值。也可以只用千问。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

DEFAULT_BASE = 'http://127.0.0.1:11434'
DEFAULT_MODEL = 'qwen2.5:3b-instruct-q4_K_M'

# 可选的云端档位。名字要和 cloud_provider 支持的模型对得上。
CLOUD_TIERS = {
    'deepseek-flash': '通用档，快、便宜，适合常规诊断与方案生成',
    'deepseek-reasoner': '推理档，慢、贵，适合本地完全卡住且问题复杂时',
}

ROUTER_SCHEMA = (
    '{"need_cloud": true, "model": "deepseek-flash", "reason": "one sentence"}'
)


def _ask(prompt: str, base_url: str, model: str, timeout: int) -> str:
    body = json.dumps({
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'format': 'json',
        'options': {'temperature': 0, 'num_predict': 160},
    }).encode('utf-8')
    req = urllib.request.Request(
        base_url.rstrip('/') + '/api/chat', data=body,
        headers={'Content-Type': 'application/json'}, method='POST')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode('utf-8', 'replace'))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f'local router unreachable: {exc}') from None
    except json.JSONDecodeError:
        raise RuntimeError('local router returned non-JSON envelope') from None
    return ((payload.get('message') or {}).get('content') or '').strip()


def decide_escalation(state: dict, base_url: str = DEFAULT_BASE,
                      model: str = DEFAULT_MODEL, timeout: int = 30) -> dict:
    """让千问判断这一轮要不要调云端、调哪个。

    返回：
      {"need_cloud": bool, "model": str|None, "reason": str, "source": "qwen_router"}
    任何异常都返回 need_cloud=False（宁可不花钱，也不乱花钱）。
    """
    if not isinstance(state, dict) or not state:
        return {'need_cloud': False, 'model': None, 'reason': 'empty_state',
                'source': 'qwen_router'}

    prompt = (
        '你是研究系统的本地调度员。根据下面的状态判断：要不要调用云端大模型？\n\n'
        f'状态：{json.dumps(state, ensure_ascii=False)}\n\n'
        f'可选云端档位：{json.dumps(CLOUD_TIERS, ensure_ascii=False)}\n\n'
        '判断规则：\n'
        '- 本地还在进步（最近分数有上升）→ 不要调云端\n'
        '- 本地卡住（连续没进步）+ 还有云端额度 → 可以考虑\n'
        '- 数据很小（<500 行）→ 调云端意义不大，不要调\n'
        '- 问题很简单（只有一两类模型可选、参数空间小）→ 用 flash 档就够\n'
        '- 本地完全卡死且问题复杂 → 才考虑 reasoner 档\n'
        f'只输出 JSON：{ROUTER_SCHEMA}\n'
    )

    try:
        text = _ask(prompt, base_url, model, timeout)
        raw = json.loads(text)
    except (RuntimeError, json.JSONDecodeError) as exc:
        return {'need_cloud': False, 'model': None,
                'reason': f'router_failed: {str(exc)[:120]}', 'source': 'qwen_router'}

    if not isinstance(raw, dict):
        return {'need_cloud': False, 'model': None, 'reason': 'router_not_object',
                'source': 'qwen_router'}

    need = raw.get('need_cloud')
    if not isinstance(need, bool):
        # 模型给了个非布尔值 —— 按"不调"处理，并留痕
        return {'need_cloud': False, 'model': None,
                'reason': f'router_bad_need_cloud: {str(need)[:40]}', 'source': 'qwen_router'}

    tier = raw.get('model')
    if need and tier not in CLOUD_TIERS:
        # 说要调，但档位不合法 → 回落到便宜的 flash 档
        tier = 'deepseek-flash'
    if not need:
        tier = None

    return {
        'need_cloud': need,
        'model': tier,
        'reason': str(raw.get('reason', ''))[:200],
        'source': 'qwen_router',
    }


if __name__ == '__main__':
    demo = {
        'task': 't2_imbalanced',
        'trigger': 'validation_plateau',
        'experiments_used': 4,
        'best_validation_score': 0.66,
        'recent_scores': [0.30, 0.66, 0.43, 0.30],
        'remaining_cloud_calls': 1,
        'rows': 3000,
    }
    print(json.dumps(decide_escalation(demo), ensure_ascii=False, indent=2))
