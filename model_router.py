"""模型分层调度：让不同的 agent 用不同档位的模型。

赛题 comment1 的原话是「每个 agent 背后的任务复杂程度可能不同，因此可以灵活选择
从而降低 token 成本」。这一层就是那个"灵活选择"。

分层原则（按任务复杂度，不是按角色名气）：

    Scientist  提实验方案   最难，要理解数据画像和历史分数     → 大模型
    RedTeam    找茬         中等，要能指出可疑点               → 小模型够用
    Reviewer   判好坏       不让模型参与（纯规则）            → 0 token
    Analyst    画数据像     纯统计                             → 0 token
    Manager    分预算       纯算术                             → 0 token

结论：真正花钱的只有 Scientist 和 RedTeam，所以只需要两档。
"""
from __future__ import annotations

import json
import time


class ModelRouter:
    """按角色分派模型。没配的角色回落到默认模型。"""

    def __init__(self, factory, routes: dict):
        """
        factory(model_name) -> callable，负责真正建一个客户端（比如 LocalQwenClient）
        routes: {'scientist': 'qwen2.5:7b-...', 'redteam': 'qwen2.5:3b-...'}
        """
        if not callable(factory):
            raise TypeError('factory 必须可调用')
        self.factory = factory
        self.routes = {k: v for k, v in (routes or {}).items() if v}
        self._clients: dict[str, object] = {}
        self.usage: list[dict] = []

    def for_role(self, role: str):
        model = self.routes.get(role)
        if not model:
            return None
        if model not in self._clients:
            self._clients[model] = self.factory(model)
        client = self._clients[model]

        def _call(messages, timeout=None):
            started = time.monotonic()
            response = client(messages, timeout=timeout)
            self.usage.append({
                'role': role,
                'model': model,
                'prompt_tokens': int(response.get('prompt_eval_count', 0) or 0),
                'completion_tokens': int(response.get('eval_count', 0) or 0),
                'seconds': round(time.monotonic() - started, 3),
            })
            return response

        return _call

    def summary(self) -> dict:
        """按角色汇总 token 消耗——这是要给人看的成本台账。"""
        by_role: dict[str, dict] = {}
        for item in self.usage:
            bucket = by_role.setdefault(item['role'], {
                'model': item['model'], 'calls': 0,
                'prompt_tokens': 0, 'completion_tokens': 0, 'seconds': 0.0,
            })
            bucket['calls'] += 1
            bucket['prompt_tokens'] += item['prompt_tokens']
            bucket['completion_tokens'] += item['completion_tokens']
            bucket['seconds'] = round(bucket['seconds'] + item['seconds'], 3)
        total = {
            'calls': sum(b['calls'] for b in by_role.values()),
            'prompt_tokens': sum(b['prompt_tokens'] for b in by_role.values()),
            'completion_tokens': sum(b['completion_tokens'] for b in by_role.values()),
        }
        return {'by_role': by_role, 'total': total}

    def to_json(self) -> str:
        return json.dumps(self.summary(), ensure_ascii=False, indent=2)
