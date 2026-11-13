"""Jev Harness 精简版：System 1.5 决策层，目的是**省 token**。

参考 https://github.com/ismaelsoilet/jev-harness （TypeSafe Jev 生态里那个 "token optimizer"）。

它解决的问题不是"要不要花钱"，而是：

    大模型很贵，但 agent 循环里 80% 的"该不该继续"其实是廉价判断——
    参数写错了、在重复同一个实验、这次改进根本不值得留、该用哪档模型……
    这些用**确定性规则**（<1ms、0 token）就能判掉，根本不该送去问大模型。

五道门（对应 jev-harness 的五个 semantic gates）：

    1. trivial_error      这次失败是不是琐碎的（参数越界/重复提案）→ 是就别升级
    2. doom_loop          是不是在死循环（同样的方案反复出现）→ 是就先打断
    3. completion_veto    这次改进值不值得留下（太小就别花力气）→ 否决
    4. effort             该给多大推理强度（cheap / normal / deep）
    5. routing            该用哪个档位的模型（local / flash / reasoner）

门全过、且最后路由说"要上云"，才真的花云端 token。
每个门都返回**可审计的判据**，不返回"我觉得"。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

# 与 cloud_provider 的档位对齐
TIER_LOCAL = 'local'
TIER_FLASH = 'deepseek-flash'
TIER_REASONER = 'deepseek-reasoner'


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str = ''

    def to_dict(self) -> dict:
        return {'gate': self.name, 'passed': self.passed, 'detail': self.detail}


@dataclass
class Decision:
    """一次决策的完整留痕。给评委看的就是这个结构。"""
    action: str                     # 'stay_local' | 'escalate' | 'takeover'
    tier: str                       # local / flash / reasoner
    effort: str                     # cheap / normal / deep
    gates: list = field(default_factory=list)
    reason: str = ''
    saved_tokens_estimate: int = 0  # 这次决策省掉的（估算）前端 token

    def to_dict(self) -> dict:
        return {
            'action': self.action,
            'tier': self.tier,
            'effort': self.effort,
            'gates': [g.to_dict() if isinstance(g, GateResult) else g for g in self.gates],
            'reason': self.reason,
            'saved_tokens_estimate': self.saved_tokens_estimate,
        }


class JevHarness:
    """确定性优先的决策层。**不调用任何模型**——它本身就是省 token 的那一步。"""

    def __init__(self, min_gain: float = 0.005, doom_loop_window: int = 3,
                 small_data_rows: int = 500):
        self.min_gain = min_gain
        self.doom_loop_window = doom_loop_window
        self.small_data_rows = small_data_rows

    # ---------- 五道门 ----------
    def gate_trivial_error(self, ctx: dict) -> GateResult:
        """参数越界/格式错误这类，重试一次本地就够，不值得上云。"""
        reasons = ctx.get('recent_rejections') or []
        if not reasons:
            return GateResult('trivial_error', True, 'no recent rejections')
        trivial = sum(1 for r in reasons if any(
            kw in str(r).lower() for kw in
            ('unsupported', 'outside', 'invalid', 'must be', 'duplicate')))
        if trivial >= len(reasons):
            return GateResult('trivial_error', False,
                              f'all {len(reasons)} rejections are schema/range errors - fix locally')
        return GateResult('trivial_error', True, f'{trivial}/{len(reasons)} trivial')

    def gate_doom_loop(self, ctx: dict) -> GateResult:
        """最近几轮是不是在原地打转（同样的模型+参数反复出现）。"""
        recent = [json.dumps({'m': r.get('model'), 'p': r.get('parameters')}, sort_keys=True)
                  for r in (ctx.get('recent_proposals') or [])[-self.doom_loop_window:]]
        if len(recent) < 2:
            return GateResult('doom_loop', True, 'too few rounds')
        unique = len(set(recent))
        if unique == 1:
            return GateResult('doom_loop', False, f'{len(recent)} identical proposals - break the loop')
        return GateResult('doom_loop', True, f'{unique}/{len(recent)} distinct')

    def gate_completion_veto(self, ctx: dict) -> GateResult:
        """上一次改进小到没有意义，就别再为它花力气。"""
        gain = ctx.get('last_gain')
        if gain is None:
            return GateResult('completion_veto', True, 'no last gain')
        if isinstance(gain, (int, float)) and 0 < gain < self.min_gain:
            return GateResult('completion_veto', False,
                              f'last gain {gain:.4f} < {self.min_gain} - not worth spending')
        return GateResult('completion_veto', True, f'last gain {gain}')

    def gate_effort(self, ctx: dict) -> tuple[GateResult, str]:
        """该给多大推理强度。数据越小、模型越简单，越不该烧 deep。"""
        rows = int(ctx.get('rows') or 0)
        stuck = int(ctx.get('no_improvement') or 0)
        if rows and rows < self.small_data_rows:
            return GateResult('effort', True, f'{rows} rows - data too small for deep reasoning'), 'cheap'
        if stuck >= 3:
            return GateResult('effort', True, f'{stuck} stuck rounds - worth deep reasoning'), 'deep'
        return GateResult('effort', True, f'{stuck} stuck rounds'), 'normal'

    def gate_routing(self, ctx: dict, effort: str) -> tuple[GateResult, str]:
        """该用哪一档。这是"省 token"的最后一道闸。"""
        rows = int(ctx.get('rows') or 0)
        if rows and rows < self.small_data_rows:
            return GateResult('routing', True, f'{rows} rows - stay local'), TIER_LOCAL
        if effort == 'deep':
            return GateResult('routing', True, 'deep reasoning needed'), TIER_REASONER
        return GateResult('routing', True, 'routine diagnosis'), TIER_FLASH

    # ---------- 主入口 ----------
    def decide(self, ctx: dict) -> Decision:
        """跑完五道门，给出一个决定。全程 0 token。"""
        gates: list[GateResult] = []

        g1 = self.gate_trivial_error(ctx)
        gates.append(g1)
        if not g1.passed:
            return Decision('stay_local', TIER_LOCAL, 'cheap', gates,
                            'rejections are schema errors; fix locally', 0)

        g2 = self.gate_doom_loop(ctx)
        gates.append(g2)
        if not g2.passed:
            return Decision('stay_local', TIER_LOCAL, 'cheap', gates,
                            'identical proposals repeating; change strategy locally', 0)

        g3 = self.gate_completion_veto(ctx)
        gates.append(g3)
        if not g3.passed:
            return Decision('takeover', TIER_LOCAL, 'cheap', gates,
                            'gain too small to justify cloud spend', 0)

        g4, effort = self.gate_effort(ctx)
        gates.append(g4)
        g5, tier = self.gate_routing(ctx, effort)
        gates.append(g5)

        if tier == TIER_LOCAL:
            return Decision('stay_local', TIER_LOCAL, effort, gates,
                            'data too small; cloud would not help', 0)

        # 估算：被这五道门挡下来的那些调用，本来要花多少 prompt token
        saved = self._estimate_saved(ctx)
        return Decision('escalate', tier, effort, gates,
                        f'passed all gates; routing to {tier} with {effort} effort', saved)

    @staticmethod
    def _estimate_saved(ctx: dict) -> int:
        """粗估：本地卡了 N 轮，每轮若都上云约 800 prompt token。"""
        stuck = int(ctx.get('no_improvement') or 0)
        return max(0, stuck) * 800


if __name__ == '__main__':
    demo = {
        'task': 't2_imbalanced',
        'rows': 3000,
        'no_improvement': 3,
        'last_gain': 0.0001,
        'recent_rejections': [],
        'recent_proposals': [
            {'model': 'random_forest', 'parameters': {'max_depth': 5}},
            {'model': 'random_forest', 'parameters': {'max_depth': 5}},
            {'model': 'random_forest', 'parameters': {'max_depth': 5}},
        ],
    }
    print(json.dumps(JevHarness().decide(demo).to_dict(), ensure_ascii=False, indent=2))
