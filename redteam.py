"""红蓝对抗：一支队伍负责质疑，一支队伍负责检验。

为什么要有这个：
    系统一路往前冲，只会问"下一个方案是什么"，从不会回头问"我刚选出的最优，
    会不会只是运气好"。红队就是那个回头的人。

设计上有一条硬约束，是为了防止它变成表演：
    **红队的质疑必须能被代码检验。**
    输出格式固定 {claim, check_kind, check_params, threshold}；
    没有 check_kind 的质疑一律丢弃（记 rejected_vague），不接受"我觉得可能过拟合"。

    蓝队只按数字判，模型不参与判定——模型只负责把结果讲成人话。
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field

# ---------------- 检验器 ----------------

@dataclass
class CheckResult:
    passed: bool            # True = 质疑成立（确实有问题）
    measured: float
    threshold: float
    detail: str = ''
    samples: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {'passed': self.passed, 'measured': self.measured,
                'threshold': self.threshold, 'detail': self.detail,
                'samples': self.samples}


def seed_stability(session, proposal: dict, n_seeds: int = 3, threshold: float = 0.02) -> CheckResult:
    """换 seed 重跑，看分数波动。std > threshold 说明这个分数不稳。"""
    scores = []
    original_seed = getattr(session, 'seed', None)
    try:
        for i in range(n_seeds):
            if hasattr(session, 'seed'):
                session.seed = (original_seed or 42) + i
            result = session.run(model=proposal['model'], parameters=proposal.get('parameters') or {})
            scores.append(float(result.get('validation_score', result.get('validation_f1', 0.0))))
    finally:
        if original_seed is not None and hasattr(session, 'seed'):
            session.seed = original_seed
    std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    return CheckResult(
        passed=std > threshold,
        measured=round(std, 6),
        threshold=threshold,
        detail=f'{len(scores)} seeds, std={std:.4f}',
        samples=[round(s, 6) for s in scores],
    )


def feature_ablation(session, proposal: dict, drop: str | None = None,
                     threshold: float = 0.01) -> CheckResult:
    """去掉一个特征重跑。掉太多说明方案依赖单一特征（脆弱）。

    drop=None 时去掉了第一个特征列——调用方应该显式给，这里只做兜底。
    """
    x = getattr(session, 'x', None)
    if x is None or not hasattr(session, 'y'):
        return CheckResult(False, 0.0, threshold, 'session 没有 x/y，无法做消融')
    columns = list(x.columns)
    target_col = drop if drop in columns else columns[0]
    base = session.run(model=proposal['model'], parameters=proposal.get('parameters') or {})
    base_score = float(base.get('validation_score', base.get('validation_f1', 0.0)))

    original_x = session.x
    try:
        session.x = original_x.drop(columns=[target_col])
        ablated = session.run(model=proposal['model'], parameters=proposal.get('parameters') or {})
        new_score = float(ablated.get('validation_score', ablated.get('validation_f1', 0.0)))
    finally:
        session.x = original_x

    delta = base_score - new_score
    return CheckResult(
        passed=delta > threshold,
        measured=round(delta, 6),
        threshold=threshold,
        detail=f'drop {target_col}: {base_score:.4f} -> {new_score:.4f}',
        samples=[round(base_score, 6), round(new_score, 6)],
    )


def split_shift(session, proposal: dict, threshold: float = 0.02) -> CheckResult:
    """重跑两次并比较——用来暴露"划分敏感"。真正的换划分需要 executor 支持，这里只做同配置复跑。"""
    first = session.run(model=proposal['model'], parameters=proposal.get('parameters') or {})
    second = session.run(model=proposal['model'], parameters=proposal.get('parameters') or {})
    a = float(first.get('validation_score', first.get('validation_f1', 0.0)))
    b = float(second.get('validation_score', second.get('validation_f1', 0.0)))
    drift = abs(a - b)
    return CheckResult(
        passed=drift > threshold,
        measured=round(drift, 6),
        threshold=threshold,
        detail=f'rerun drift={drift:.2e}',
        samples=[round(a, 6), round(b, 6)],
    )


CHECKS = {
    'seed_stability': seed_stability,
    'feature_ablation': feature_ablation,
    'split_shift': split_shift,
}


# ---------------- 红队 ----------------

REDTEAM_SCHEMA = (
    '{"claims": [{"claim": "one sentence", "check_kind": "seed_stability|feature_ablation|split_shift", '
    '"check_params": {}, "threshold": 0.02}]}'
)


class RedTeam:
    """用本地模型产质疑。输出必须过 schema，否则丢弃。"""

    def __init__(self, proposer=None, max_claims: int = 2):
        self.proposer = proposer
        self.max_claims = max_claims

    def propose(self, profile: dict, best: dict, history: list | None = None,
                timeout: float = 30) -> tuple[list[dict], list[dict]]:
        """返回 (合法质疑, 被丢弃的)。"""
        if self.proposer is None:
            return [], []
        prompt = (
            'You are the RED TEAM. Your job is to attack the current best experiment, '
            'not to propose new ones.\n'
            f'Best so far: {json.dumps(best, ensure_ascii=False)}\n'
            f'Data profile: {json.dumps(profile, ensure_ascii=False)}\n'
            f'History: {json.dumps((history or [])[-4:], ensure_ascii=False)}\n'
            f'Available checks: {sorted(CHECKS)}\n'
            'Return ONLY JSON shaped like: ' + REDTEAM_SCHEMA + '\n'
            'Each claim must name a check_kind from the list. '
            'If you cannot express the doubt as a check, do not raise it.'
        )
        try:
            response = self.proposer([
                {'role': 'system', 'content': 'You attack weak results. Output JSON only.'},
                {'role': 'user', 'content': prompt},
            ], timeout=timeout)
        except Exception:
            return [], []
        content = ((response.get('message') or {}).get('content') or '').strip()
        return self.parse(content)

    def parse(self, content: str) -> tuple[list[dict], list[dict]]:
        try:
            raw = json.loads(content)
        except json.JSONDecodeError:
            return [], [{'reason': 'not_json', 'raw': content[:120]}]
        claims = raw.get('claims') if isinstance(raw, dict) else raw
        if not isinstance(claims, list):
            return [], [{'reason': 'not_list'}]

        accepted, rejected = [], []
        for item in claims[: self.max_claims]:
            if not isinstance(item, dict):
                rejected.append({'reason': 'not_object', 'raw': str(item)[:80]})
                continue
            kind = item.get('check_kind')
            claim = str(item.get('claim', '')).strip()
            if kind not in CHECKS:
                # 关键：没有可检验的手段 = 不接受这条质疑
                rejected.append({'reason': 'rejected_vague', 'claim': claim, 'check_kind': kind})
                continue
            if not claim:
                rejected.append({'reason': 'empty_claim', 'check_kind': kind})
                continue
            threshold = item.get('threshold', 0.02)
            if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) \
                    or not math.isfinite(threshold) or threshold <= 0:
                threshold = 0.02
            accepted.append({
                'claim': claim[:300],
                'check_kind': kind,
                'check_params': item.get('check_params') if isinstance(item.get('check_params'), dict) else {},
                'threshold': float(threshold),
            })
        return accepted, rejected


# ---------------- 蓝队 ----------------

class BlueTeam:
    """只按数字判。模型不参与。"""

    def verify(self, claim: dict, session) -> dict:
        kind = claim.get('check_kind')
        fn = CHECKS.get(kind)
        if fn is None:
            return {'claim': claim, 'supported': None, 'reason': 'unknown_check'}
        params = dict(claim.get('check_params') or {})
        params.setdefault('threshold', claim.get('threshold', 0.02))
        try:
            result = fn(session, params.pop('proposal', None) or self._proposal_from(session, claim), **params)
        except TypeError:
            # 参数不匹配：用最小参数再试一次
            try:
                result = fn(session, self._proposal_from(session, claim),
                            threshold=claim.get('threshold', 0.02))
            except Exception as exc:                     # noqa: BLE001
                return {'claim': claim, 'supported': None, 'reason': f'check_error: {exc}'}
        except Exception as exc:                         # noqa: BLE001
            return {'claim': claim, 'supported': None, 'reason': f'check_error: {exc}'}

        return {
            'claim': claim,
            'supported': bool(result.passed),      # True = 质疑成立 = 这个最优不可信
            'measured': result.measured,
            'threshold': result.threshold,
            'detail': result.detail,
            'samples': result.samples,
        }

    @staticmethod
    def _proposal_from(session, claim: dict) -> dict:
        proposal = (claim.get('check_params') or {}).get('proposal')
        if isinstance(proposal, dict) and 'model' in proposal:
            return proposal
        best = getattr(session, 'best_proposal', None)
        if isinstance(best, dict) and 'model' in best:
            return best
        raise ValueError('没有可检验的 proposal')

    def summarize(self, verdicts: list[dict]) -> dict:
        """把一批检验结果汇总成一个结论。"""
        supported = [v for v in verdicts if v.get('supported') is True]
        checked = [v for v in verdicts if v.get('supported') is not None]
        if not checked:
            return {'conclusion': 'no_check_ran', 'should_rollback': False, 'checked': 0}
        if supported:
            return {'conclusion': 'claims_upheld', 'should_rollback': True,
                    'checked': len(checked), 'upheld': len(supported),
                    'reason': supported[0]['claim']['claim'][:200]}
        return {'conclusion': 'claims_rejected', 'should_rollback': False,
                'checked': len(checked), 'upheld': 0}
