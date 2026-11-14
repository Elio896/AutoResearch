"""可计算信号路由：把"置信度"从玄学数字换成可审计的行为证据。

来自二德子的建议（2026-11-14）：

    置信度的用途是决定「要不要花钱升级」，那它应该是**可验证的行为信号**，
    不是让 LLM 给自己打分。7B 自评全是 0.6 不是模型坏了，
    是"自评"这个方法本身没有校准依据——换成 70B 也一样。

四条信号（全部来自现有系统，不需要训练、不需要云端）：

    1. 首次合法率   提案第一次就过白名单的比例。老要重试 = 模型不懂约束
    2. 重复率       和已有实验重复的比例。高重复 = 没在探索
    3. 停滞轮数     连续几轮验证分不涨（这个系统里已经有了）
    4. 同类改动 Δ   改动类型（换模型 / 调 C / 调树深）历史上平均涨多少

决策规则（直接 if-else，不训练）：

    如果 停滞轮数 >= 2 且 历史同类 Δ <= 0
    或   首次合法率 < 0.5
    → 升级云端（1 次）
    否则继续本地

7B 自报的 confidence_before **保留但不参与决策**，只当遥测——
将来真要做校准器时，它就是现成的特征。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SignalState:
    """路由需要的全部输入。每一项都能从事件日志里算出来。"""
    # 1) 首次合法率
    proposals_attempted: int = 0
    proposals_accepted_first_try: int = 0
    # 2) 重复率
    proposals_total: int = 0
    proposals_duplicate: int = 0
    # 3) 停滞
    stalled_rounds: int = 0
    # 4) 同类改动历史
    recent_change_kinds: list = field(default_factory=list)      # ['swap_model','tune_depth',...]
    historical_delta_by_kind: dict = field(default_factory=dict)  # {'swap_model': 0.03, ...}

    def first_try_pass_rate(self) -> float | None:
        if self.proposals_attempted <= 0:
            return None
        return self.proposals_accepted_first_try / self.proposals_attempted

    def duplicate_rate(self) -> float | None:
        if self.proposals_total <= 0:
            return None
        return self.proposals_duplicate / self.proposals_total


def classify_change_kind(prev: dict | None, curr: dict) -> str:
    """把一次改动归类，用于查"同类改动历史上有没有用"。"""
    if not prev:
        return 'baseline'
    if prev.get('model') != curr.get('model'):
        return 'swap_model'
    prev_params = prev.get('parameters') or {}
    curr_params = curr.get('parameters') or {}
    changed = {k for k in set(prev_params) | set(curr_params)
               if prev_params.get(k) != curr_params.get(k)}
    if not changed:
        return 'no_change'
    if changed & {'max_depth', 'min_samples_leaf', 'n_estimators', 'max_iter', 'learning_rate'}:
        return 'tune_capacity'
    if changed & {'C', 'class_weight'}:
        return 'tune_regularization'
    return 'tune_other'


def historical_delta(state: SignalState) -> float | None:
    """最近几次改动的平均历史涨幅。没有记录就返回 None（表示"不知道"）。"""
    deltas = [state.historical_delta_by_kind.get(kind) for kind in state.recent_change_kinds]
    deltas = [d for d in deltas if d is not None]
    if not deltas:
        return None
    return sum(deltas) / len(deltas)


def decide(state: SignalState) -> dict:
    """返回 {action, reason, signals}。action 取 'escalate' / 'continue'。

    规则严格按二德子给的写法，不额外发明阈值：
        停滞 >= 2 且 历史同类 Δ <= 0   → 升级
        首次合法率 < 0.5              → 升级
        否则                          → 继续本地
    """
    pass_rate = state.first_try_pass_rate()
    dup_rate = state.duplicate_rate()
    delta = historical_delta(state)

    signals = {
        'first_try_pass_rate': round(pass_rate, 3) if pass_rate is not None else None,
        'duplicate_rate': round(dup_rate, 3) if dup_rate is not None else None,
        'stalled_rounds': state.stalled_rounds,
        'historical_delta': round(delta, 4) if delta is not None else None,
        'recent_change_kinds': list(state.recent_change_kinds),
    }

    # 规则 1：卡住了，而且同类改动历史上没涨过
    if state.stalled_rounds >= 2 and delta is not None and delta <= 0:
        return {'action': 'escalate',
                'reason': (f'连续 {state.stalled_rounds} 轮没涨，'
                           f'同类改动历史平均 Δ={delta:+.4f}，本地再试大概率没用'),
                'signals': signals, 'rule': 'stalled_and_no_gain'}

    # 规则 2：模型连约束都过不了
    if pass_rate is not None and pass_rate < 0.5:
        return {'action': 'escalate',
                'reason': f'首次合法率 {pass_rate:.2%} < 50%，本地模型提不出合格方案',
                'signals': signals, 'rule': 'low_first_try_pass'}

    # 其它情况：继续本地
    if delta is None:
        reason = '还没有同类改动的历史数据，先让本地继续探索以积累记录'
    elif state.stalled_rounds < 2:
        reason = f'只连续 {state.stalled_rounds} 轮没涨，还没到阈值'
    else:
        reason = f'同类改动历史平均 Δ={delta:+.4f} > 0，本地还有空间'
    return {'action': 'continue', 'reason': reason,
            'signals': signals, 'rule': 'local_still_useful'}


# ---------------- 从事件日志构造 SignalState ----------------

def state_from_events(events: list[dict], experiments: list[dict]) -> SignalState:
    """把一轮研究的事件日志汇总成路由输入。

    只统计能算出来的事实，缺的留 None——不猜。
    """
    proposals = [e for e in events if e.get('event') == 'scientist_proposal']
    rejections = [e for e in events if e.get('event') == 'scientist_rejected']
    verdicts = [e for e in events if e.get('event') == 'reviewer_verdict']

    state = SignalState()
    # 首次合法率：一次就过的提案 / 所有尝试
    state.proposals_attempted = len(proposals) + len(rejections)
    state.proposals_accepted_first_try = len(proposals)
    state.proposals_total = len(proposals) + len(rejections)
    # 重复率：被拒原因里带 duplicate 的
    state.proposals_duplicate = sum(
        1 for r in rejections if 'duplicate' in str(r.get('reason', '')).lower())
    # 停滞轮数：连续 flat/regressed
    stalled = 0
    for v in reversed(verdicts):
        verdict = (v.get('verdict') or {}).get('verdict')
        if verdict in ('flat', 'regressed'):
            stalled += 1
        else:
            break
    state.stalled_rounds = stalled
    # 同类改动的历史 Δ：从已有实验里累计
    by_kind: dict[str, list[float]] = {}
    prev = None
    for exp in experiments:
        kind = classify_change_kind(prev, {'model': exp.get('model'),
                                           'parameters': exp.get('parameters') or {}})
        prev_score = (prev or {}).get('validation_score', (prev or {}).get('validation_f1'))
        curr_score = exp.get('validation_score', exp.get('validation_f1'))
        if prev_score is not None and curr_score is not None and kind != 'baseline':
            by_kind.setdefault(kind, []).append(float(curr_score) - float(prev_score))
        prev = exp
    state.historical_delta_by_kind = {
        kind: sum(vals) / len(vals) for kind, vals in by_kind.items() if vals}
    state.recent_change_kinds = list(state.historical_delta_by_kind.keys())[-3:]
    return state


if __name__ == '__main__':
    demo = SignalState(proposals_attempted=6, proposals_accepted_first_try=3,
                       proposals_total=6, proposals_duplicate=2, stalled_rounds=3,
                       recent_change_kinds=['tune_capacity', 'swap_model'],
                       historical_delta_by_kind={'tune_capacity': -0.01, 'swap_model': -0.03})
    import json
    print(json.dumps(decide(demo), ensure_ascii=False, indent=2))
