"""最小可检测效应（MDE）：判断"这点提升是不是噪声"。

来自二德子（2026-11-14）：

    对候选 c 和当前最优 b，用同一组 S 个种子各跑一次：
        d_i = score_c(seed_i) − score_b(seed_i)      # 成对差值
        mean_d = mean(d_1..d_S)
        SE     = std(d) / sqrt(S)
        MDE    = t(0.95, S−1) × SE                   # S=5 → 2.132；S=3 → 2.920

        improved   if mean_d >  MDE
        regressed  if mean_d < −MDE
        flat       otherwise

为什么成对：候选和最优在**同一个种子**上比，把数据划分带来的波动抵消掉，
剩下的才是"改动本身"的效果。

为什么 S=5 不用 S=3：一次 fit 只要 0.05–0.2 秒，5 个种子不到 1 秒；
而 S=3 时 t=2.92，只能发现"巨大提升"，会把真实改进误判成 flat。

对称性：差得少也算 flat，不回滚——省得系统在噪声上反复横跳。
"""
from __future__ import annotations

import math

# t 分布临界值
# 单侧 95%：用于「只关心一个方向」的判断
T_CRITICAL_ONE_SIDED = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015,
                        6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
                        11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761, 15: 1.753,
                        20: 1.725, 30: 1.697, 50: 1.676, 100: 1.660}

# 双侧 95%（等价单侧 97.5%）：同时判「升」和「降」时必须用这个，
# 否则两个方向各用 95%，合起来的误判率约 10%，不是 5%。
# 猫（章鱼猫猫头）2026-11-14 指出这一点，是对的。
T_CRITICAL_TWO_SIDED = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
                        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
                        20: 2.086, 30: 2.042, 50: 2.009, 100: 1.984}

# 兼容旧名字：默认仍取单侧表，但实际判定走 two_sided=True 的双侧表
T_CRITICAL = T_CRITICAL_ONE_SIDED

DEFAULT_SEEDS = 5


def t_critical(dof: int, two_sided: bool = True) -> float:
    """取 95% 临界值；表里没有就用最近的下界，绝不返回比实际更小的。

    two_sided=True 默认——因为我们同时判 improved 和 regressed 两个方向。
    """
    if dof <= 0:
        return float('inf')
    table = T_CRITICAL_TWO_SIDED if two_sided else T_CRITICAL_ONE_SIDED
    if dof in table:
        return table[dof]
    keys = sorted(table)
    for key in keys:
        if key >= dof:
            return table[key]
    return table[keys[-1]]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    """样本标准差（除以 n-1）。1 个点时返回 0。"""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def compare(candidate_scores: list[float], best_scores: list[float],
            min_effect: float = 0.0, two_sided: bool = True) -> dict:
    """成对比较两组同种子分数，返回判定和全部中间量。

    candidate_scores / best_scores 必须**一一对应**（同一个种子跑出来的）。

    two_sided=True（默认）：因为同时判 improved 和 regressed 两个方向，
    必须用双侧临界值，否则合计误判率约 10%。
    """
    if len(candidate_scores) != len(best_scores):
        raise ValueError('成对比较需要同样数量的种子分数')
    n = len(candidate_scores)
    if n == 0:
        raise ValueError('至少需要一个种子')

    deltas = [c - b for c, b in zip(candidate_scores, best_scores)]
    mean_d = mean(deltas)
    std_d = sample_std(deltas)
    se = std_d / math.sqrt(n) if n > 0 else 0.0
    t_val = t_critical(n - 1, two_sided=two_sided) if n > 1 else float('inf')
    mde = t_val * se if n > 1 else float('inf')

    # 只有 1 个种子时无法估计噪声，直接按"不显著"处理，不许假装能判
    if n <= 1:
        return {
            'verdict': 'flat',
            'delta': round(mean_d, 6),
            'delta_vs_best': round(mean_d, 6),
            'delta_significant': False,
            'mde': None,
            'seed_mean': round(mean(candidate_scores), 6),
            'seed_std': round(sample_std(candidate_scores), 6),
            'per_seed': [round(v, 6) for v in candidate_scores],
            'deltas': [round(d, 6) for d in deltas],
            'seeds': n,
            'note': '只有一个种子，无法估计噪声——按 flat 处理',
        }

    # min_effect 是"就算显著，差得这么少也不值得动"的额外门槛
    threshold = max(mde, min_effect)
    if mean_d > threshold:
        verdict = 'improved'
    elif mean_d < -threshold:
        verdict = 'regressed'
    else:
        verdict = 'flat'

    return {
        'verdict': verdict,
        'delta': round(mean_d, 6),
        'delta_vs_best': round(mean_d, 6),
        'delta_significant': verdict != 'flat',
        'mde': round(mde, 6),
        'threshold_used': round(threshold, 6),
        'seed_mean': round(mean(candidate_scores), 6),
        'seed_std': round(sample_std(candidate_scores), 6),
        'best_seed_mean': round(mean(best_scores), 6),
        'best_seed_std': round(sample_std(best_scores), 6),
        'per_seed': [round(v, 6) for v in candidate_scores],
        'best_per_seed': [round(v, 6) for v in best_scores],
        'deltas': [round(d, 6) for d in deltas],
        'seeds': n,
        't_critical': t_val,
        'two_sided': two_sided,
        'note': (f'{n} 个种子，成对差值的 95% {"双侧" if two_sided else "单侧"}区间 = '
                 f'{mean_d:+.4f} ± {mde:.4f}'),
    }


def required_seeds(min_detectable: float, std_d: float, max_seeds: int = 30) -> int:
    """反过来问：要发现 min_detectable 这么小的差异，至少需要几个种子。

    用于解释"为什么这条路发现不了微小提升"。
    """
    if std_d <= 0:
        return 1
    for n in range(2, max_seeds + 1):
        if t_critical(n - 1) * std_d / math.sqrt(n) <= min_detectable:
            return n
    return max_seeds


if __name__ == '__main__':
    # 演示：候选平均高 0.002，但每个种子的差有正有负
    cand = [0.951, 0.948, 0.955, 0.949, 0.952]
    base = [0.949, 0.947, 0.953, 0.948, 0.951]
    result = compare(cand, base)
    for key in ('verdict', 'delta_vs_best', 'mde', 'delta_significant',
                'seed_mean', 'seed_std', 'per_seed'):
        print(f'{key:18s} {result[key]}')
