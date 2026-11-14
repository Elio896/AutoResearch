"""Decide whether a task is hard enough to justify one expensive diagnostic.

Design borrowed from open-source prompt routers (heuristic complexity scoring as in
alexbeattie/model-router and Mattbusel/llm-router, budget-aware cascades as in
hybrid-model-router), adapted to tabular research: the signals are data-profile and
execution feedback, not prompt embeddings, because we have no preference dataset to
train RouteLLM-style routers on.
"""

MIN_SCORE = 0.5
SCORE_GATE = 0.75


def signals(profile, goal, best_score=None):
    """Return (score 0..1, list of human-readable reasons)."""
    profile = profile or {}
    text = goal or ''
    score = 0.0
    reasons = []

    classes = profile.get('class_count')
    if isinstance(classes, int) and classes >= 4:
        score += 0.25
        reasons.append('%d 分类' % classes)

    minority = profile.get('minority_fraction')
    if isinstance(minority, (int, float)) and minority < 0.12:
        score += 0.20
        reasons.append('少数类仅 %.1f%%' % (minority * 100))

    missing = profile.get('missing_fraction')
    if isinstance(missing, (int, float)) and missing > 0.05:
        score += 0.15
        reasons.append('缺失率 %.1f%%' % (missing * 100))

    features = profile.get('features')
    if isinstance(features, int) and features >= 20:
        score += 0.10
        reasons.append('%d 个特征' % features)

    rows = profile.get('rows')
    if isinstance(rows, int) and rows >= 5000:
        score += 0.05
        reasons.append('%d 行' % rows)

    for word, weight in (('诊断', 0.20), ('失败', 0.10), ('不稳定', 0.10), ('多试几轮', 0.10)):
        if word in text:
            score += weight
            reasons.append('问题提到「%s」' % word)

    if isinstance(best_score, (int, float)) and best_score < SCORE_GATE:
        score += 0.25
        reasons.append('首轮验证分只有 %.3f' % best_score)

    return min(score, 1.0), reasons


def should_escalate(profile, goal, best_score, *, expert_calls, max_expert_calls, fits, max_fits):
    """One bounded high-tier diagnostic per run, and only while the result is weak.

    A hard dataset that already scores well does not need an expensive second opinion;
    the point of the gate is to catch plateaus, not to reward complexity.
    """
    if expert_calls >= max_expert_calls or fits >= max_fits:
        return None
    if not isinstance(best_score, (int, float)) or best_score >= SCORE_GATE:
        return None
    score, reasons = signals(profile, goal, best_score)
    if score < MIN_SCORE:
        return None
    return {'reason': 'data_difficulty', 'score': round(score, 3),
            'signals': reasons[:6], 'first_score': best_score}
