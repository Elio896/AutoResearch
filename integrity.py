"""Training-only feature-risk signals, not a proof of general leakage absence."""
import json
from pathlib import Path
import re


# 事后性词汇：这些名字暗示「结果出来之后才有的值」。
# 真实案例：fraud 数据里的 chargeback_filed（只有立案后才是 1），
# 名字不含 id、也不是唯一整数、与标签相关 0.87 没到 0.98——
# 老规则全漏，于是系统报出 0.955 的假好成绩。
POST_HOC_PATTERN = re.compile(
    r'(^|[_\s-])('
    r'chargeback|charge_back|refund|reversal|settled|settlement|'
    r'filed|confirmed|verified|approved|declined|resolved|closed|'
    r'outcome|result|final|label|verdict|decision|status_after|'
    r'post_?hoc|after_?the_?fact'
    r')([_\s-]|$)')


def inspect_features(features, labels, dataset_hash, target):
    import pandas as pd
    registry_path = Path(__file__).with_name('known_leaks.json')
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    known = {item['column']: item for item in registry
             if item['dataset_sha256'] == dataset_hash and item['target'] == target}
    findings = []
    encoded = pd.Series(pd.factorize(labels)[0], index=features.index)
    for column in features.columns:
        series = features[column]
        signals = []
        if re.search(r'(^|[_\s-])(id|identifier|index|rowid|serial)([_\s-]|$)', column.lower()):
            signals.append('identifier_name')
        # 事后字段：名字本身就说明它是结果出来之后才有的
        if POST_HOC_PATTERN.search(column.lower()):
            signals.append('post_hoc_name')
        numeric = pd.to_numeric(series, errors='coerce')
        valid = numeric.dropna()
        unique = series.nunique(dropna=True) / max(len(series), 1)
        if len(valid) >= 20 and len(valid) == len(series) and unique >= 0.98:
            integer_like = (valid % 1 == 0).all()
            if integer_like and (valid.is_monotonic_increasing or valid.is_monotonic_decreasing):
                signals.append('unique_ordered_integer')
            ordered_labels = labels.loc[numeric.sort_values(kind='stable').index]
            transitions = int((ordered_labels != ordered_labels.shift()).sum()) - 1
            if transitions == labels.nunique() - 1:
                signals.append('unique_numeric_separates_training_classes')
        if series.notna().all() and (series.astype(str) == labels.astype(str)).all():
            signals.append('matches_training_target')
        if labels.nunique() == 2 and numeric.notna().sum() >= 20 and numeric.nunique() > 1:
            correlation = numeric.corr(encoded)
            # 0.98 太严：事后二值列与标签相关常在 0.85~0.95 之间。
            # 阈值放到 0.85 只标「候选」，仍然要求人审，不自动排除。
            if pd.notna(correlation) and abs(correlation) >= 0.85:
                signals.append('strong_training_association')
            # 二值列 + 与标签几乎一致：典型的「结果字段」
            if numeric.nunique() == 2 and pd.notna(correlation) and abs(correlation) >= 0.80:
                agree = (numeric.round() == encoded).mean()
                if agree >= 0.90 or agree <= 0.10:
                    signals.append('binary_matches_label_outcome')
        if column in known:
            findings.append({'column': column, 'level': 'confirmed',
                             'signals': signals, 'evidence': known[column]['source']})
        elif signals:
            findings.append({'column': column, 'level': 'candidate', 'signals': signals,
                             'evidence': 'Heuristic on training rows only; requires provenance review, not proof of leakage.'})
    return findings


def apply_review(features, findings, decisions):
    if type(decisions) is not dict:
        raise ValueError('Feature decisions must be a dictionary.')
    confirmed = {item['column'] for item in findings if item['level'] == 'confirmed'}
    excluded = []
    for column, decision in decisions.items():
        if column not in features.columns or type(decision) is not dict or set(decision) != {'action', 'reason'}:
            raise ValueError('Each decision needs an existing feature, action and reason.')
        if decision['action'] not in ('exclude', 'keep') or not isinstance(decision['reason'], str) or not 3 <= len(decision['reason'].strip()) <= 500:
            raise ValueError('Review action must be exclude/keep with a meaningful reason (3..500 characters).')
        if column in confirmed and decision['action'] != 'exclude':
            raise ValueError('Confirmed leakage must be explicitly excluded.')
        if decision['action'] == 'exclude':
            excluded.append(column)
    unresolved = sorted({item['column'] for item in findings} - set(decisions))
    cleaned = features.drop(columns=excluded)
    if not len(cleaned.columns):
        raise ValueError('At least one feature must remain.')
    report = {'version': 1, 'scope': 'training_only', 'findings': findings,
              'decisions': decisions, 'excluded_columns': sorted(excluded),
              'unresolved_columns': unresolved, 'blocked': bool(unresolved),
              'limitation': 'A cleared review is not proof that all leakage has been detected.'}
    return cleaned, report
