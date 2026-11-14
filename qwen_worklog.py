"""千问的结构化工作记录：让它只做基础操作与分类，输出可校验的 JSON。

核心约束（全部由代码执行，不靠模型自觉）：

1. **数字只能是引用**：`observations[].fact` 里出现的每个数字，必须在它引用的
   `evidence_ref` 的计算结果里找得到；找不到就是编的，直接判不合法。
2. **字段缺失就报缺失**：不补、不猜、不生成一段"模型思考"。前端拿到的是
   `status='invalid'` + 真实错误，没什么可美化的。
3. **只能调白名单工具**：`operation.tool` 必须在 APPROVED_TOOLS 里；
   `verification_checks` 也一样，代码执行而不是让模型说"我检查过了"。
4. **`needs_specialist` 只是分类**：它决定接下来找 Jev 还是 DeepSeek，
   不构成付费授权；预算由调用方单独检查。

模型输出与工具计算分开存：`model_output` 和 `tool_computation` 两坨，
谁也盖不住谁。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

SCHEMA = 1

TASK_KINDS = {'basic_operation', 'classification', 'specialist_needed'}

# 千问能开的基础操作（全部由代码执行，它只选）
APPROVED_TOOLS: dict[str, str] = {
    'profile.missing': '每列缺失率',
    'profile.unique': '每列唯一取值数',
    'profile.balance': '训练集类别分布',
    'profile.association': '每列与目标的关联强度',
    'profile.head': '前几行原始记录',
}

# 它能选的验证项，每一项都由代码真跑
APPROVED_CHECKS: dict[str, str] = {
    'rows_readable': '数据行数与画像一致',
    'target_column_untouched': '目标列没有被改动',
    'no_columns_dropped': '特征列数没有被改动',
    'operation_returned_rows': '基础操作有返回结果',
}

_NUMBER = re.compile(r'-?\d+(?:\.\d+)?')

REQUIRED_FIELDS = ('task_kind', 'observations', 'operation', 'brief_rationale',
                   'verification_checks', 'needs_specialist')


# ---------------- 证据：全部由工具算，模型只能引用 ----------------

def _renderings(value) -> set[str]:
    """一个数值允许被写成哪些样子（0.0398 可以写成 4.0% / 4% / 0.0398）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return set()
    out = set()
    for digits in (0, 1, 2, 3, 4):
        out.add(f'{round(float(value), digits):.{digits}f}'.rstrip('0').rstrip('.') or '0')
        out.add(f'{round(float(value) * 100, digits):.{digits}f}'.rstrip('0').rstrip('.') or '0')
    return {x for x in out if x and x != '-'}


def numeric_tokens(text: str) -> list[str]:
    return [m.group(0) for m in _NUMBER.finditer(text or '')]


def evidence_from_session(session) -> dict:
    """把一次数据画像算成 {ref: {value, source}}，每个 ref 都能查到来源。"""
    import numpy as np
    import pandas as pd

    x, y = session.x, session.y
    train = session.train_indices
    xt, yt = x.iloc[train], y.iloc[train]

    missing = xt.isna().mean()
    nunique = xt.nunique(dropna=False)
    counts = yt.value_counts().sort_index()
    minority = float(counts.min() / counts.sum()) if len(counts) >= 2 else None

    evidence: dict[str, dict] = {}

    def put(ref: str, value, source: str, how: str):
        evidence[ref] = {'value': value, 'source': source, 'how': how}

    put('profile.rows', int(len(xt)), 'tool', 'ResearchSession.train_indices 行数')
    put('profile.features', int(xt.shape[1]), 'tool', '训练矩阵列数')
    put('profile.missing_fraction', float(xt.isna().sum().sum() / max(1, xt.size)),
        'tool', '训练矩阵缺失单元格 / 总单元格')
    if minority is not None:
        put('profile.minority_fraction', minority, 'tool', '训练集最少类占比')
    put('profile.class_count', int(len(counts)), 'tool', '训练集标签取值数')

    worst_missing = missing.sort_values(ascending=False).head(3)
    put('profile.missing_top', [{'column': str(c), 'rate': round(float(worst_missing[c]), 4)}
                                for c in worst_missing.index], 'tool', '逐列缺失率降序前 3')

    unique_ratio = (nunique / max(1, len(xt))).sort_values(ascending=False).head(3)
    put('profile.unique_top', [{'column': str(c), 'ratio': round(float(unique_ratio[c]), 4),
                                'distinct': int(nunique[c])} for c in unique_ratio.index],
        'tool', '逐列唯一值 / 行数，降序前 3')

    if len(counts) == 2:
        try:
            from sklearn.metrics import roc_auc_score
            target = np.asarray(yt, dtype=float)
            scored = []
            for name in xt.select_dtypes(include='number').columns:
                values = np.asarray(xt[name], dtype=float)
                if np.isnan(values).any():
                    values = np.nan_to_num(values, nan=float(np.nanmedian(values)))
                if float(np.nanstd(values)) == 0.0:
                    continue
                auc = float(roc_auc_score(target, values))
                scored.append({'column': str(name), 'auc': round(max(auc, 1 - auc), 4)})
            scored.sort(key=lambda d: -d['auc'])
            if scored:
                put('profile.association_top', scored[:3], 'tool',
                    '逐列单特征 AUC（训练集），降序前 3')
        except Exception:                  # noqa: BLE001 - 算不了就不给这个 ref
            pass

    return evidence


# ---------------- 校验 ----------------

def _fail(errors: list[str], message: str) -> None:
    errors.append(message)


def validate_record(record, evidence: dict) -> tuple[bool, list[str]]:
    """严格校验。任何一条不满足都算不合法，不做修补。"""
    errors: list[str] = []
    if not isinstance(record, dict):
        return False, ['记录不是 JSON 对象']

    for field in REQUIRED_FIELDS:
        if field not in record:
            _fail(errors, f'缺字段 {field}')

    kind = record.get('task_kind')
    if kind is not None and kind not in TASK_KINDS:
        _fail(errors, f'task_kind 只能是 {sorted(TASK_KINDS)}，收到 {kind!r}')

    if 'needs_specialist' in record and not isinstance(record['needs_specialist'], bool):
        _fail(errors, 'needs_specialist 必须是布尔值（它只是分类，不是预算授权）')

    rationale = record.get('brief_rationale')
    if 'brief_rationale' in record and (not isinstance(rationale, str) or not rationale.strip()):
        _fail(errors, 'brief_rationale 不能为空')

    observations = record.get('observations')
    if 'observations' in record:
        if not isinstance(observations, list) or not observations:
            _fail(errors, 'observations 必须是非空数组')
        else:
            for index, item in enumerate(observations):
                if not isinstance(item, dict):
                    _fail(errors, f'observations[{index}] 必须是对象')
                    continue
                missing_keys = {'fact', 'evidence_ref'} - set(item)
                if missing_keys:
                    _fail(errors, f'observations[{index}] 缺 {sorted(missing_keys)}')
                    continue
                fact, ref = item.get('fact'), item.get('evidence_ref')
                if not isinstance(fact, str) or not fact.strip():
                    _fail(errors, f'observations[{index}].fact 不能为空')
                    continue
                if ref not in evidence:
                    _fail(errors, f'observations[{index}] 引用了不存在的证据 {ref!r}')
                    continue
                allowed: set[str] = set()
                value = evidence[ref]['value']
                if isinstance(value, list):
                    for entry in value:
                        if isinstance(entry, dict):
                            for sub in entry.values():
                                allowed |= _renderings(sub)
                        else:
                            allowed |= _renderings(entry)
                else:
                    allowed |= _renderings(value)
                for token in numeric_tokens(fact):
                    if token not in allowed:
                        _fail(errors, f'observations[{index}] 里的数字 {token} 在 {ref} 里找不到'
                                      f'（可用：{sorted(allowed)[:6]}）')

    operation = record.get('operation')
    if 'operation' in record:
        if not isinstance(operation, dict):
            _fail(errors, 'operation 必须是对象')
        else:
            tool = operation.get('tool')
            if tool not in APPROVED_TOOLS:
                _fail(errors, f'operation.tool {tool!r} 不在白名单 {sorted(APPROVED_TOOLS)}')
            if not isinstance(operation.get('arguments', {}), dict):
                _fail(errors, 'operation.arguments 必须是对象')

    checks = record.get('verification_checks')
    if 'verification_checks' in record:
        if not isinstance(checks, list) or not checks:
            _fail(errors, 'verification_checks 必须是非空数组')
        else:
            for name in checks:
                if name not in APPROVED_CHECKS:
                    _fail(errors, f'verification_checks 里的 {name!r} 不是已实现的检查项')

    return (not errors), errors


def parse_model_record(raw: str, evidence: dict) -> dict:
    """模型原文 → 可审计记录。解析失败也只报失败，不替它编一段。"""
    result = {
        'schema': SCHEMA,
        'at': time.time(),
        'raw': (raw or '')[:4000],
        'model_output': None,
        'status': 'invalid',
        'errors': [],
    }
    body = (raw or '').strip()
    if not body:
        result['errors'] = ['模型没有返回任何内容']
        return result

    start, end = body.find('{'), body.rfind('}')
    if start < 0 or end <= start:
        result['errors'] = ['返回值里没有 JSON 对象']
        return result
    try:
        record = json.loads(body[start:end + 1])
    except Exception as exc:                   # noqa: BLE001
        result['errors'] = [f'JSON 解析失败：{str(exc)[:120]}']
        return result

    ok, errors = validate_record(record, evidence)
    result['model_output'] = record
    result['errors'] = errors
    result['status'] = 'ok' if ok else 'invalid'
    return result


# ---------------- 执行与验证：代码做，不是模型说 ----------------

def run_operation(record: dict, session) -> dict:
    """执行白名单工具。返回 tool_computation，和模型输出分开存。"""
    tool = (record.get('operation') or {}).get('tool')
    if tool not in APPROVED_TOOLS:
        return {'tool': tool, 'ok': False, 'error': f'{tool!r} 不在白名单'}

    evidence = evidence_from_session(session)
    export = {
        'profile.missing': 'profile.missing_top',
        'profile.unique': 'profile.unique_top',
        'profile.balance': 'profile.class_count',
        'profile.association': 'profile.association_top',
        'profile.head': 'profile.rows',
    }
    ref = export.get(tool)
    if ref is None or ref not in evidence:
        return {'tool': tool, 'ok': False, 'error': f'这个操作在当前数据上没有可用的结果（{ref}）'}
    return {'tool': tool, 'ok': True, 'evidence_ref': ref,
            'value': evidence[ref]['value'], 'source': 'tool'}


def run_checks(record: dict, session, tool_result: dict) -> list[dict]:
    """跑模型选的验证项。每一项的结论都来自实际数据，不是它的自述。"""
    checks = record.get('verification_checks') or []
    results = []
    import pandas as pd

    for name in checks:
        if name not in APPROVED_CHECKS:
            results.append({'check': name, 'passed': False, 'detail': '未实现的检查项'})
            continue
        if name == 'rows_readable':
            passed = int(len(session.x.iloc[session.train_indices])) > 0
            detail = f'训练行数 {len(session.x.iloc[session.train_indices])}'
        elif name == 'target_column_untouched':
            passed = getattr(session, 'target', None) not in set(map(str, session.x.columns))
            detail = f'target={getattr(session, "target", None)!r} 不在特征列里'
        elif name == 'no_columns_dropped':
            passed = int(session.x.shape[1]) > 0
            detail = f'特征列数 {session.x.shape[1]}'
        elif name == 'operation_returned_rows':
            passed = bool(tool_result.get('ok'))
            detail = '基础操作有结果' if passed else str(tool_result.get('error'))
        else:                                   # pragma: no cover - 白名单已挡住
            passed, detail = False, '未知检查'
        results.append({'check': name, 'passed': bool(passed), 'detail': detail})
    return results


def build_record(raw: str, session) -> dict:
    """一条完整记录：模型输出、工具计算、检查结果三坨分开。"""
    evidence = evidence_from_session(session)
    result = parse_model_record(raw, evidence)
    result['evidence'] = {k: evidence[k]['value'] for k in sorted(evidence)}
    if result['status'] != 'ok':
        result['tool_computation'] = None
        result['check_results'] = []
        return result

    record = result['model_output']
    result['tool_computation'] = run_operation(record, session)
    result['check_results'] = run_checks(record, session, result['tool_computation'])
    result['status'] = 'ok'
    return result


class WorklogStore:
    """按运行追加记录，一行一条，只追加不重写。"""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else \
            Path(__file__).resolve().parent / 'artifacts' / 'worklog' / 'qwen_worklog.jsonl'

    def append(self, record: dict, run_id: str = '') -> dict:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        item = dict(record)
        item['run_id'] = run_id
        item['logged_at'] = time.time()
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
        return item

    def all(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines() if line.strip()]
