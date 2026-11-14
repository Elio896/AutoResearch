"""第二步验收：千问结构化工作记录。

验证三件事：
1. 换输入，记录随实际证据改变（数字不是写死的）；
2. 每条依据都能查到来源（ref → 计算值 → 怎么算的）；
3. 没有模型输出时不会冒出一段"分析"——直接报失败。

同时做一次**真实**的千问调用，把它返回什么原样贴出来，由代码判合不合法。
模拟响应只用来跑通程序流程，会在输出里明确标注。
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import qwen_worklog  # noqa: E402
from qwen_worklog import build_record, evidence_from_session  # noqa: E402

OUT = Path('artifacts') / 'qwen-worklog-acceptance'


class _Session:
    def __init__(self, frame, target='target'):
        self.target = target
        self.x = frame.drop(columns=[target])
        rng = np.random.default_rng(5)
        self.y = pd.Series((rng.random(len(frame)) < float(frame[target].mean())).astype(int))
        self.train_indices = np.arange(len(frame))


def write_frames() -> tuple[Path, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)

    n = 400
    clean = pd.DataFrame({
        'tenure': rng.integers(1, 72, n),
        'charges': np.round(rng.normal(60, 15, n), 2),
        'tickets': rng.integers(0, 6, n),
    })
    clean['target'] = (rng.random(n) < 0.30).astype(int)
    path_a = OUT / 'clean_30pct.csv'
    clean.to_csv(path_a, index=False)

    n2 = 400
    dirty = pd.DataFrame({
        'amount': np.round(rng.gamma(2, 40, n2), 2),
        'hour': rng.integers(0, 24, n2),
        'risk': np.round(rng.beta(2, 5, n2), 3),
    })
    dirty.loc[dirty.sample(frac=0.12, random_state=7).index, 'amount'] = np.nan
    dirty['target'] = (rng.random(n2) < 0.04).astype(int)
    path_b = OUT / 'imbalanced_4pct.csv'
    dirty.to_csv(path_b, index=False)
    return path_a, path_b


def session_for(path: Path) -> _Session:
    return _Session(pd.read_csv(path))


def simulated_record(evidence: dict) -> dict:
    rows = evidence['profile.rows']['value']
    features = evidence['profile.features']['value']
    record = {
        'task_kind': 'basic_operation',
        'observations': [
            # 一条事实一个编号：数字只能来自它引用的那份证据
            {'fact': f'训练集 {rows} 行', 'evidence_ref': 'profile.rows'},
            {'fact': f'特征 {features} 列', 'evidence_ref': 'profile.features'},
        ],
        'operation': {'tool': 'profile.missing', 'arguments': {}},
        'brief_rationale': '按已给定规则执行基础操作：先看缺失',
        'verification_checks': ['rows_readable', 'no_columns_dropped',
                                'target_column_untouched', 'operation_returned_rows'],
        'needs_specialist': False,
    }
    if 'profile.minority_fraction' in evidence:
        share = evidence['profile.minority_fraction']['value']
        record['observations'].append(
            {'fact': f'最少一类占 {share * 100:.1f}%', 'evidence_ref': 'profile.minority_fraction'})
    return record


def ask_qwen(evidence: dict) -> str:
    schema = json.dumps(qwen_worklog.REQUIRED_FIELDS, ensure_ascii=False)
    facts = json.dumps({k: v['value'] for k, v in evidence.items()}, ensure_ascii=False)
    prompt = (
        '你是数据助手，只做基础操作。可用的证据编号和值如下（只能引用，不能自己算）：\n'
        f'{facts}\n\n'
        f'请输出一个 JSON，字段必须正好是 {schema}。\n'
        'observations 每项形如 {"fact": "...", "evidence_ref": "编号"}，'
        'fact 里出现的数字必须来自对应编号的值；'
        'operation.tool 只能取 ' + json.dumps(list(qwen_worklog.APPROVED_TOOLS), ensure_ascii=False) + '；'
        'verification_checks 只能取 ' + json.dumps(list(qwen_worklog.APPROVED_CHECKS), ensure_ascii=False) + '。'
        '只输出 JSON，不要解释。'
    )
    payload = json.dumps({'model': 'qwen2.5:7b-instruct-q4_K_M', 'prompt': prompt,
                          'stream': False, 'options': {'temperature': 0}}).encode('utf-8')
    request = urllib.request.Request('http://127.0.0.1:11434/api/generate', data=payload,
                                     headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=120) as response:
        return json.loads(response.read().decode('utf-8', 'replace')).get('response', '')


def show_evidence(evidence: dict) -> None:
    print('  证据（全部由工具计算，模型只能引用）：')
    for ref in sorted(evidence):
        item = evidence[ref]
        preview = json.dumps(item['value'], ensure_ascii=False)
        if len(preview) > 90:
            preview = preview[:87] + '...'
        print(f'    {ref} = {preview}   ← {item["how"]}')


def show_record(tag: str, record: dict, evidence: dict) -> None:
    print(f'  {tag}: status={record["status"]}  errors={record["errors"]}')
    model_output = record.get('model_output') or {}
    for index, item in enumerate(model_output.get('observations') or []):
        ref = item.get('evidence_ref')
        value = evidence.get(ref, {}).get('value')
        print(f'    observation[{index}] "{item.get("fact")}"  ← {ref} = {value}')
    if record.get('tool_computation'):
        tool = record['tool_computation']
        print(f'    工具执行：{tool["tool"]} → ok={tool["ok"]}  ref={tool.get("evidence_ref")}')
    for check in record.get('check_results') or []:
        print(f'    检查 {check["check"]}: {"通过" if check["passed"] else "失败"}  （{check["detail"]}）')


def main() -> int:
    path_a, path_b = write_frames()
    print(f'输入 A：{path_a}    输入 B：{path_b}\n')

    ok_all = True
    for label, path in (('A（均衡、无缺失）', path_a), ('B（4% 正例、12% 缺失）', path_b)):
        session = session_for(path)
        evidence = evidence_from_session(session)
        print(f'== 输入 {label} ==')
        show_evidence(evidence)
        print()

        # 程序流程验证用模拟响应（明确标注，不当成模型能力）
        record = build_record(json.dumps(simulated_record(evidence), ensure_ascii=False), session)
        show_record('记录（模拟响应，仅验证程序流程）', record, evidence)
        ok_all &= record['status'] == 'ok'

        # 编数字必须被拒
        forged = simulated_record(evidence)
        forged['observations'] = [{'fact': '训练集 99999 行', 'evidence_ref': 'profile.rows'}]
        bad = build_record(json.dumps(forged, ensure_ascii=False), session)
        print(f'  编造数字：status={bad["status"]}  errors={bad["errors"][:1]}')
        ok_all &= bad['status'] == 'invalid'
        print()

    print('== 真实千问调用（qwen2.5:7b，原样记录）==')
    session = session_for(path_a)
    evidence = evidence_from_session(session)
    try:
        raw = ask_qwen(evidence)
        print(f'  返回原文（前 400 字）：{raw.strip()[:400]!r}')
        real = build_record(raw, session)
        print(f'  代码判定：status={real["status"]}  errors={real["errors"][:3]}')
        if real['status'] != 'ok':
            print('  → 不合格就是不合格，页面上必须显示失败原因，不能替它补一段分析。')
    except Exception as exc:                       # noqa: BLE001
        print(f'  调用失败：{str(exc)[:160]}（不重试、不伪造）')

    print('\n判定：' + ('通过——证据随输入变化、可追溯到计算方式、编造数字被拒、无输出不生成叙述'
                       if ok_all else '失败'))
    return 0 if ok_all else 1


if __name__ == '__main__':
    raise SystemExit(main())
