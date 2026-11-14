"""本地数据迭代测验：按顺序喂一批数据，记录路由如何从"没证据"变成"有证据"。

全部本地推理，零 API 花费。每一步：
  1. 用当前能力表判断该怎么派（explore / proven_capable / proven_incapable）
  2. 真的让本地模型提一次方案并试跑
  3. 结果写回能力表
  4. 记录指标：token、耗时、累计浪费、有证据比例

跑之前会快照能力表，可随时回退。
"""
import argparse
import json
import shutil
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents import Scientist
from capability_memory import CapabilityMemory
from executor import ResearchSession
from local_catalog import LocalCatalog
from research_loop import DEFAULT_ALLOWED

URL = 'http://127.0.0.1:11434/v1/chat/completions'
MODEL = 'qwen2.5:3b-instruct-q4_K_M'
ROOT = Path(__file__).resolve().parents[1]
MEMORY = ROOT / 'artifacts' / 'ml' / 'capability.sqlite3'
SNAPSHOTS = ROOT / 'artifacts' / 'evolution' / 'snapshots'
STATIC = ROOT / 'static' / 'evolution.json'
RAW = ROOT / 'artifacts' / 'evolution' / 'data'

BATCH = ['iris', 'wine', 'breast_cancer', 'ops_churn', 'customer_churn', 'sme_credit', 'sensor_fault']


def make_proposer(crippled, meter):
    def proposer(messages, timeout=120):
        body = {'model': MODEL, 'temperature': 0, 'messages': messages,
                'max_tokens': 60 if crippled else 400}
        req = urllib.request.Request(URL, data=json.dumps(body).encode('utf-8'),
                                     headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode('utf-8'))
        usage = data.get('usage') or {}
        meter['prompt'] += usage.get('prompt_tokens') or 0
        meter['completion'] += usage.get('completion_tokens') or 0
        return data['choices'][0]['message']['content']
    return proposer


def snapshot():
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    target = SNAPSHOTS / ('capability-' + stamp + '.sqlite3')
    if MEMORY.exists():
        shutil.copy2(MEMORY, target)
    return target


def main(reset=False, faulty=False, rounds=3):
    RAW.mkdir(parents=True, exist_ok=True)
    saved = snapshot()
    if reset and MEMORY.exists():
        MEMORY.unlink()
    memory = CapabilityMemory(MEMORY)
    catalog = LocalCatalog()

    steps = []
    wasted = 0
    tokens = 0
    seen = 0
    evidenced = 0
    delegated = 0
    index = 0

    for round_no in range(1, rounds + 1):
        crippled = faulty and round_no <= 2
        meter = {'prompt': 0, 'completion': 0}
        scientist = Scientist(make_proposer(crippled, meter), DEFAULT_ALLOWED)
        for key in BATCH:
            try:
                csv, target, _ = catalog.download(key)
            except Exception:
                continue
            path = RAW / (key + '.csv')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(csv.encode('utf-8'))
            session = ResearchSession(path, target)
            session.ensure_ready()
            profile = session.planning_context()

            verdict = memory.route(MODEL, 'model_search', profile)
            before = dict(meter)
            began = time.monotonic()
            error = None
            attempted = verdict['tier'] == 'low'
            ok = True
            if attempted:
                try:
                    proposal = scientist.propose([], profile)
                    session.run(proposal['model'], proposal.get('parameters') or {})
                except Exception as exc:
                    ok, error = False, type(exc).__name__
            else:
                delegated += 1
            seconds = time.monotonic() - began
            step_tokens = (meter['prompt'] - before['prompt']) + (meter['completion'] - before['completion'])
            memory.record(MODEL, 'model_search', ok, error, profile)

            index += 1
            seen += 1
            tokens += step_tokens
            if attempted and not ok:
                wasted += 1
            if verdict['reason'] != 'unproven_explore':
                evidenced += 1
            steps.append({'step': index, 'round': round_no, 'dataset': key,
                          'bucket': verdict.get('bucket'), 'decision': verdict['reason'],
                          'tier': verdict['tier'], 'attempted': attempted, 'ok': ok,
                          'error': error, 'seconds': round(seconds, 2),
                          'tokens': step_tokens, 'cumulative_tokens': tokens,
                          'cumulative_wasted': wasted, 'cumulative_delegated': delegated,
                          'evidenced_ratio': round(evidenced / seen * 100, 1)})
            print('%2d r%d %-14s %-16s %s %5.2fs %5dtok 浪费=%d 有证据=%.0f%%' % (
                index, round_no, key, verdict['reason'],
                ('PASS' if ok else 'FAIL') if attempted else 'DELEGATED',
                seconds, step_tokens, wasted, round(evidenced / seen * 100, 1)), flush=True)

    payload = {
        'generated_at': datetime.now().astimezone().isoformat(),
        'model': MODEL, 'snapshot': str(saved), 'rounds': rounds, 'faulty': faulty,
        'mode_label': '故障注入' if faulty else '正常',
        'steps': steps,
        'summary': {'tasks': len(steps), 'passed': sum(1 for s in steps if s['ok']),
                    'wasted_local_calls': wasted, 'delegated': delegated,
                    'tokens': tokens,
                    'final_evidenced_ratio': steps[-1]['evidenced_ratio'] if steps else 0},
        'notes': ['本地推理，零 API 花费；token 数取自 Ollama 返回的 usage',
                  '有证据比例 = 路由时能力表里已有该数据桶记录的次数占比',
                  '故障注入 = 前两轮把输出上限压到 60 token，制造真实截断失败',
                  'DELEGATED = 能力表判定本地做不了，直接转交，不再浪费一次调用',
                  '不做任何质量或泛化结论'],
    }
    STATIC.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n写入', STATIC)
    print(json.dumps(payload['summary'], ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--faulty', action='store_true')
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    main(args.reset, args.faulty, args.rounds)
