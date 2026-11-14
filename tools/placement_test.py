"""摸底测试 v2：直接调用生产 Scientist，测的就是线上那份提示词。

对每份数据：提一个方案 → 用真实校验跑一次 → 记进能力表（按数据桶）。
纯本地推理，零花费。
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents import Scientist
from capability_memory import CapabilityMemory
from executor import ResearchSession
from local_catalog import LocalCatalog
from research_loop import DEFAULT_ALLOWED

URL = 'http://127.0.0.1:11434/v1/chat/completions'
MODEL = 'qwen2.5:3b-instruct-q4_K_M'
OUT = Path(__file__).resolve().parent
MEMORY = Path(__file__).resolve().parents[1] / 'artifacts' / 'ml' / 'capability.sqlite3'
RAW = OUT / 'data' / 'placement'


def proposer(messages, timeout=120):
    body = {'model': MODEL, 'temperature': 0, 'messages': messages, 'max_tokens': 400}
    req = urllib.request.Request(URL, data=json.dumps(body).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))['choices'][0]['message']['content']


def main(repeats=2):
    RAW.mkdir(parents=True, exist_ok=True)
    memory = CapabilityMemory(MEMORY)
    catalog = LocalCatalog()
    scientist = Scientist(proposer, DEFAULT_ALLOWED)
    rows = []
    for repeat in range(repeats):
        for record in catalog.records():
            key = record['id']
            try:
                csv, target, _ = catalog.download(key)
            except Exception:
                continue
            path = RAW / (key + '.csv')
            path.write_bytes(csv.encode('utf-8'))
            try:
                session = ResearchSession(path, target)
                session.ensure_ready()
            except Exception as exc:
                print('%-14s 跳过 %s' % (key, type(exc).__name__))
                continue
            profile = session.planning_context()
            began = time.monotonic()
            error = None
            proposal = None
            try:
                proposal = scientist.propose([], profile)
                session.run(proposal['model'], proposal.get('parameters') or {})
                ok = True
            except Exception as exc:
                ok, error = False, type(exc).__name__
            memory.record(MODEL, 'model_search', ok, error, profile)
            rows.append({'dataset': key, 'repeat': repeat, 'ok': ok, 'error': error,
                         'seconds': round(time.monotonic() - began, 1),
                         'proposal': {k: proposal[k] for k in ('model', 'parameters') if k in (proposal or {})} if proposal else None})
            print('%-14s #%d %s %5.1fs %s' % (key, repeat + 1,
                                              'PASS' if ok else 'FAIL ' + str(error),
                                              time.monotonic() - began,
                                              (proposal or {}).get('model', '')), flush=True)
    (OUT / 'placement-results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    passed = sum(1 for r in rows if r['ok'])
    print('\n合计 %d/%d 通过' % (passed, len(rows)))
    for row in memory.summary(12):
        print('  %-34s %-14s %d/%d' % (row['bucket'], row['category'], row['successes'], row['attempts']))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
