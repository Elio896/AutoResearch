"""端到端验证：/demo/analyze 现在是否真的调了本地千问。"""
import json
import sys
import urllib.request
from pathlib import Path

BASE = 'http://127.0.0.1:8002'
CSV = Path('data/iris.csv').read_text(encoding='utf-8')


def post(path, payload):
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(
        BASE + path, data=body,
        headers={'Content-Type': 'application/json'}, method='POST')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=180) as resp:
        return json.loads(resp.read().decode('utf-8'))


def main():
    data = post('/demo/analyze', {
        'csv_text': CSV, 'target': 'target',
        'metric': 'f1_macro', 'pos_label': '1',
    })
    thought = data.get('thought') or {}
    print('source:', thought.get('source'))
    print('model :', thought.get('model'))
    print('--- 千问原话 ---')
    print(thought.get('text', '')[:500])
    print('--- 拆出来的两堆 ---')
    for x in thought.get('can_do') or []:
        print('  能做:', x)
    for x in thought.get('cannot_do') or []:
        print('  做不了:', x.replace('**', ''))
    return 0 if thought.get('source') == 'local_model' else 1


if __name__ == '__main__':
    sys.exit(main())
