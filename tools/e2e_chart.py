"""端到端：走完三步 + 连点三次，最后取服务端渲染的 SVG 检查有没有点。

这是给"图到底出不出来"这个问题一个不靠猜的答案。
"""
import json
import sys
import urllib.request
from pathlib import Path

BASE = 'http://127.0.0.1:8002'
CSV = Path('data/iris.csv').read_text(encoding='utf-8')


def call(path, payload=None, timeout=300):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if payload is None:
        with opener.open(BASE + path, timeout=timeout) as resp:
            return resp.read().decode('utf-8', 'replace')
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(
        BASE + path, data=body,
        headers={'Content-Type': 'application/json'}, method='POST')
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def main():
    analyzed = call('/demo/analyze', {
        'csv_text': CSV, 'target': 'target',
        'metric': 'f1_macro', 'pos_label': '1',
    })
    sid = analyzed['session_id']
    print('session:', sid)
    print('千问来源:', analyzed['thought'].get('source'))

    call('/demo/approve', {
        'session_id': sid,
        'overrides': {'metric': 'f1_macro',
                      'baseline_model': 'logistic_regression',
                      'max_experiments': 6},
    })

    for i in range(3):
        step = call('/demo/step', {'session_id': sid})
        print(f"  第 {i + 1} 次: rounds={step['rounds']} "
              f"best={step['summary']['best_score']:.4f} done={step['done']}")

    svg = call(f'/demo/chart.svg?session_id={sid}')
    circles = svg.count('<circle')
    polyline = svg.count('<polyline')
    print(f'\n服务端 SVG: {len(svg)} 字节, {circles} 个点, {polyline} 条线')
    if circles >= 3 and polyline >= 1:
        print('结果: 图有数据 ✓')
        return 0
    print('结果: 图是空的 ✗')
    return 1


if __name__ == '__main__':
    sys.exit(main())
