"""真调本地千问，把它的原话打出来——证明那三句话是模型说的还是我编的。"""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

PROFILE = {
    'rows': 150, 'features': 4, 'class_count': 3,
    'minority_fraction': 0.333, 'imbalanced': False,
    'risks': [],
}
CONTEXT = {'missing_fraction': 0.0}


def ask(model: str, base: str = 'http://127.0.0.1:11434') -> dict:
    prompt = (
        '你是数据研究员。先判断这份数据要做的事里，哪些是基础的数据处理'
        '（清洗/填补/编码/重采样），哪些需要人的判断（选模型/定指标/诊断失败）。'
        '用三句话说完，中文，口语，不要给方案。\n'
        + json.dumps({'profile': PROFILE, 'risks': []}, ensure_ascii=False)
    )
    body = json.dumps({
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'options': {'temperature': 0.3, 'num_predict': 300},
    }).encode('utf-8')
    req = urllib.request.Request(
        base + '/api/chat', data=body,
        headers={'Content-Type': 'application/json'}, method='POST')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=180) as resp:
        payload = json.loads(resp.read().decode('utf-8', 'replace'))
    content = ((payload.get('message') or {}).get('content') or '').strip()
    return {'content': content,
            'eval_count': payload.get('eval_count'),
            'total_ms': round((payload.get('total_duration') or 0) / 1e6)}


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else 'qwen2.5:7b-instruct-q4_K_M'
    print(f'model: {model}')
    try:
        result = ask(model)
    except Exception as exc:                # noqa: BLE001
        print('调用失败:', exc)
        return 1
    print(f"tokens: {result['eval_count']}  用时: {result['total_ms']} ms")
    print('--- 模型原话 ---')
    print(result['content'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
