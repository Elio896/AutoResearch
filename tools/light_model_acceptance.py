"""验收：把「本地千问」换成轻量模型 API 之后，链路是不是真的走 API、标签是不是真的变了。

用一个本地假服务冒充 API（真实 HTTP、真实超时与错误路径），**不花钱、不碰真 key**。
"""
from __future__ import annotations

import json
import os
import sys
import threading
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import demo_flow  # noqa: E402

REPLY = '这批数据没有缺失，但 txn_id 每行都不一样，是行标识；目标类只占 4%，不能看准确率。'


class _Handler(BaseHTTPRequestHandler):
    calls: list = []

    def do_POST(self):                       # noqa: N802
        length = int(self.headers.get('Content-Length') or 0)
        body = json.loads(self.rfile.read(length) or b'{}')
        type(self).calls.append(body)
        payload = {'choices': [{'message': {'content': REPLY}}],
                   'usage': {'prompt_tokens': 120, 'completion_tokens': 32}}
        raw = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


def make_csv() -> str:
    rng = np.random.default_rng(3)
    n = 160
    frame = pd.DataFrame({
        'amount': np.round(rng.gamma(2, 30, n), 2),
        'hour': rng.integers(0, 24, n),
        'target': (rng.random(n) < 0.05).astype(int),
    })
    return frame.to_csv(index=False)


def main() -> int:
    server = HTTPServer(('127.0.0.1', 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_port}/v1'
    csv_text = make_csv()

    print('== 1. 配了 key：应该走 API ==')
    os.environ['LIGHT_MODEL_API_KEY'] = 'sk-local-mock-not-real'
    os.environ['LIGHT_MODEL_BASE_URL'] = base
    os.environ['LIGHT_MODEL_NAME'] = 'mock-small'
    os.environ['LIGHT_MODEL_MAX_CALLS'] = '3'
    try:
        result = demo_flow.start(csv_text, 'target', 'f1_macro', '1', use_local_model=True)
    finally:
        for name in ('LIGHT_MODEL_API_KEY', 'LIGHT_MODEL_BASE_URL',
                     'LIGHT_MODEL_NAME', 'LIGHT_MODEL_MAX_CALLS'):
            os.environ.pop(name, None)

    thought = result['thought']
    usage = result.get('model_usage') or {}
    print(f"  model_source={result['model_source']}  descriptor={result['model_descriptor']}")
    print(f"  attribution={thought.get('attribution')}  model={thought.get('model')}")
    print(f"  模型原文（前 40 字）：{(thought.get('raw') or '')[:40]!r}")
    print(f"  调用次数={usage.get('calls')}  token={usage.get('prompt_tokens')}/{usage.get('completion_tokens')}  "
          f"cost_cny={usage.get('cost_cny')}  cost_basis={usage.get('cost_basis')}")
    print(f'  假服务实际收到 {len(_Handler.calls)} 次 POST，'
          f'model={"mock-small" if _Handler.calls and _Handler.calls[0]["model"] == "mock-small" else "?"}')

    api_ok = (result['model_source'] == 'api_model'
              and thought.get('attribution') == 'model'
              and REPLY[:10] in (thought.get('raw') or '')
              and usage.get('calls', 0) >= 1
              and usage.get('cost_cny') is None)

    print('\n== 2. 没有 key：不许假装调过 ==')
    saved = demo_flow._local_proposer_if_available
    demo_flow._local_proposer_if_available = lambda model=None: None    # 假装 Ollama 也没起
    try:
        fallback = demo_flow.start(csv_text, 'target', 'f1_macro', '1', use_local_model=True)
    finally:
        demo_flow._local_proposer_if_available = saved
    print(f"  model_source={fallback['model_source']}  attribution={fallback['thought'].get('attribution')}")
    print(f"  model_usage={fallback.get('model_usage')}")
    print(f"  原文：{fallback['thought'].get('raw')}")
    fallback_ok = (fallback['model_source'] == 'rules'
                   and fallback['thought'].get('attribution') == 'program'
                   and fallback.get('model_usage') is None)

    server.shutdown()
    ok = api_ok and fallback_ok
    print('\n判定：' + ('通过——配 key 走 API 且标签为「轻量模型 API」、记账记 token 且成本未知；'
                       '没 key 退回程序规则并如实标注' if ok else '失败'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
