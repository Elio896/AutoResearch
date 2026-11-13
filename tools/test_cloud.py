"""验证云端 DeepSeek 通道（本地卡住时才用）。"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloud_provider import DeepSeekClient   # noqa: E402

key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
if not key:
    print('no DEEPSEEK_API_KEY')
    sys.exit(2)

client = DeepSeekClient(key)
try:
    response = client([
        {'role': 'system', 'content': 'Return only JSON.'},
        {'role': 'user', 'content': 'Return {"ok": true, "who": "deepseek"}'},
    ], timeout=60)
except Exception as exc:                      # noqa: BLE001
    print('FAILED:', type(exc).__name__, str(exc)[:300])
    sys.exit(1)

if isinstance(response, str):
    content = response
    usage = {}
else:
    content = (response.get('message') or {}).get('content') or ''
    usage = response.get('usage') or {}
print('raw:', content[:300])
print('usage:', json.dumps(usage, ensure_ascii=False))
sys.exit(0 if content.strip() else 1)
