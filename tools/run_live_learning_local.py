"""Local secure-key launcher. Never writes the key to files or command history."""
import getpass
import os
import sys
import time
import json
import urllib.request
import urllib.error
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
os.chdir(root)
os.environ['LIGHT_MODEL_API_KEY'] = getpass.getpass('DeepSeek API Key (hidden): ').strip()
os.environ['LIGHT_MODEL_BASE_URL'] = 'https://api.deepseek.com/v1'
os.environ['LIGHT_MODEL_NAME'] = 'deepseek-chat'

from light_model import from_env, LightModelError
from cloud_provider import NoRedirect
try:
    client = from_env()
    # A successful unauthenticated GET does not verify authenticated POST.
    # Check the two stages separately without exposing the credential.
    models_request = urllib.request.Request(client.base_url + '/models', headers={
        'Authorization': 'Bearer ' + os.environ['LIGHT_MODEL_API_KEY'], 'Accept': 'application/json'})
    opener = urllib.request.build_opener(NoRedirect(), client._proxy_handler())
    print('Stage 1/2: authenticated model-list request...')
    with opener.open(models_request, timeout=15) as response:
        raw = response.read(65537)
    if len(raw) > 65536:
        raise LightModelError('Model-list response exceeds size limit.')
    models = json.loads(raw)
    available = [item.get('id') for item in models.get('data', []) if isinstance(item, dict)]
    print('Authentication accepted; advertised models:', ', '.join(str(m) for m in available[:16]))
    if available and client.model not in available:
        raise LightModelError('Configured model is not in the advertised list: ' + client.model)
    print('Stage 2/2: generation POST request...')
    client.max_calls = 1
    client([{'role': 'user', 'content': 'Reply with exactly: {"ok":true}'}], timeout=15)
except urllib.error.HTTPError as exc:
    status = exc.code
    exc.close()
    print('Authenticated model-list request failed: HTTP', status)
    print('No learning benchmark was started.')
    raise SystemExit(2)
except urllib.error.URLError as exc:
    reason = exc.reason
    print('Authenticated model-list connection failed:', type(reason).__name__, getattr(reason, 'reason', 'transport_error'))
    print('No learning benchmark was started.')
    raise SystemExit(2)
except (LightModelError, ValueError) as exc:
    print('Model connection preflight failed. No learning benchmark was started.')
    print(str(exc))
    raise SystemExit(2)

print('Model endpoint returned a response. Starting the bounded 18-call learning comparison.')
from tools.evaluate_live_learning import evaluate
output = root / 'artifacts' / time.strftime('live-learning-local-%Y%m%d-%H%M%S')
report = evaluate(output)
print('Report:', output / 'report.json')
raise SystemExit(0 if report.get('passed') else 2)
