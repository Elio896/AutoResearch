"""One bounded live model verification, only if server credentials are configured."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from light_model import from_env
from agents import Scientist
from research_loop import DEFAULT_ALLOWED

client = from_env()
if client is None:
    print(json.dumps({'live_model_verified': False, 'reason': 'No server model credentials configured.', 'calls': 0}))
else:
    client.max_calls = 1
    scientist = Scientist(client, DEFAULT_ALLOWED)
    try:
        result = scientist.propose([], {'rows': 600, 'features': 2, 'class_count': 2,
                                        'selection_metric': 'f1_macro'}, timeout=8)
        print(json.dumps({'live_model_verified': True, 'proposal_model': result['model'],
                          'usage': client.usage_summary()}))
    except Exception as exc:
        print(json.dumps({'live_model_verified': False, 'error_type': type(exc).__name__,
                          'usage': client.usage_summary()}))
