"""用真 key 验证 JevGate：这是接通的唯一硬证据。"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_router import JevGate   # noqa: E402

key = os.environ.get('TYPESAFE_API_KEY', '').strip()
if not key:
    print('no TYPESAFE_API_KEY')
    sys.exit(2)

state = {
    'task': 't2_imbalanced / f1_minority',
    'trigger': 'validation_plateau',
    'local_calls_used': 4,
    'experiments_used': 4,
    'best_validation_score': 0.6475,
    'recent_scores': [0.2994, 0.6475, 0.2984, 0.4464],
    'remaining_cloud_calls': 1,
    'note': 'local planner stopped improving for 2 rounds; budget remains',
}

gate = JevGate.auto(api_key=key, threshold=0.75)
print('client:', type(gate.client).__name__)
result = gate(state, timeout=30)
print(json.dumps(result, ensure_ascii=False, indent=2))
ok = result.get('source') == 'jev' and isinstance(result.get('probability'), float)
sys.exit(0 if ok else 1)
