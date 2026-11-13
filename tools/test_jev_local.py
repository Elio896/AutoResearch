"""验证 JevGate.auto()：没有官方 key 时走本地 JEV，且给出真实判断。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_router import JevGate, LocalJevClient   # noqa: E402

state = {
    'task': 't2_imbalanced / f1_minority',
    'trigger': 'validation_plateau',
    'local_calls_used': 4,
    'experiments_used': 4,
    'best_validation_score': 0.6620,
    'recent_scores': [0.2994, 0.6620, 0.4310, 0.2984],
    'note': 'local planner stopped improving for 2 rounds; budget remains 1 cloud call',
}

gate = JevGate.auto()          # 不传 key -> 本地
print('client:', type(gate.client).__name__)
try:
    direct = gate.client(state, timeout=60)
    print('direct:', json.dumps(direct, ensure_ascii=False))
except Exception as exc:                      # noqa: BLE001 - 这里就是要看原始异常
    print('direct FAILED:', type(exc).__name__, str(exc)[:300])
result = gate(state, timeout=60)
print(json.dumps(result, ensure_ascii=False, indent=2))
sys.exit(0 if result.get('source') in ('jev', 'local_qwen') else 1)
