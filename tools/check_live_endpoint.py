"""验证现场演示端点：离线模式（不依赖 Ollama）能不能跑通整条链。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient   # noqa: E402

from api import create_app                   # noqa: E402

app = create_app()
client = TestClient(app)

csv_text = Path('bench/tasks/t2_imbalanced/data.csv').read_text(encoding='utf-8')[:200000]
response = client.post('/research/five-roles', json={
    'csv_text': csv_text,
    'target': 'target',
    'metric': 'f1_minority',
    'pos_label': '1',
    'max_experiments': 3,
    'offline': True,
})
print('HTTP', response.status_code)
if response.status_code != 201:
    print(response.text[:500])
    sys.exit(1)

data = response.json()
summary = data.get('summary') or {}
print('best:', summary.get('best_model'), summary.get('best_score'))
print('versions:', summary.get('evoundo_versions'),
      'rollbacks:', summary.get('evoundo_rollbacks'))
print('offline:', data.get('offline'))
print('events:', len(data.get('events') or []))
