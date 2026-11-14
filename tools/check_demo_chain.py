"""验证演示页走的整条链：自带鸢尾花 → 五角色 → 结果。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient   # noqa: E402

from api import create_app                   # noqa: E402


def iris_csv() -> str:
    """和 demo.html 里内嵌的那份一致（前 60 行足够跑）。"""
    path = Path('data/samples.csv')
    if path.is_file():
        return path.read_text(encoding='utf-8')
    # 兜底：手写一份小样本
    rows = ['sepal_length,sepal_width,petal_length,petal_width,species']
    for i in range(20):
        rows.append(f'{5.0 + i * 0.05:.1f},3.4,{1.4 + i * 0.01:.1f},0.2,setosa')
        rows.append(f'{6.0 + i * 0.05:.1f},2.9,{4.5 + i * 0.02:.1f},1.4,versicolor')
        rows.append(f'{6.5 + i * 0.05:.1f},3.0,{5.8 + i * 0.02:.1f},2.1,virginica')
    return '\n'.join(rows)


client = TestClient(create_app())
response = client.post('/research/five-roles', json={
    'csv_text': iris_csv(),
    'target': 'species',
    'metric': 'f1_macro',
    'pos_label': '1',
    'max_experiments': 3,
    'offline': True,
})
print('HTTP', response.status_code)
if response.status_code != 201:
    print(response.text[:400])
    sys.exit(1)
data = response.json()
s = data['summary']
print('status:', s['status'], '| stop:', s['stop_reason'])
print('best:', s['best_model'], round(s['best_score'], 4) if s['best_score'] else None)
print('versions:', s.get('evoundo_versions'), 'rollbacks:', s.get('evoundo_rollbacks'))
print('experiments:', len(data.get('experiments') or []))
