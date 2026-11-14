"""复现妈妈报的三个问题，定位到底哪儿断了。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import demo_flow

CSV = Path('data/iris.csv').read_text(encoding='utf-8')

# 1) 千问的思考长什么样
session = demo_flow.start(CSV, 'target', 'f1_macro', '1')
print('=== 千问的思考 ===')
print(json.dumps(session['thought'], ensure_ascii=False, indent=2)[:600])
print('=== 建议 ===')
print(json.dumps(session['suggested'], ensure_ascii=False, indent=2)[:300])

sid = session['session_id']
demo_flow.approve(sid, {
    'metric': 'f1_macro',
    'baseline_model': 'logistic_regression',
    'max_experiments': 8,
})

# 2) 连点五次，看每步返回什么
print('\n=== 连续点五次 ===')
for i in range(5):
    try:
        r = demo_flow.step(sid)
        print(f'第 {i+1} 次: rounds={r["rounds"]} done={r["done"]} '
              f'series={[round(x, 4) for x in r["best_series"]]}')
    except Exception as exc:
        print(f'第 {i+1} 次失败: {type(exc).__name__}: {exc}')
