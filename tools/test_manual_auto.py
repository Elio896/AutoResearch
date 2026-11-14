"""验证手动/自动两种迭代模式。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import demo_flow

CSV = Path('data/iris.csv').read_text(encoding='utf-8')

session = demo_flow.start(CSV, 'target', 'f1_macro', '1')
sid = session['session_id']
demo_flow.approve(sid, {
    'metric': 'f1_macro',
    'baseline_model': 'logistic_regression',
    'max_experiments': 8,
})
print('approved:', sid)

# 手动：点三次
for i in range(3):
    result = demo_flow.step(sid)
    print(f'step {i + 1}: rounds={result["rounds"]} '
          f'best={result["summary"]["best_score"]:.4f} '
          f'done={result["done"]} series={result["best_series"]}')

# 自动：跑 2 代（受预算上限约束）
result = demo_flow.auto(sid, 2)
print(f'auto: rounds={result["rounds"]} requested={result["requested"]} '
      f'capped_to={result["capped_to"]} stop={result["stop_reason"]}')
