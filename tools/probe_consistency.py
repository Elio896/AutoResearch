"""测方案 C：同一问题让模型答 3 次，看答案一致性能不能当置信度。

关键：必须给 temperature > 0，否则模型对同样输入永远给同样答案，
一致性恒为 1.0，等于没有信号。

这个脚本对同一个 (数据画像 + 历史) 让 7B 答 5 次，看：
    - 有多少次答案不同
    - 不同答案之间的"距离"（模型族不同 / 参数不同）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator import LocalQwenClient   # noqa: E402

BASE = 'http://127.0.0.1:11434'
MODEL = 'qwen2.5:7b-instruct-q4_K_M'

PROFILE = {
    'rows': 5000, 'features': 20, 'class_count': 2,
    'minority_fraction': 0.077, 'imbalanced': True,
    'risks': ['class_imbalance'], 'selection_metric': 'f1_minority',
}
HISTORY = [
    {'model': 'logistic_regression', 'parameters': {'class_weight': 'balanced'}, 'score': 0.2994},
    {'model': 'random_forest', 'parameters': {'n_estimators': 100}, 'score': 0.5426},
]


def make_prompt() -> list[dict]:
    return [
        {'role': 'system', 'content':
         'Propose ONE classifier experiment. Return only a JSON object with '
         'model, parameters, hypothesis, expected_direction, falsifier, confidence_before.'},
        {'role': 'user', 'content':
         f'Allowed: logistic_regression (C, max_iter, class_weight); '
         f'random_forest (n_estimators, max_depth, min_samples_leaf, class_weight); '
         f'hist_gradient_boosting (max_iter, max_depth, min_samples_leaf, learning_rate, class_weight)\n'
         f'Data profile: {json.dumps(PROFILE)}\n'
         f'Validation history: {json.dumps(HISTORY)}\n'
         'Never repeat an experiment.'},
    ]


def ask(n: int = 5) -> list[dict]:
    """问 n 次。每次用不同的 temperature（0.2 / 0.6 / 1.0 轮换）制造多样性。"""
    import urllib.request

    answers = []
    temps = [0.2, 0.6, 1.0, 0.8, 0.4]
    for i in range(n):
        temp = temps[i % len(temps)]
        body = json.dumps({
            'model': MODEL, 'messages': make_prompt(), 'stream': False,
            'format': 'json',
            'options': {'temperature': temp, 'num_predict': 220},
        }).encode('utf-8')
        req = urllib.request.Request(
            BASE + '/api/chat', data=body,
            headers={'Content-Type': 'application/json'}, method='POST')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode('utf-8', 'replace'))
            content = ((payload.get('message') or {}).get('content') or '').strip()
            parsed = json.loads(content)
            answers.append({'temp': temp, 'model': parsed.get('model'),
                            'parameters': parsed.get('parameters'),
                            'confidence': parsed.get('confidence_before')})
        except Exception as exc:                # noqa: BLE001
            answers.append({'temp': temp, 'error': str(exc)[:120]})
    return answers


def summarize(answers: list[dict]) -> dict:
    ok = [a for a in answers if 'model' in a]
    if not ok:
        return {'error': '所有调用都失败', 'answers': answers}
    models = [a['model'] for a in ok]
    unique_models = len(set(models))
    # 参数完全一致的组数
    param_keys = [json.dumps(a.get('parameters') or {}, sort_keys=True) for a in ok]
    unique_params = len(set(param_keys))
    confidences = [a.get('confidence') for a in ok if a.get('confidence') is not None]
    return {
        'n_calls': len(answers),
        'n_ok': len(ok),
        'unique_models': unique_models,
        'unique_parameter_sets': unique_params,
        'model_agreement': round(models.count(max(set(models), key=models.count)) / len(ok), 3),
        'confidence_values': confidences,
        'confidence_spread': (max(confidences) - min(confidences)) if len(confidences) > 1 else 0,
        'answers': answers,
    }


if __name__ == '__main__':
    print('=== 方案 C：同一问题问 5 次，看答案多样性 ===')
    answers = ask(5)
    report = summarize(answers)
    print(json.dumps({k: v for k, v in report.items() if k != 'answers'},
                     ensure_ascii=False, indent=2))
    print('\n--- 每次答了什么 ---')
    for a in answers:
        if 'error' in a:
            print(f"  temp={a['temp']}: 失败 {a['error']}")
        else:
            print(f"  temp={a['temp']}: {a['model']} {json.dumps(a['parameters'])} conf={a['confidence']}")

    out = Path('artifacts/consistency_probe.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nsaved:', out)
