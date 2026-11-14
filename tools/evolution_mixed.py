"""混合难度迭代测验：一半数据本地能做，一半做不好 → 学会转交。

- 6 份难度递进的数据（g1_easy … g6_extreme）
- 判定标准是**质量**不是报错：首轮验证分 < 0.6 视为本地做不好
- 每轮 6 个任务，跑 3 轮；前两轮试，第三轮按能力表转交

不需要人为制造故障。跑之前快照能力表，可回退。
"""
import argparse
import json
import shutil
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents import Scientist
from capability_memory import CapabilityMemory
from executor import ResearchSession
from research_loop import DEFAULT_ALLOWED

URL = 'http://127.0.0.1:11434/v1/chat/completions'
MODEL = 'qwen2.5:3b-instruct-q4_K_M'
ROOT = Path(__file__).resolve().parents[1]
MEMORY = ROOT / 'artifacts' / 'ml' / 'capability.sqlite3'
SNAPSHOTS = ROOT / 'artifacts' / 'evolution' / 'snapshots'
STATIC = ROOT / 'static' / 'evolution.json'
GRADED = ROOT / 'data' / 'gen' / 'graded'
QUALITY_GATE = 0.6

MODES = {
    'mixed': [('g1_easy', 'easy'), ('g2_mild', 'easy'), ('g3_medium', 'medium'),
              ('g4_hard', 'hard'), ('g5_harder', 'hard'), ('g6_extreme', 'hard')],
}


def proposer(messages, timeout=120):
    body = {'model': MODEL, 'temperature': 0, 'messages': messages, 'max_tokens': 400}
    req = urllib.request.Request(URL, data=json.dumps(body).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode('utf-8'))
    usage = data.get('usage') or {}
    proposer.meter['prompt'] += usage.get('prompt_tokens') or 0
    proposer.meter['completion'] += usage.get('completion_tokens') or 0
    return data['choices'][0]['message']['content']


proposer.meter = {'prompt': 0, 'completion': 0}


def snapshot():
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    target = SNAPSHOTS / ('capability-' + stamp + '.sqlite3')
    if MEMORY.exists():
        shutil.copy2(MEMORY, target)
    return target


def main(reset=True, rounds=3, mode='mixed'):
    saved = snapshot()
    if reset and MEMORY.exists():
        MEMORY.unlink()
    memory = CapabilityMemory(MEMORY)
    scientist = Scientist(proposer, DEFAULT_ALLOWED)
    tasks = MODES[mode]

    steps = []
    tokens = 0
    seen = evidenced = delegated = low_quality = 0
    index = 0

    for round_no in range(1, rounds + 1):
        for name, difficulty in tasks:
            path = GRADED / (name + '.csv')
            if not path.exists():
                continue
            session = ResearchSession(path, 'label')
            session.ensure_ready()
            profile = session.planning_context()

            verdict = memory.route(MODEL, 'model_search', profile)
            before = dict(proposer.meter)
            began = time.monotonic()
            attempted = verdict['tier'] == 'low'
            ok, score, error = True, None, None
            if attempted:
                try:
                    proposal = scientist.propose([], profile)
                    result = session.run(proposal['model'], proposal.get('parameters') or {})
                    score = result.get('validation_f1')
                    if score is not None and score < QUALITY_GATE:
                        ok, error = False, 'below_quality_gate'
                        low_quality += 1
                except Exception as exc:
                    ok, error = False, type(exc).__name__
            else:
                delegated += 1
            seconds = time.monotonic() - began
            step_tokens = (proposer.meter['prompt'] - before['prompt']) + (proposer.meter['completion'] - before['completion'])
            if attempted:
                # 只有真的试过才写能力表；转交不是成功，不能污染统计
                memory.record(MODEL, 'model_search', ok, error, profile)

            index += 1
            seen += 1
            tokens += step_tokens
            if verdict['reason'] != 'unproven_explore':
                evidenced += 1
            steps.append({'step': index, 'round': round_no, 'dataset': name,
                          'difficulty': difficulty, 'bucket': verdict.get('bucket'),
                          'decision': verdict['reason'], 'tier': verdict['tier'],
                          'attempted': attempted, 'ok': ok, 'error': error,
                          'score': None if score is None else round(score, 3),
                          'seconds': round(seconds, 2), 'tokens': step_tokens,
                          'cumulative_tokens': tokens,
                          'evidenced_ratio': round(evidenced / seen * 100, 1)})
            print('%2d r%d %-11s %-7s %-16s %s score=%s %.2fs %4dtok 有证据=%.0f%%' % (
                index, round_no, name, difficulty, verdict['reason'],
                ('PASS' if ok else 'FAIL') if attempted else 'DELEGATED',
                ('—' if score is None else '%.3f' % score), seconds, step_tokens,
                steps[-1]['evidenced_ratio']), flush=True)

    payload = {
        'generated_at': datetime.now().astimezone().isoformat(),
        'model': MODEL, 'snapshot': str(saved), 'rounds': rounds,
        'faulty': False, 'mode_label': '混合难度（质量判据 %.1f）' % QUALITY_GATE,
        'quality_gate': QUALITY_GATE,
        'steps': steps,
        'summary': {'tasks': len(steps), 'passed': sum(1 for s in steps if s['ok']),
                    'wasted_local_calls': low_quality + sum(1 for s in steps if s['attempted'] and not s['ok'] and s['error'] != 'below_quality_gate'),
                    'delegated': delegated, 'low_quality': low_quality, 'tokens': tokens,
                    'final_evidenced_ratio': steps[-1]['evidenced_ratio'] if steps else 0},
        'notes': ['本地推理，零 API 花费',
                  '判定看质量：首轮验证分低于 %.1f 记为本地做不好（不是报错）' % QUALITY_GATE,
                  'DELEGATED = 能力表判定本地做不好，直接转交，不再消耗一次本地调用',
                  '数据由 artifacts/acceptance-20261114/make_graded_data.py 生成，难度递进',
                  '不做任何质量或泛化结论'],
    }
    STATIC.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n写入', STATIC)
    print(json.dumps(payload['summary'], ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--mode', default='mixed')
    args = parser.parse_args()
    main(args.reset, args.rounds, args.mode)
