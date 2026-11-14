"""第四步验收：DeepSeek 提候选——必须有假设、有否证条件、只动白名单、不动正式策略。

顺序：
1. 读第三步真实跑出来的失败报告当输入证据；
2. 先试**真实** DeepSeek（没 key 就直说不可用，不伪造调用）；
3. 用明确标注的模拟响应验证程序流程；
4. 展示三种被拒绝的情形。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from candidate_proposer import (CandidateRejected, propose_from_failures,  # noqa: E402
                                simulated_client)
from strategy_store import StrategyStore  # noqa: E402

WORK = Path('artifacts') / 'candidate-acceptance'
FAILURE_REPORT = Path('artifacts') / 'failure-modes-acceptance' / 'report.json'

BASE = {
    'classifier_prompt': '按类别分布选择指标。',
    'planning_prompt': '先看缺失与唯一值，再选基线模型。',
    'candidate_selection_rules': [{'when': 'imbalanced', 'prefer': 'class_weight'}],
}


def load_failures() -> dict:
    if not FAILURE_REPORT.is_file():
        raise SystemExit(f'缺少第三步的失败报告：{FAILURE_REPORT}（先跑 tools/failure_modes_acceptance.py）')
    data = json.loads(FAILURE_REPORT.read_text(encoding='utf-8'))
    return data['stuck_run']


def show_candidate(version: dict) -> None:
    print(f"  strategy_id={version['strategy_id']}  status={version['status']}  "
          f"parent={version['parent_id']}  simulated={version.get('simulated')}")
    print(f"  target_failure_types={version['target_failure_types']}")
    print('  hypothesis：')
    for key, value in (version['hypothesis'] or {}).items():
        print(f'    {key}: {value}')
    print(f"  reason（拼给审计看的一行）：{version['reason'][:160]}")


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)

    failures = load_failures()
    print(f"== 输入失败报告：{len(failures['records'])} 条记录，"
          f"verdict={failures['verdict']} ==")
    for record in failures['records']:
        print(f"  - {record['failure_type']} refs={record['event_refs']} "
              f"measurements={json.dumps(record['measurements'], ensure_ascii=False)[:100]}")

    store = StrategyStore(WORK / 'strategy_store.json')
    active = store.bootstrap(BASE, reason='基线')
    print(f"\n== 当前正式策略：{active['strategy_id']} ==")
    print(f"  planning_prompt={active['changes']['planning_prompt']!r}")

    print('\n== 1. 真实 DeepSeek 调用 ==')
    key = os.environ.get('DEEPSEEK_API_KEY')
    if not key:
        print('  不可用：环境里没有 DEEPSEEK_API_KEY（不伪造调用，也不假装已经想过）')
    else:
        from cloud_provider import DeepSeekClient
        client = DeepSeekClient(key)
        try:
            version = propose_from_failures(store, failures, client, timeout=90)
            print('  真实调用成功：')
            show_candidate(version)
        except CandidateRejected as exc:
            print(f'  真实调用返回的候选被拒：{exc}')

    print('\n== 2. 模拟响应（仅验证程序流程，不代表 DeepSeek 能力）==')
    version = propose_from_failures(store, failures, simulated_client(), simulated=True)
    show_candidate(version)

    active_now = store.active()
    print(f"\n  正式策略仍是：{active_now['strategy_id']}（内容未变："
          f"{active_now['changes'] == BASE}）")
    diff = store.diff(active['strategy_id'], version['strategy_id'])
    print(f"  版本差异：{diff['changed_fields']}")

    print('\n== 3. 应当被拒绝的三种情形 ==')
    cases = {
        '越权字段（想改评分函数）': {
            'changes': {'scoring_function': 'accuracy'},
            'hypothesis': {'what': 'w', 'why': 'y', 'expected_improvement': 'e',
                           'falsifier': 'f', 'side_effects': 's'},
        },
        '没有否证条件': {
            'changes': {'planning_prompt': '改一点点'},
            'hypothesis': {'what': 'w', 'why': 'y', 'expected_improvement': 'e',
                           'falsifier': '  ', 'side_effects': 's'},
        },
        '一次改两个字段': {
            'changes': {'planning_prompt': '改 A', 'classifier_prompt': '改 B'},
            'hypothesis': {'what': 'w', 'why': 'y', 'expected_improvement': 'e',
                           'falsifier': 'f', 'side_effects': 's'},
        },
    }
    for label, payload in cases.items():
        try:
            propose_from_failures(store, failures, simulated_client(payload), simulated=True)
            print(f'  !! {label} 竟然通过，这是 bug')
            return 1
        except CandidateRejected as exc:
            print(f'  {label} → 拒绝：{exc}')

    print('\n== 4. 证据不足时不允许提候选 ==')
    thin = {'insufficient': True, 'records': [], 'insufficient_reasons': ['实验只有 1 次']}
    try:
        propose_from_failures(store, thin, simulated_client())
        print('  !! 证据不足竟然能提，这是 bug')
        return 1
    except CandidateRejected as exc:
        print(f'  拒绝：{exc}')

    versions = store.versions()
    print(f'\n最终：版本 {len(versions)} 个，事件 {[e["type"] for e in store.events()]}')
    ok = (len(versions) == 2 and store.active()['strategy_id'] == active['strategy_id'])
    print('判定：' + ('通过——候选只是候选，正式策略没被覆盖，越权与缺否证条件都被挡住'
                     if ok else '失败'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
