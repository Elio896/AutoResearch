"""第一步验收：策略版本的建立、差异、拒绝切换、启用、重启后仍在、回滚。

跑法：
    .venv\\Scripts\\python.exe tools\\strategy_acceptance.py
输出全部是真实结果，脚本失败会直接抛异常，不会打印假的"通过"。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy_store import StrategyStore  # noqa: E402

STORE = Path('artifacts') / 'strategy-acceptance' / 'strategy_store.json'

BASE = {
    'classifier_prompt': '按类别分布选择指标；不平衡时用少数类 F1。',
    'planning_prompt': '先看缺失与唯一值，再选基线模型。',
    'candidate_selection_rules': [
        {'when': 'imbalanced', 'prefer': 'class_weight'},
        {'when': 'has_missing', 'prefer': 'imputer'},
    ],
}


def main() -> int:
    if STORE.parent.exists():
        shutil.rmtree(STORE.parent)
    STORE.parent.mkdir(parents=True, exist_ok=True)

    store = StrategyStore(STORE)
    print('== 1. 建立初始策略 ==')
    base = store.bootstrap(BASE, reason='当前硬编码行为的基线')
    print(f"  {base['strategy_id']}  status={base['status']}  digest={base['config_digest']}")

    print('== 2. 提一个候选（只改 planning_prompt） ==')
    candidate = store.propose(
        {'planning_prompt': '先查事后字段与 ID 列，再定指标和基线。'},
        reason='连续三次运行都保留了事后字段，规划里没有排雷步骤',
        evidence_run_ids=['run_001', 'run_002'],
    )
    print(f"  {candidate['strategy_id']}  status={candidate['status']}  parent={candidate['parent_id']}")
    print(f"  正式策略仍是：{store.active()['strategy_id']}")

    print('== 3. 版本差异 ==')
    diff = store.diff(base['strategy_id'], candidate['strategy_id'])
    print(f"  改动字段：{diff['changed_fields']}")
    for item in diff['changes']:
        print(f"    - {item['field']}: {item['from']!r} → {item['to']!r}")

    print('== 4. 没有证据不能切换 ==')
    for label, checks, jev in (
        ('代码检查为空', {}, {'decision': 'upgrade'}),
        ('Jev 说保留', {'legal_rate': True}, {'decision': 'keep'}),
        ('代码检查有 false', {'legal_rate': False}, {'decision': 'upgrade'}),
    ):
        try:
            store.activate(candidate['strategy_id'], checks=checks, jev=jev)
            print(f'  !! {label} 竟然通过了，这是 bug')
            return 1
        except PermissionError as exc:
            print(f'  {label} → 拒绝：{exc}')
    print(f"  正式策略未被改动：{store.active()['strategy_id']}")

    print('== 5. 两项都过才切换 ==')
    activated = store.activate(
        candidate['strategy_id'],
        checks={'legal_rate': True, 'no_regression': True, 'cost_within_budget': True},
        jev={'decision': 'upgrade', 'source': 'jev', 'confidence': 0.71},
        by='acceptance-demo',
        reason='新任务上合法输出率不降、重复率下降',
    )
    print(f"  active={activated['strategy_id']}  status={activated['status']}")
    print(f"  旧版本状态：{store.get(base['strategy_id'])['status']}（内容仍完整：{len(store.get(base['strategy_id'])['changes'])} 个字段）")

    print('== 6. 模拟进程重启 ==')
    reopened = StrategyStore(STORE)
    print(f"  重新打开后 active={reopened.active()['strategy_id']}")
    print(f"  版本数={len(reopened.versions())}  事件数={len(reopened.events())}")
    print(f"  事件序列：{[e['type'] for e in reopened.events()]}")

    print('== 7. 回滚到旧版本 ==')
    rolled = reopened.restore(
        base['strategy_id'],
        checks={'rollback_trigger': True},
        jev={'decision': 'upgrade', 'source': 'jev'},
        by='acceptance-demo',
        reason='新策略在第 4 个任务上合法输出率跌破阈值',
    )
    print(f"  active={rolled['strategy_id']}（回到基线）")
    print(f"  被回滚的候选状态：{reopened.get(candidate['strategy_id'])['status']}")
    print(f"  候选内容没有被删除：{reopened.get(candidate['strategy_id'])['changes']['planning_prompt']!r}")

    print('== 8. 运行绑定：pin 之后库里换版本，快照不动 ==')
    pinned = reopened.pin()
    other = reopened.propose({'planning_prompt': '再改一版试试。'}, reason='演示用')
    reopened.activate(other['strategy_id'], checks={'ok': True},
                      jev={'decision': 'upgrade'}, reason='演示启用')
    print(f"  pin 的是 {pinned['strategy_id']}，库里现在是 {reopened.active()['strategy_id']}")
    print(f"  快照内容仍是旧版：{pinned['changes']['planning_prompt']!r}")

    print('\n全部步骤执行完毕，下面是磁盘上的最终状态：')
    print(f"  文件：{STORE.resolve()}")
    print(f"  active_id={reopened.active()['strategy_id']}  版本={len(reopened.versions())}  事件={len(reopened.events())}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
