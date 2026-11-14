"""策略版本存储的验收测试：候选、差异、恢复、持久化、运行绑定。"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import strategy_store
from strategy_store import StrategyStore

BASE_CHANGES = {
    'classifier_prompt': '按类别分布选择指标。',
    'planning_prompt': '先看缺失，再选模型。',
    'candidate_selection_rules': [{'when': 'imbalanced', 'prefer': 'class_weight'}],
}


class StrategyStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='strategy-')
        self.path = Path(self.tmp) / 'strategy_store.json'

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def store(self) -> StrategyStore:
        return StrategyStore(self.path)

    # ---- 版本建立与候选 ----

    def test_bootstrap_creates_active_and_is_idempotent(self):
        store = self.store()
        first = store.bootstrap(BASE_CHANGES)
        second = store.bootstrap({'classifier_prompt': '另一套'})
        self.assertEqual(first['strategy_id'], 'strategy_001')
        self.assertEqual(first['status'], 'active')
        self.assertEqual(second['strategy_id'], 'strategy_001')
        self.assertEqual(store.active()['strategy_id'], 'strategy_001')
        self.assertEqual(len(store.versions()), 1)

    def test_propose_creates_candidate_without_touching_active(self):
        store = self.store()
        base = store.bootstrap(BASE_CHANGES)
        candidate = store.propose(
            {'planning_prompt': '先看泄漏列，再选模型。'},
            reason='treasury: 任务里出现事后字段，当前规划没查',
            evidence_run_ids=['run_001'],
        )
        self.assertEqual(candidate['status'], 'candidate')
        self.assertEqual(candidate['parent_id'], base['strategy_id'])
        self.assertEqual(store.active()['strategy_id'], base['strategy_id'])
        self.assertNotEqual(candidate['config_digest'], base['config_digest'])

    def test_propose_rejects_out_of_scope_fields(self):
        store = self.store()
        store.bootstrap(BASE_CHANGES)
        with self.assertRaises(ValueError):
            store.propose({'scoring_function': 'x'}, reason='想改评分')
        with self.assertRaises(ValueError):
            store.propose({'planning_prompt': 123}, reason='类型不对')
        with self.assertRaises(ValueError):
            store.propose({'planning_prompt': '一样'}, reason='   ')

    def test_propose_rejects_identical_to_parent(self):
        store = self.store()
        store.bootstrap(BASE_CHANGES)
        with self.assertRaises(ValueError):
            store.propose(dict(BASE_CHANGES), reason='原样复制')

    # ---- 差异 ----

    def test_diff_reports_only_changed_fields(self):
        store = self.store()
        base = store.bootstrap(BASE_CHANGES)
        candidate = store.propose({'planning_prompt': '先看泄漏列。'}, reason='补一步')
        diff = store.diff(base['strategy_id'], candidate['strategy_id'])
        self.assertEqual(diff['changed_fields'], ['planning_prompt'])
        self.assertFalse(diff['identical'])
        self.assertEqual(diff['changes'][0]['to'], '先看泄漏列。')
        self.assertEqual(diff['changes'][0]['from'], BASE_CHANGES['planning_prompt'])

    # ---- 切换的两道硬闸 ----

    def test_activate_requires_code_checks_and_jev(self):
        store = self.store()
        store.bootstrap(BASE_CHANGES)
        candidate = store.propose({'planning_prompt': '先看泄漏列。'}, reason='补一步')

        with self.assertRaises(PermissionError):
            store.activate(candidate['strategy_id'], checks={}, jev={'decision': 'upgrade'})
        with self.assertRaises(PermissionError):
            store.activate(candidate['strategy_id'],
                           checks={'legal_rate': True},
                           jev={'decision': 'keep'})
        with self.assertRaises(PermissionError):
            store.activate(candidate['strategy_id'],
                           checks={'legal_rate': False},
                           jev={'decision': 'upgrade'})

        activated = store.activate(candidate['strategy_id'],
                                   checks={'legal_rate': True, 'no_regression': True},
                                   jev={'decision': 'upgrade', 'source': 'jev'})
        self.assertEqual(activated['status'], 'active')
        self.assertEqual(store.active()['strategy_id'], candidate['strategy_id'])

        old = store.get('strategy_001')
        self.assertEqual(old['status'], 'retired')
        self.assertEqual(len(old['changes']), 3)          # 旧版本原样保留

    def test_events_record_every_switch(self):
        store = self.store()
        store.bootstrap(BASE_CHANGES)
        candidate = store.propose({'planning_prompt': '先看泄漏列。'}, reason='补一步')
        store.activate(candidate['strategy_id'], checks={'ok': True}, jev={'decision': 'upgrade'})
        kinds = [event['type'] for event in store.events()]
        self.assertEqual(kinds, ['bootstrap', 'propose', 'activate'])
        self.assertEqual(store.events()[-1]['previous_id'], 'strategy_001')

    # ---- 恢复旧版本 ----

    def test_restore_puts_old_version_back(self):
        store = self.store()
        store.bootstrap(BASE_CHANGES)
        candidate = store.propose({'planning_prompt': '先看泄漏列。'}, reason='补一步')
        store.activate(candidate['strategy_id'], checks={'ok': True}, jev={'decision': 'upgrade'})
        restored = store.restore('strategy_001', checks={'rollback_ok': True},
                                 jev={'decision': 'upgrade'}, reason='新策略退化')
        self.assertEqual(restored['strategy_id'], 'strategy_001')
        self.assertEqual(store.active()['strategy_id'], 'strategy_001')
        self.assertEqual(store.get(candidate['strategy_id'])['status'], 'retired')
        self.assertEqual(store.events()[-1]['type'], 'activate')
        self.assertEqual(store.events()[-1]['reason'], '新策略退化')

    def test_reject_keeps_candidate_out_of_service(self):
        store = self.store()
        store.bootstrap(BASE_CHANGES)
        candidate = store.propose({'planning_prompt': '先看泄漏列。'}, reason='补一步')
        rejected = store.reject(candidate['strategy_id'], '证据不足')
        self.assertEqual(rejected['status'], 'rejected')
        self.assertEqual(store.active()['strategy_id'], 'strategy_001')
        with self.assertRaises(ValueError):
            store.activate(candidate['strategy_id'], checks={'ok': True},
                           jev={'decision': 'upgrade'})

    # ---- 持久化与运行绑定 ----

    def test_records_survive_new_instance(self):
        first = self.store()
        first.bootstrap(BASE_CHANGES)
        candidate = first.propose({'planning_prompt': '先看泄漏列。'}, reason='补一步')
        first.activate(candidate['strategy_id'], checks={'ok': True}, jev={'decision': 'upgrade'})

        second = StrategyStore(self.path)                 # 模拟进程重启
        self.assertEqual(second.active()['strategy_id'], candidate['strategy_id'])
        self.assertEqual(len(second.versions()), 2)
        self.assertEqual([e['type'] for e in second.events()], ['bootstrap', 'propose', 'activate'])
        on_disk = json.loads(self.path.read_text(encoding='utf-8'))
        self.assertEqual(on_disk['active_id'], candidate['strategy_id'])

    def test_pin_is_immutable_while_store_moves_on(self):
        store = self.store()
        store.bootstrap(BASE_CHANGES)
        pinned = store.pin()
        candidate = store.propose({'planning_prompt': '先看泄漏列。'}, reason='补一步')
        store.activate(candidate['strategy_id'], checks={'ok': True}, jev={'decision': 'upgrade'})

        self.assertEqual(pinned['strategy_id'], 'strategy_001')
        self.assertEqual(pinned['changes'], BASE_CHANGES)      # 快照没被换掉
        self.assertTrue(store.verify_pin(pinned))
        self.assertNotEqual(pinned['strategy_id'], store.active()['strategy_id'])

    def test_empty_store_refuses_to_pin(self):
        with self.assertRaises(LookupError):
            self.store().pin()


if __name__ == '__main__':
    unittest.main()
