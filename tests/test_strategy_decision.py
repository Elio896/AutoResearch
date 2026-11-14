"""第六步测试：Jev 只说选择，代码说了算——两关都过才切换。"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from strategy_decision import code_checks, decide, normalize_decision
from strategy_store import StrategyStore

ACTIVE = {
    'classifier_prompt': '按类别分布选择指标。',
    'planning_prompt': '先看缺失。',
    'candidate_selection_rules': [],
}
CANDIDATE = {'planning_prompt': '先查重，再定指标。'}


def eval_report(verdict='upgrade', **overrides) -> dict:
    checks = overrides.pop('checks', [
        {'check': 'min_tasks', 'passed': True},
        {'check': 'legal_output_rate', 'passed': True},
        {'check': 'duplicate_rate', 'passed': True},
        {'check': 'demonstrated_improvement', 'passed': True},
    ])
    report = {
        'verdict': verdict,
        'reason': '全部条件通过' if verdict == 'upgrade' else '未通过',
        'checks': checks,
        'old': {'legal_output_rate': 1.0, 'duplicate_rate': 0.5, 'mean_best_score': 0.70},
        'new': {'legal_output_rate': 1.0, 'duplicate_rate': 0.0, 'mean_best_score': 0.70},
    }
    report.update(overrides)
    return report


def gate(answer='upgrade', ok=True, model='jev-test'):
    calls = []

    def _gate(state, questions):
        calls.append({'state': state, 'questions': questions})
        if not ok:
            return {'ok': False, 'error': 'no_key'}
        return {'ok': True, 'model': model,
                'answers': {'strategy_decision': answer}}

    _gate.calls = calls
    return _gate


class StoreFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='decision-')
        self.store = StrategyStore(Path(self.tmp) / 'store.json')
        self.active = self.store.bootstrap(ACTIVE, reason='基线')
        self.candidate = self.store.propose(CANDIDATE, reason='治重复',
                                            evidence_run_ids=['run_1'])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class CodeGateTests(StoreFixture):
    def test_failed_eval_blocks_before_asking_jev(self):
        spy = gate('upgrade')
        result = decide(self.store, self.candidate['strategy_id'],
                        eval_report(verdict='keep'), spy)
        self.assertEqual(result['decision'], 'keep')
        self.assertFalse(result['applied'])
        self.assertEqual(result['source'], 'code')
        self.assertEqual(spy.calls, [], 'Jev 不该被调用')
        self.assertEqual(self.store.active()['strategy_id'], self.active['strategy_id'])
        self.assertEqual(self.store.events()[-1]['type'], 'decision_keep')

    def test_missing_eval_blocks(self):
        result = decide(self.store, self.candidate['strategy_id'], {}, gate())
        self.assertEqual(result['decision'], 'keep')
        self.assertFalse(result['applied'])

    def test_code_checks_translate_eval_report(self):
        checks, problems = code_checks(eval_report())
        self.assertTrue(all(checks.values()), checks)
        self.assertEqual(problems, [])

        checks, problems = code_checks(eval_report(verdict='keep'))
        self.assertFalse(checks['eval_verdict_upgrade'])
        self.assertTrue(problems)


class JevGateTests(StoreFixture):
    def test_jev_upgrade_switches(self):
        result = decide(self.store, self.candidate['strategy_id'],
                        eval_report(), gate('upgrade'))
        self.assertTrue(result['applied'])
        self.assertEqual(self.store.active()['strategy_id'], self.candidate['strategy_id'])
        event = self.store.events()[-1]
        self.assertEqual(event['type'], 'activate')
        self.assertEqual(event['previous_id'], self.active['strategy_id'])
        self.assertEqual(event['jev']['decision'], 'upgrade')
        self.assertTrue(all(event['checks'].values()))

    def test_jev_keep_does_not_switch(self):
        result = decide(self.store, self.candidate['strategy_id'],
                        eval_report(), gate('keep'))
        self.assertFalse(result['applied'])
        self.assertEqual(result['decision'], 'keep')
        self.assertEqual(self.store.active()['strategy_id'], self.active['strategy_id'])

    def test_jev_defer_keeps_candidate_pending(self):
        result = decide(self.store, self.candidate['strategy_id'],
                        eval_report(), gate('defer'))
        self.assertFalse(result['applied'])
        self.assertEqual(result['decision'], 'defer')
        self.assertEqual(self.store.get(self.candidate['strategy_id'])['status'], 'candidate')
        self.assertEqual(self.store.events()[-1]['type'], 'decision_defer')

    def test_jev_unavailable_keeps_old_strategy(self):
        result = decide(self.store, self.candidate['strategy_id'], eval_report(), None)
        self.assertFalse(result['applied'])
        self.assertEqual(result['source'], 'jev_unavailable')
        self.assertEqual(self.store.active()['strategy_id'], self.active['strategy_id'])

    def test_jev_error_keeps_old_strategy(self):
        def boom(state, questions):
            raise RuntimeError('网络断了')
        result = decide(self.store, self.candidate['strategy_id'], eval_report(), boom)
        self.assertFalse(result['applied'])
        self.assertEqual(result['source'], 'jev_error')
        self.assertIn('网络断了', result['reason'])

    def test_jev_not_ok_keeps_old_strategy(self):
        result = decide(self.store, self.candidate['strategy_id'],
                        eval_report(), gate(ok=False))
        self.assertFalse(result['applied'])
        self.assertEqual(result['source'], 'jev_unavailable')

    def test_invalid_answer_keeps_old_strategy(self):
        result = decide(self.store, self.candidate['strategy_id'],
                        eval_report(), gate('随便吧'))
        self.assertFalse(result['applied'])
        self.assertEqual(result['source'], 'jev_invalid')

    def test_request_carries_diff_and_evidence(self):
        spy = gate('keep')
        decide(self.store, self.candidate['strategy_id'], eval_report(), spy)
        sent = spy.calls[0]
        self.assertIn('planning_prompt', sent['state'])
        self.assertIn('duplicate_rate', sent['state'])
        self.assertEqual(sent['questions']['strategy_decision']['criteria'],
                         ['upgrade', 'keep', 'defer'])


class NormalizeTests(unittest.TestCase):
    def test_accepts_plain_and_chinese(self):
        self.assertEqual(normalize_decision('UPGRADE'), 'upgrade')
        self.assertEqual(normalize_decision('升级'), 'upgrade')
        self.assertEqual(normalize_decision({'choice': 'defer'}), 'defer')
        self.assertEqual(normalize_decision('保留'), 'keep')

    def test_rejects_unknown(self):
        self.assertIsNone(normalize_decision('maybe'))
        self.assertIsNone(normalize_decision(None))
        self.assertIsNone(normalize_decision({'other': 1}))


if __name__ == '__main__':
    unittest.main()
