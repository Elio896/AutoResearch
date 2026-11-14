"""第七步测试：回滚条件事先冻结、触发才动、失败案例不删。"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from strategy_rollback import evaluate, freeze_rules, rollback_if_needed
from strategy_store import StrategyStore

OLD = {'classifier_prompt': 'A', 'planning_prompt': 'P1', 'candidate_selection_rules': []}
NEW = {'planning_prompt': 'P2'}


def metrics(legal=1.0, dup=0.0, score=0.70, tasks=2, violations=0):
    return {'task_count': tasks, 'legal_output_rate': legal, 'duplicate_rate': dup,
            'mean_best_score': score, 'budget_violations': violations}


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.rules = freeze_rules()['rules']

    def test_healthy_strategy_holds(self):
        report = evaluate(metrics(), metrics(), self.rules)
        self.assertEqual(report['verdict'], 'hold')
        self.assertEqual(report['triggers'], [])

    def test_legal_rate_collapse_triggers(self):
        report = evaluate(metrics(legal=1.0), metrics(legal=0.60), self.rules)
        self.assertEqual(report['verdict'], 'rollback')
        self.assertIn('legal_output_rate_drop', report['triggers'])

    def test_duplicate_explosion_triggers(self):
        report = evaluate(metrics(dup=0.0), metrics(dup=0.40), self.rules)
        self.assertIn('duplicate_rate_rise', report['triggers'])

    def test_score_regression_triggers(self):
        report = evaluate(metrics(score=0.70), metrics(score=0.60), self.rules)
        self.assertIn('mean_best_score_drop', report['triggers'])

    def test_budget_violation_triggers(self):
        report = evaluate(metrics(), metrics(violations=2), self.rules)
        self.assertIn('budget_violations', report['triggers'])

    def test_small_drift_does_not_trigger(self):
        report = evaluate(metrics(legal=1.0, dup=0.0), metrics(legal=0.95, dup=0.05),
                          self.rules)
        self.assertEqual(report['verdict'], 'hold', report['checks'])

    def test_unknown_metric_is_insufficient_not_hold(self):
        broken = metrics()
        broken['legal_output_rate'] = None
        report = evaluate(metrics(), broken, self.rules)
        self.assertEqual(report['verdict'], 'insufficient_evidence')
        self.assertFalse(report['triggers'])

    def test_too_few_tasks_is_insufficient(self):
        report = evaluate(metrics(tasks=0), metrics(tasks=0), self.rules)
        self.assertIn(report['verdict'], ('insufficient_evidence', 'hold'))


class RollbackFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='rollback-')
        self.store = StrategyStore(Path(self.tmp) / 'store.json')
        self.old = self.store.bootstrap(OLD, reason='基线')
        self.new = self.store.propose(NEW, reason='治重复', evidence_run_ids=['run_1'])
        self.store.activate(self.new['strategy_id'], checks={'ok': True},
                            jev={'decision': 'upgrade'}, reason='上线')
        self.rules = freeze_rules()['rules']

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_triggers_switch_back_and_keep_evidence(self):
        report = rollback_if_needed(
            self.store, current_id=self.new['strategy_id'],
            fallback_id=self.old['strategy_id'],
            baseline=metrics(), current=metrics(legal=0.5), rules=self.rules)
        self.assertTrue(report['applied'])
        self.assertEqual(self.store.active()['strategy_id'], self.old['strategy_id'])

        event = self.store.events()[-1]
        self.assertEqual(event['type'], 'rollback')
        self.assertEqual(event['from'], self.new['strategy_id'])
        self.assertIn('legal_output_rate_drop', event['triggers'])
        self.assertIn('checks', event['evidence'])

        # 失败的那个版本不能被删除
        failed = self.store.get(self.new['strategy_id'])
        self.assertIsNotNone(failed)
        self.assertEqual(failed['status'], 'retired')
        self.assertEqual(failed['changes'], {**OLD, **NEW})

    def test_no_trigger_keeps_current(self):
        report = rollback_if_needed(
            self.store, current_id=self.new['strategy_id'],
            fallback_id=self.old['strategy_id'],
            baseline=metrics(), current=metrics(legal=0.98), rules=self.rules)
        self.assertFalse(report['applied'])
        self.assertEqual(self.store.active()['strategy_id'], self.new['strategy_id'])

    def test_insufficient_evidence_does_not_rollback(self):
        broken = metrics()
        broken['legal_output_rate'] = None
        report = rollback_if_needed(
            self.store, current_id=self.new['strategy_id'],
            fallback_id=self.old['strategy_id'],
            baseline=metrics(), current=broken, rules=self.rules)
        self.assertFalse(report['applied'])
        self.assertEqual(report['verdict'], 'insufficient_evidence')
        self.assertEqual(self.store.active()['strategy_id'], self.new['strategy_id'])

    def test_rollback_requires_triggers(self):
        with self.assertRaises(ValueError):
            self.store.rollback(self.old['strategy_id'], triggers=[], evidence={})


if __name__ == '__main__':
    unittest.main()
