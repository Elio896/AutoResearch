"""对照评估的判定逻辑：条件先冻结、样本不足不下结论、分数掉了不许升。"""
from __future__ import annotations

import unittest

from strategy_eval import DEFAULT_CRITERIA, compare, freeze_criteria


def arm(legal=0.9, dup=0.1, score=0.60, tasks=3, calls=10, label='arm'):
    return {
        'label': label, 'strategy_id': 's', 'task_count': tasks, 'completed': tasks,
        'completion_rate': 1.0, 'mean_best_score': score,
        'legal_output_rate': legal, 'duplicate_rate': dup,
        'local_calls': calls, 'cloud_calls': 0, 'cost_basis': 'local_no_charge',
        'proposals': 10, 'invalid_outputs': 1, 'tasks_detail': [],
    }


class FreezeTests(unittest.TestCase):
    def test_freeze_is_deterministic_and_stamped(self):
        first = freeze_criteria()
        second = freeze_criteria()
        self.assertEqual(first['digest'], second['digest'])
        self.assertGreater(first['frozen_at'], 0)
        self.assertEqual(first['criteria'], DEFAULT_CRITERIA)

    def test_freeze_copies_instead_of_aliasing(self):
        custom = {'min_tasks': 5, 'legal_output_rate': {'min_delta': 0},
                  'duplicate_rate': {'max_delta': 0}, 'mean_best_score': {'min_delta': 0},
                  'calls': {'max_ratio': 2.0}}
        frozen = freeze_criteria(custom)
        custom['min_tasks'] = 9
        self.assertEqual(frozen['criteria']['min_tasks'], 5)


class CompareTests(unittest.TestCase):
    def test_better_new_strategy_upgrades(self):
        frozen = freeze_criteria()
        report = compare(arm(legal=0.80, dup=0.30), arm(legal=0.95, dup=0.10), frozen)
        self.assertEqual(report['verdict'], 'upgrade', report['checks'])
        self.assertEqual(report['failed_checks'], [])

    def test_legal_rate_drop_blocks(self):
        frozen = freeze_criteria()
        report = compare(arm(legal=0.95), arm(legal=0.70), frozen)
        self.assertEqual(report['verdict'], 'keep')
        self.assertIn('legal_output_rate', report['failed_checks'])

    def test_duplicate_rate_increase_blocks(self):
        frozen = freeze_criteria()
        report = compare(arm(dup=0.05), arm(dup=0.40), frozen)
        self.assertEqual(report['verdict'], 'keep')
        self.assertIn('duplicate_rate', report['failed_checks'])

    def test_score_regression_beyond_tolerance_blocks(self):
        frozen = freeze_criteria()
        report = compare(arm(score=0.70), arm(score=0.65), frozen)
        self.assertEqual(report['verdict'], 'keep')
        self.assertIn('mean_best_score', report['failed_checks'])

    def test_small_score_drift_allowed(self):
        """分数小幅下降不单独判死，但必须另有可测改善才升。"""
        frozen = freeze_criteria()
        report = compare(arm(score=0.700, legal=0.80), arm(score=0.690, legal=0.95), frozen)
        self.assertEqual(report['verdict'], 'upgrade', report['checks'])

    def test_no_improvement_is_keep_not_upgrade(self):
        """一模一样的两条臂：没退化，但也没改善，不许升。"""
        frozen = freeze_criteria()
        report = compare(arm(), arm(), frozen)
        self.assertEqual(report['verdict'], 'keep')
        self.assertIn('demonstrated_improvement', report['failed_checks'])
        self.assertEqual(report['checks'][-1]['detail']['improved'], [])

    def test_call_inflation_blocks(self):
        frozen = freeze_criteria()
        report = compare(arm(calls=10), arm(calls=40), frozen)
        self.assertEqual(report['verdict'], 'keep')
        self.assertIn('calls', report['failed_checks'])

    def test_too_few_tasks_is_insufficient(self):
        frozen = freeze_criteria()
        report = compare(arm(tasks=1), arm(tasks=1), frozen)
        self.assertEqual(report['verdict'], 'insufficient_evidence')
        self.assertIn('min_tasks', report['failed_checks'] + report['unknown_checks']
                      or [c['check'] for c in report['checks'] if not c['passed']])

    def test_missing_metric_is_insufficient_not_pass(self):
        frozen = freeze_criteria()
        broken = arm()
        broken['legal_output_rate'] = None
        report = compare(arm(), broken, frozen)
        self.assertEqual(report['verdict'], 'insufficient_evidence')
        self.assertIn('legal_output_rate', report['unknown_checks'])

    def test_report_keeps_frozen_criteria(self):
        frozen = freeze_criteria()
        report = compare(arm(), arm(legal=0.99), frozen)
        self.assertEqual(report['criteria']['digest'], frozen['digest'])


if __name__ == '__main__':
    unittest.main()
