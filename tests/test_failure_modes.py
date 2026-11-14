"""失败模式检测的验收测试：能追溯、不硬凑、证据不足就说不足。"""
from __future__ import annotations

import unittest

import failure_modes
from failure_modes import detect, event_refs


def proposal(model: str, params: dict | None = None) -> dict:
    return {'event': 'scientist_proposal',
            'proposal': {'model': model, 'parameters': params or {}}}


def experiment(model: str, score: float) -> dict:
    return {'event': 'experiment', 'model': model, 'score': score}


class TraceabilityTests(unittest.TestCase):
    def test_event_refs_point_at_the_right_events(self):
        events = [{'event': 'analyst_profile'}, proposal('logistic_regression'),
                  {'event': 'manager_plan'}, proposal('logistic_regression')]
        refs = event_refs(events, [1, 3])
        self.assertEqual(refs, ['event_001', 'event_003'])
        for ref in refs:
            index = int(ref.split('_')[1])
            self.assertEqual(events[index]['event'], 'scientist_proposal')


class DetectionTests(unittest.TestCase):
    def test_repeated_proposal_found_with_refs(self):
        events = [proposal('logistic_regression', {'max_iter': 1000}),
                  experiment('logistic_regression', 0.5),
                  proposal('logistic_regression', {'max_iter': 1000}),
                  experiment('logistic_regression', 0.5)]
        result = detect(events, run_id='run_1', strategy_id='strategy_001')
        self.assertEqual(result['verdict'], 'failures_found')
        record = next(r for r in result['records'] if r['failure_type'] == 'repeated_proposal')
        self.assertEqual(record['measurements']['duplicate_count'], 2)
        self.assertEqual(record['event_refs'], ['event_000', 'event_002'])
        self.assertEqual(record['run_id'], 'run_1')
        self.assertEqual(record['strategy_id'], 'strategy_001')

    def test_same_family_no_gain_needs_three_flat_rounds(self):
        events = [proposal('random_forest', {'n_estimators': 50}),
                  experiment('random_forest', 0.60)]
        for _ in range(3):
            events += [proposal('random_forest', {'n_estimators': 60}),
                       experiment('random_forest', 0.59)]
        result = detect(events)
        record = next(r for r in result['records']
                      if r['failure_type'] == 'same_family_tuning_no_gain')
        self.assertEqual(record['measurements']['no_gain_streak'], 3)
        self.assertEqual(record['measurements']['model_family'], 'tree')

    def test_two_flat_rounds_is_not_enough(self):
        events = [proposal('random_forest'), experiment('random_forest', 0.60),
                  proposal('random_forest'), experiment('random_forest', 0.59)]
        result = detect(events)
        self.assertFalse([r for r in result['records']
                          if r['failure_type'] == 'same_family_tuning_no_gain'])

    def test_invalid_structure_repeat(self):
        events = [{'event': 'scientist_error', 'error': 'JSON 解析失败'},
                  {'event': 'scientist_error', 'error': '没有 proposal 字段'}]
        result = detect(events)
        record = next(r for r in result['records']
                      if r['failure_type'] == 'invalid_structure_repeat')
        self.assertEqual(record['measurements']['invalid_count'], 2)
        self.assertEqual(len(record['measurements']['samples']), 2)

    def test_operation_check_failed_uses_worklog(self):
        worklog = [{'run_id': 'run_9', 'logged_at': 123.0,
                    'tool_computation': {'tool': 'profile.missing'},
                    'check_results': [{'check': 'rows_readable', 'passed': False,
                                       'detail': '训练行数 0'}]}]
        result = detect([], worklog_records=worklog)
        record = next(r for r in result['records']
                      if r['failure_type'] == 'operation_check_failed')
        self.assertEqual(record['measurements']['failed_checks'], ['rows_readable'])
        self.assertTrue(record['event_refs'][0].startswith('worklog:run_9'))

    def test_improving_run_is_clean(self):
        events = [proposal('logistic_regression', {'max_iter': 100}),
                  experiment('logistic_regression', 0.50),
                  proposal('random_forest', {'n_estimators': 50}),
                  experiment('random_forest', 0.62),
                  proposal('random_forest', {'n_estimators': 80}),
                  experiment('random_forest', 0.65)]
        result = detect(events)
        self.assertEqual(result['verdict'], 'clean')
        self.assertEqual(result['records'], [])


class InsufficientEvidenceTests(unittest.TestCase):
    def test_empty_run_says_insufficient(self):
        result = detect([])
        self.assertTrue(result['insufficient'])
        self.assertEqual(result['verdict'], 'insufficient_evidence')
        self.assertEqual(result['records'], [])
        self.assertTrue(result['insufficient_reasons'])

    def test_single_experiment_says_insufficient(self):
        result = detect([proposal('logistic_regression'), experiment('logistic_regression', 0.5)])
        self.assertTrue(result['insufficient'])
        self.assertEqual(result['failure_count'], 0)

    def test_no_worklog_is_reported_not_guessed(self):
        result = detect([proposal('logistic_regression'), experiment('logistic_regression', 0.5)])
        self.assertFalse(result['detectors']['operation_check_failed']['evaluated'])
        self.assertIn('没有提供工作记录', result['detectors']['operation_check_failed']['reason'])


if __name__ == '__main__':
    unittest.main()
