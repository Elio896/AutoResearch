"""可计算信号路由的测试。

规则来自二德子（2026-11-14）：
    停滞 >= 2 且 同类改动历史 Δ <= 0  → 升级
    首次合法率 < 0.5                → 升级
    否则                            → 本地继续
"""
import unittest

from signal_router import (SignalState, classify_change_kind, decide,
                           historical_delta, state_from_events)


class DecideTests(unittest.TestCase):
    def test_stalled_and_no_gain_escalates(self):
        state = SignalState(stalled_rounds=3,
                            recent_change_kinds=['tune_capacity'],
                            historical_delta_by_kind={'tune_capacity': -0.02})
        result = decide(state)
        self.assertEqual(result['action'], 'escalate')
        self.assertEqual(result['rule'], 'stalled_and_no_gain')

    def test_stalled_but_gain_positive_stays_local(self):
        state = SignalState(stalled_rounds=3,
                            recent_change_kinds=['swap_model'],
                            historical_delta_by_kind={'swap_model': 0.04})
        result = decide(state)
        self.assertEqual(result['action'], 'continue')

    def test_low_first_try_pass_rate_escalates(self):
        state = SignalState(proposals_attempted=10, proposals_accepted_first_try=3,
                            proposals_total=10, stalled_rounds=0)
        result = decide(state)
        self.assertEqual(result['action'], 'escalate')
        self.assertEqual(result['rule'], 'low_first_try_pass')

    def test_not_stalled_yet_stays_local(self):
        state = SignalState(stalled_rounds=1,
                            recent_change_kinds=['tune_capacity'],
                            historical_delta_by_kind={'tune_capacity': -0.05})
        result = decide(state)
        self.assertEqual(result['action'], 'continue')
        self.assertIn('还没到阈值', result['reason'])

    def test_no_history_keeps_local_and_says_why(self):
        state = SignalState(stalled_rounds=0)
        result = decide(state)
        self.assertEqual(result['action'], 'continue')
        self.assertIn('还没有同类改动的历史数据', result['reason'])

    def test_signals_always_reported(self):
        state = SignalState(proposals_attempted=4, proposals_accepted_first_try=2,
                            proposals_total=4, proposals_duplicate=1, stalled_rounds=0)
        result = decide(state)
        for key in ('first_try_pass_rate', 'duplicate_rate',
                    'stalled_rounds', 'historical_delta'):
            self.assertIn(key, result['signals'])

    def test_confidence_is_not_an_input(self):
        """模型自评不该参与决策——SignalState 里根本没有这个字段。"""
        self.assertNotIn('confidence', SignalState.__dataclass_fields__)


class ChangeKindTests(unittest.TestCase):
    def test_swap_model(self):
        self.assertEqual(
            classify_change_kind({'model': 'lr', 'parameters': {}},
                                 {'model': 'rf', 'parameters': {}}),
            'swap_model')

    def test_tune_capacity(self):
        self.assertEqual(
            classify_change_kind({'model': 'rf', 'parameters': {'max_depth': 3}},
                                 {'model': 'rf', 'parameters': {'max_depth': 6}}),
            'tune_capacity')

    def test_tune_regularization(self):
        self.assertEqual(
            classify_change_kind({'model': 'lr', 'parameters': {'C': 1}},
                                 {'model': 'lr', 'parameters': {'C': 0.1}}),
            'tune_regularization')

    def test_no_change(self):
        self.assertEqual(
            classify_change_kind({'model': 'rf', 'parameters': {'max_depth': 3}},
                                 {'model': 'rf', 'parameters': {'max_depth': 3}}),
            'no_change')

    def test_baseline(self):
        self.assertEqual(classify_change_kind(None, {'model': 'lr'}), 'baseline')


class StateFromEventsTests(unittest.TestCase):
    def test_counts_rejections_and_stall(self):
        events = [
            {'event': 'scientist_proposal', 'proposal': {'model': 'rf'}},
            {'event': 'scientist_rejected', 'reason': 'Duplicate experiment; choose different parameters.'},
            {'event': 'reviewer_verdict', 'verdict': {'verdict': 'improved'}},
            {'event': 'reviewer_verdict', 'verdict': {'verdict': 'flat'}},
            {'event': 'reviewer_verdict', 'verdict': {'verdict': 'regressed'}},
        ]
        experiments = [
            {'model': 'lr', 'parameters': {}, 'validation_score': 0.30},
            {'model': 'rf', 'parameters': {}, 'validation_score': 0.45},
        ]
        state = state_from_events(events, experiments)
        self.assertEqual(state.proposals_attempted, 2)
        self.assertEqual(state.proposals_accepted_first_try, 1)
        self.assertEqual(state.proposals_duplicate, 1)
        self.assertEqual(state.stalled_rounds, 2)     # flat + regressed 连续
        self.assertIn('swap_model', state.historical_delta_by_kind)

    def test_historical_delta_averages(self):
        state = SignalState(recent_change_kinds=['a', 'b'],
                            historical_delta_by_kind={'a': 0.02, 'b': 0.06})
        self.assertAlmostEqual(historical_delta(state), 0.04, places=6)


if __name__ == '__main__':
    unittest.main()
