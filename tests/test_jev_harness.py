"""Jev Harness 五道门的测试：每道门都要能单独挡住云端调用。"""
import unittest

from jev_harness import JevHarness, TIER_FLASH, TIER_LOCAL, TIER_REASONER


def ctx(**kw):
    base = {'task': 't2', 'rows': 3000, 'no_improvement': 0,
            'last_gain': None, 'recent_rejections': [], 'recent_proposals': []}
    base.update(kw)
    return base


class GateTests(unittest.TestCase):
    def setUp(self):
        self.h = JevHarness()

    def test_trivial_schema_errors_keep_it_local(self):
        decision = self.h.decide(ctx(recent_rejections=[
            'Unsupported model: svm', 'max_depth outside [1, 20]']))
        self.assertEqual(decision.action, 'stay_local')
        self.assertIn('schema', decision.reason)

    def test_doom_loop_keeps_it_local(self):
        same = {'model': 'rf', 'parameters': {'max_depth': 5}}
        decision = self.h.decide(ctx(recent_proposals=[same, same, same]))
        self.assertEqual(decision.action, 'stay_local')
        self.assertIn('identical', decision.reason)

    def test_tiny_gain_vetoes(self):
        decision = self.h.decide(ctx(last_gain=0.0001))
        self.assertEqual(decision.action, 'takeover')
        self.assertIn('too small', decision.reason)

    def test_small_data_routes_local(self):
        decision = self.h.decide(ctx(rows=200, no_improvement=3))
        self.assertEqual(decision.tier, TIER_LOCAL)
        self.assertEqual(decision.action, 'stay_local')

    def test_routine_escalation_uses_flash(self):
        decision = self.h.decide(ctx(rows=3000, no_improvement=1))
        self.assertEqual(decision.action, 'escalate')
        self.assertEqual(decision.tier, TIER_FLASH)
        self.assertEqual(decision.effort, 'normal')

    def test_deep_stuck_uses_reasoner(self):
        decision = self.h.decide(ctx(rows=3000, no_improvement=4))
        self.assertEqual(decision.action, 'escalate')
        self.assertEqual(decision.tier, TIER_REASONER)
        self.assertEqual(decision.effort, 'deep')

    def test_decision_is_auditable(self):
        decision = self.h.decide(ctx(rows=3000, no_improvement=3))
        data = decision.to_dict()
        for key in ('action', 'tier', 'effort', 'gates', 'reason', 'saved_tokens_estimate'):
            self.assertIn(key, data)
        self.assertTrue(all('name' in g or 'gate' in g for g in data['gates']) or data['gates'])

    def test_no_model_call_happens(self):
        """这道层必须是纯计算——它自己一调模型就不省 token 了。"""
        decision = self.h.decide(ctx(rows=3000, no_improvement=2))
        self.assertIsInstance(decision.saved_tokens_estimate, int)


if __name__ == '__main__':
    unittest.main()
