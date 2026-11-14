import unittest

from routing_policy import MIN_SCORE, should_escalate, signals

HARD = {'rows': 9000, 'features': 23, 'class_count': 3, 'minority_fraction': 0.12, 'missing_fraction': 0.029}
VERY_HARD = {'rows': 6000, 'features': 26, 'class_count': 5, 'minority_fraction': 0.08, 'missing_fraction': 0.0}
EASY = {'rows': 5000, 'features': 20, 'class_count': 2, 'minority_fraction': 0.27, 'missing_fraction': 0.0}


class RoutingPolicyTests(unittest.TestCase):
    def test_easy_profile_does_not_escalate(self):
        self.assertIsNone(should_escalate(EASY, '请训练分类模型', 0.88,
                                          expert_calls=0, max_expert_calls=1, fits=2, max_fits=4))

    def test_many_classes_and_low_first_score_escalate(self):
        verdict = should_escalate(VERY_HARD, '请训练多分类模型', 0.49,
                                  expert_calls=0, max_expert_calls=1, fits=2, max_fits=4)
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict['reason'], 'data_difficulty')
        self.assertGreaterEqual(verdict['score'], MIN_SCORE)
        self.assertTrue(verdict['signals'])

    def test_high_first_score_does_not_escalate_even_when_data_is_hard(self):
        self.assertIsNone(should_escalate(VERY_HARD, '请训练多分类模型', 0.86,
                                          expert_calls=0, max_expert_calls=1, fits=2, max_fits=4))

    def test_expert_budget_zero_disables_the_gate(self):
        self.assertIsNone(should_escalate(VERY_HARD, '请训练多分类模型', 0.40,
                                          expert_calls=0, max_expert_calls=0, fits=2, max_fits=4))

    def test_only_one_diagnostic_per_run(self):
        self.assertIsNone(should_escalate(VERY_HARD, '请训练多分类模型', 0.40,
                                          expert_calls=1, max_expert_calls=1, fits=2, max_fits=4))

    def test_training_budget_exhausted_blocks_the_gate(self):
        self.assertIsNone(should_escalate(VERY_HARD, '请训练多分类模型', 0.40,
                                          expert_calls=0, max_expert_calls=1, fits=4, max_fits=4))

    def test_diagnosis_wording_raises_the_score(self):
        plain, _ = signals(VERY_HARD, '请训练分类模型', 0.49)
        hinted, reasons = signals(VERY_HARD, '请训练分类模型并诊断为什么提不上去', 0.49)
        self.assertGreater(hinted, plain)
        self.assertTrue(any('诊断' in r for r in reasons))


if __name__ == '__main__':
    unittest.main()
