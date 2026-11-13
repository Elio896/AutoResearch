"""五角色 + 研究闭环的单元测试（不依赖网络、不依赖真实模型）。"""
import json
import unittest

from agents import Analyst, Budget, Manager, Reviewer, Scientist
from research_loop import DEFAULT_ALLOWED, ResearchLoop


def fake_proposer(reply):
    """返回一个固定回复的 proposer（模拟 Ollama 的响应结构）。"""
    def _call(messages, timeout=None):
        return {'message': {'content': reply}, 'prompt_eval_count': 12, 'eval_count': 8}
    return _call


class AnalystTests(unittest.TestCase):
    def test_detects_imbalance(self):
        profile = Analyst().profile({
            'class_counts': [{'class_id': 0, 'count': 923}, {'class_id': 1, 'count': 77}],
            'rows': 1000, 'features': 20, 'missing_fraction': 0.0,
            'constant_features': 0, 'class_count': 2,
        })
        self.assertTrue(profile['imbalanced'])
        self.assertEqual(profile['baseline_class_weight'], 'balanced')
        self.assertIn('class_imbalance', profile['risks'])

    def test_balanced_data_has_no_forced_weight(self):
        profile = Analyst().profile({
            'class_counts': [{'class_id': 0, 'count': 500}, {'class_id': 1, 'count': 500}],
            'rows': 1000, 'features': 20, 'missing_fraction': 0.0,
            'constant_features': 0, 'class_count': 2,
        })
        self.assertFalse(profile['imbalanced'])
        self.assertIsNone(profile['baseline_class_weight'])

    def test_small_sample_risk(self):
        profile = Analyst().profile({
            'class_counts': [{'class_id': 0, 'count': 400}, {'class_id': 1, 'count': 400}],
            'rows': 800, 'features': 10, 'missing_fraction': 0.0,
            'constant_features': 0, 'class_count': 2,
        })
        self.assertIn('small_sample', profile['risks'])


class ManagerTests(unittest.TestCase):
    def test_stop_on_experiment_budget(self):
        m = Manager(Budget(max_experiments=2))
        stop, reason = m.should_stop({'experiments': 2})
        self.assertTrue(stop)
        self.assertEqual(reason, 'experiment_budget')

    def test_stop_on_plateau(self):
        m = Manager(Budget(), patience=2)
        stop, reason = m.should_stop({'experiments': 1, 'no_improvement': 2})
        self.assertTrue(stop)
        self.assertEqual(reason, 'validation_plateau')

    def test_no_escalation_without_plateau(self):
        m = Manager(Budget(max_cloud_calls=1))
        ok, why = m.should_escalate({'used': {'experiments': 1}}, {'rows': 5000})
        self.assertFalse(ok)
        self.assertEqual(why, 'local_still_progressing')

    def test_escalate_on_plateau_with_budget(self):
        m = Manager(Budget(max_cloud_calls=1), patience=2)
        ok, why = m.should_escalate(
            {'used': {'experiments': 2, 'local_calls': 2}, 'no_improvement': 2},
            {'rows': 5000})
        self.assertTrue(ok)
        self.assertEqual(why, 'plateau_on_nontrivial_data')

    def test_no_escalation_on_tiny_data(self):
        m = Manager(Budget(max_cloud_calls=1), patience=2)
        ok, why = m.should_escalate(
            {'used': {'experiments': 2}, 'no_improvement': 2}, {'rows': 200})
        self.assertFalse(ok)

    def test_budget_split_sums_to_one(self):
        total = sum(Manager(Budget()).split_budget().values())
        self.assertAlmostEqual(total, 1.0)


class ScientistTests(unittest.TestCase):
    def test_accepts_valid_proposal(self):
        s = Scientist(fake_proposer('{"model":"hist_gradient_boosting","parameters":{"max_depth":4},"hypothesis":"test hypothesis","expected_direction":"up","falsifier":"验证分数没有提升"}'), DEFAULT_ALLOWED)
        out = s.parse({'message': {'content': '{"model":"hist_gradient_boosting","parameters":{"max_depth":4},"hypothesis":"test hypothesis","expected_direction":"up","falsifier":"验证分数没有提升"}'}})
        self.assertEqual(out['model'], 'hist_gradient_boosting')

    def test_rejects_unknown_model(self):
        s = Scientist(fake_proposer('{}'), DEFAULT_ALLOWED)
        with self.assertRaises(ValueError):
            s.parse({'message': {'content': '{"model":"xgboost","parameters":{},"hypothesis":"test hypothesis","expected_direction":"up","falsifier":"验证分数没有提升"}'}})

    def test_rejects_out_of_range(self):
        s = Scientist(fake_proposer('{}'), DEFAULT_ALLOWED)
        with self.assertRaises(ValueError):
            s.parse({'message': {'content': '{"model":"random_forest","parameters":{"max_depth":999},"hypothesis":"test hypothesis","expected_direction":"up","falsifier":"验证分数没有提升"}'}})

    def test_rejects_bad_class_weight(self):
        s = Scientist(fake_proposer('{}'), DEFAULT_ALLOWED)
        with self.assertRaises(ValueError):
            s.parse({'message': {'content': '{"model":"random_forest","parameters":{"class_weight":"yes"},"hypothesis":"test hypothesis","expected_direction":"up","falsifier":"验证分数没有提升"}'}})

    def test_rejects_non_json(self):
        s = Scientist(fake_proposer('ok'), DEFAULT_ALLOWED)
        with self.assertRaises(ValueError):
            s.parse({'message': {'content': 'sure, use random forest'}})


class ReviewerTests(unittest.TestCase):
    def test_improved(self):
        r = Reviewer().judge({'validation_score': 0.7}, {'validation_score': 0.6}, {'rows': 5000})
        self.assertEqual(r['verdict'], 'improved')
        self.assertAlmostEqual(r['delta'], 0.1, places=6)

    def test_flat(self):
        r = Reviewer().judge({'validation_score': 0.6}, {'validation_score': 0.6}, {'rows': 5000})
        self.assertEqual(r['verdict'], 'flat')

    def test_suspicious_on_tiny_data(self):
        r = Reviewer().judge({'validation_score': 1.0}, {'validation_score': 0.6}, {'rows': 300})
        self.assertEqual(r['verdict'], 'suspicious_improvement')
        self.assertTrue(r['suspicious'])

    def test_gates_are_recorded_for_every_round(self):
        """轨迹规范第 07 项：闸门要逐条记 pass/fail。"""
        r = Reviewer().judge({'validation_score': 0.7}, {'validation_score': 0.6}, {'rows': 5000})
        self.assertIsInstance(r['gates'], list)
        self.assertGreaterEqual(len(r['gates']), 3)
        for gate in r['gates']:
            self.assertIn('name', gate)
            self.assertIn('pass', gate)
            self.assertIsInstance(gate['pass'], bool)

    def test_improved_is_not_reverted(self):
        r = Reviewer().judge({'validation_score': 0.7}, {'validation_score': 0.6}, {'rows': 5000})
        self.assertFalse(r['reverted'])
        self.assertEqual(r['redlines'], [])

    def test_suspicious_is_reverted_and_redlined(self):
        r = Reviewer().judge({'validation_score': 1.0}, {'validation_score': 0.6}, {'rows': 300})
        self.assertTrue(r['reverted'])
        self.assertEqual(r['redlines'], ['score_jump_too_good'])

    def test_regressed_is_reverted_but_kept_as_evidence(self):
        r = Reviewer().judge({'validation_score': 0.5}, {'validation_score': 0.6}, {'rows': 5000})
        self.assertEqual(r['verdict'], 'regressed')
        self.assertTrue(r['reverted'])


class FakeSession:
    """假的 executor：按模型名给固定分数，不真训练。"""

    SCORES = {'logistic_regression': 0.30, 'random_forest': 0.43, 'hist_gradient_boosting': 0.63}

    def __init__(self):
        self.calls = []

    def planning_context(self):
        return {'class_counts': [{'class_id': 0, 'count': 923}, {'class_id': 1, 'count': 77}],
                'rows': 1000, 'features': 20, 'missing_fraction': 0.05,
                'constant_features': 0, 'class_count': 2, 'selection_metric': 'f1_minority'}

    def run(self, model='logistic_regression', parameters=None):
        self.calls.append((model, parameters))
        return {'model': model, 'parameters': parameters or {},
                'validation_score': self.SCORES.get(model, 0.1),
                'validation_f1': self.SCORES.get(model, 0.1)}


class ResearchLoopTests(unittest.TestCase):
    def test_full_loop_improves_and_respects_budget(self):
        replies = [
            '{"model":"random_forest","parameters":{"n_estimators":100},"hypothesis":"test hypothesis","expected_direction":"up","falsifier":"验证分数没有提升"}',
            '{"model":"hist_gradient_boosting","parameters":{"max_depth":4},"hypothesis":"test hypothesis","expected_direction":"up","falsifier":"验证分数没有提升"}',
            '{"model":"hist_gradient_boosting","parameters":{"max_depth":6},"hypothesis":"test hypothesis","expected_direction":"up","falsifier":"验证分数没有提升"}',
        ]
        state = {'i': 0}

        def proposer(messages, timeout=None):
            reply = replies[min(state['i'], len(replies) - 1)]
            state['i'] += 1
            return {'message': {'content': reply}, 'prompt_eval_count': 40, 'eval_count': 20}

        session = FakeSession()
        loop = ResearchLoop(session, proposer=proposer,
                            budget=Budget(max_experiments=4, max_seconds=60, max_local_calls=4),
                            patience=2)
        result = loop.run()

        summary = result['summary']
        self.assertEqual(summary['status'], 'completed')
        self.assertLessEqual(len(session.calls), 4)               # 预算守住
        self.assertEqual(summary['best_model'], 'hist_gradient_boosting')
        self.assertAlmostEqual(summary['best_score'], 0.63, places=2)
        self.assertFalse(summary['test_evaluated'])                # 不碰测试集
        # baseline 必须带 balanced（数据不平衡）
        self.assertEqual(session.calls[0][1].get('class_weight'), 'balanced')
        # token 必须被记进账（每次调用 40+20）
        self.assertEqual(summary['local_tokens'], 60 * summary['local_calls'])

    def test_stops_when_no_proposer(self):
        session = FakeSession()
        loop = ResearchLoop(session, proposer=None, budget=Budget(max_experiments=3))
        result = loop.run()
        self.assertEqual(result['summary']['stop_reason'], 'no_proposer')
        self.assertEqual(len(session.calls), 1)

    def test_rejected_proposals_do_not_crash(self):
        def proposer(messages, timeout=None):
            return {'message': {'content': 'not json at all'}}

        session = FakeSession()
        loop = ResearchLoop(session, proposer=proposer,
                            budget=Budget(max_experiments=3, max_local_calls=2))
        result = loop.run()
        self.assertIn(result['summary']['status'], ('completed', 'failed'))
        rejected = [e for e in result['events'] if e['event'] == 'scientist_rejected']
        self.assertGreaterEqual(len(rejected), 1)

    def test_rejections_stop_on_call_budget_not_plateau(self):
        """模型一直说不出合法方案时，应该被 local_calls 卡住，而不是判"研究停滞"。"""
        def proposer(messages, timeout=None):
            return {'message': {'content': 'garbage'}}

        session = FakeSession()
        loop = ResearchLoop(session, proposer=proposer,
                            budget=Budget(max_experiments=5, max_local_calls=3),
                            patience=2)
        result = loop.run()
        self.assertEqual(result['summary']['stop_reason'], 'local_call_budget')
        # 只有 baseline 跑成功，没有实验被"误判为没进步"
        self.assertEqual(len(session.calls), 1)


if __name__ == '__main__':
    unittest.main()
