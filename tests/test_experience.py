"""经验记忆的测试：写入约束、匹配、否证、闭环复用。"""
import json
import tempfile
import unittest
from pathlib import Path

from agents import Budget
from experience import ExperienceStore, advice_lines
from research_loop import ResearchLoop


class ExperienceStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ExperienceStore(Path(self.tmp.name) / 'exp.jsonl')

    def tearDown(self):
        self.tmp.cleanup()

    def test_requires_evidence(self):
        with self.assertRaises(ValueError):
            self.store.add(condition={'imbalanced': True}, advice='试试 histgb', evidence={})

    def test_requires_advice(self):
        with self.assertRaises(ValueError):
            self.store.add(condition={'imbalanced': True}, advice='', evidence={'task': 't1'})

    def test_add_and_read(self):
        rid = self.store.add(condition={'imbalanced': True}, advice='试试 histgb',
                             evidence={'task': 't2', 'delta': 0.12})
        records = self.store.all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['id'], rid)
        self.assertEqual(records[0]['kind'], 'observed')

    def test_match_by_condition(self):
        self.store.add(condition={'imbalanced': True}, advice='A', evidence={'task': 't2'})
        self.store.add(condition={'imbalanced': False}, advice='B', evidence={'task': 't3'})
        matched = self.store.match({'imbalanced': True})
        self.assertEqual([r['advice'] for r in matched], ['A'])

    def test_more_specific_condition_ranks_first(self):
        self.store.add(condition={'imbalanced': True}, advice='general', evidence={'t': 1})
        self.store.add(condition={'imbalanced': True, 'class_count': 2,
                                  'imbalanced_extra': True}, advice='specific',
                       evidence={'t': 2})
        matched = self.store.match({'imbalanced': True, 'class_count': 2, 'imbalanced_extra': True})
        # 只有两条都满足时才比较具体度；这里 specific 条件更多
        self.assertEqual(matched[0]['advice'], 'specific')

    def test_contradicted_not_suggested(self):
        rid = self.store.add(condition={'imbalanced': True}, advice='A', evidence={'t': 1})
        self.store.contradict(rid, evidence={'task': 't9', 'delta': -0.2})
        matched = self.store.match({'imbalanced': True})
        self.assertEqual(matched, [])
        self.assertEqual(self.store.stats()['contradicted'], 1)

    def test_advice_lines(self):
        self.store.add(condition={'imbalanced': True}, advice='试试 histgb', evidence={'task': 't2'})
        lines = advice_lines(self.store.match({'imbalanced': True}))
        self.assertEqual(len(lines), 1)
        self.assertIn('试试 histgb', lines[0])
        self.assertIn('t2', lines[0])


class FakeSession:
    SCORES = {'logistic_regression': 0.30, 'random_forest': 0.43, 'hist_gradient_boosting': 0.63}

    def __init__(self, imbalanced=True):
        self.imbalanced = imbalanced
        self.calls = []

    def planning_context(self):
        counts = ([{'class_id': 0, 'count': 923}, {'class_id': 1, 'count': 77}]
                  if self.imbalanced
                  else [{'class_id': 0, 'count': 500}, {'class_id': 1, 'count': 500}])
        return {'class_counts': counts, 'rows': 4000, 'features': 20,
                'missing_fraction': 0.0, 'constant_features': 0, 'class_count': 2,
                'selection_metric': 'f1_minority'}

    def run(self, model='logistic_regression', parameters=None):
        self.calls.append((model, parameters))
        score = self.SCORES.get(model, 0.1)
        return {'model': model, 'parameters': parameters or {},
                'validation_score': score, 'validation_f1': score}


class LearningLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ExperienceStore(Path(self.tmp.name) / 'exp.jsonl')

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def proposer_histgb(messages, timeout=None):
        return {'message': {'content': '{"model":"hist_gradient_boosting","parameters":{"max_depth":4},"hypothesis":"test hypothesis","expected_direction":"up","falsifier":"验证分数没有提升"}'},
                'prompt_eval_count': 10, 'eval_count': 5}

    def test_loop_learns_from_a_win(self):
        session = FakeSession(imbalanced=True)
        loop = ResearchLoop(session, proposer=self.proposer_histgb,
                            budget=Budget(max_experiments=3, max_local_calls=3),
                            patience=5, experience=self.store, task_name='t2_fake')
        loop.run()
        records = self.store.all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['evidence']['task'], 't2_fake')
        self.assertGreater(records[0]['evidence']['delta'], 0)
        self.assertIn('hist_gradient_boosting', records[0]['advice'])

    def test_learned_experience_shows_up_in_next_task_prompt(self):
        # 第一轮：学到"不平衡时用 histgb"
        ResearchLoop(FakeSession(imbalanced=True), proposer=self.proposer_histgb,
                     budget=Budget(max_experiments=2, max_local_calls=2),
                     patience=5, experience=self.store, task_name='t2_fake').run()
        # 第二轮：同一个库里，prompt 里应该带上这条经验
        seen = {}

        def capture(messages, timeout=None):
            text = messages[-1]['content']
            # 只有 Scientist 的提案请求带 Allowed:；红队也会调同一个 proposer
            if 'Allowed:' in text:
                seen['text'] = text
            return {'message': {'content': '{"model":"logistic_regression","parameters":{},"hypothesis":"test hypothesis","expected_direction":"up","falsifier":"验证分数没有提升"}'},
                    'prompt_eval_count': 5, 'eval_count': 5}

        ResearchLoop(FakeSession(imbalanced=True), proposer=capture,
                     budget=Budget(max_experiments=2, max_local_calls=2),
                     patience=5, experience=self.store, task_name='t2_again').run()
        self.assertIn('Prior research experience', seen['text'])
        self.assertIn('hist_gradient_boosting', seen['text'])

    def test_no_lesson_without_improvement(self):
        # 一直提和 baseline 同分的方案（delta = 0），不该沉淀经验
        def no_gain(messages, timeout=None):
            return {'message': {'content': '{"model":"logistic_regression","parameters":{"C":1.0},"hypothesis":"test hypothesis","expected_direction":"up","falsifier":"验证分数没有提升"}'},
                    'prompt_eval_count': 5, 'eval_count': 5}

        ResearchLoop(FakeSession(imbalanced=True), proposer=no_gain,
                     budget=Budget(max_experiments=2, max_local_calls=2),
                     patience=5, experience=self.store, task_name='t2_fake').run()
        self.assertEqual(self.store.all(), [])


if __name__ == '__main__':
    unittest.main()
