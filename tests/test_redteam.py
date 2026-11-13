"""红蓝对抗测试：质疑必须可检验，判定必须只看数字。"""
import json
import unittest

import pandas as pd

from redteam import BlueTeam, RedTeam, seed_stability, split_shift


class FakeSession:
    """固定分数的假会话，用来测检验器逻辑，不跑真模型。"""

    def __init__(self, scores=None, seed=42):
        self.scores = list(scores or [0.60, 0.61, 0.59])
        self.seed = seed
        self._i = 0
        self.x = None
        self.y = None

    def run(self, model='logistic_regression', parameters=None):
        score = self.scores[self._i % len(self.scores)]
        self._i += 1
        return {'model': model, 'parameters': parameters or {},
                'validation_score': score, 'validation_f1': score}


def proposer_with(content):
    def _call(messages, timeout=None):
        return {'message': {'content': content}}
    return _call


class RedTeamParseTests(unittest.TestCase):
    def test_accepts_valid_claim(self):
        team = RedTeam()
        accepted, rejected = team.parse(json.dumps({'claims': [{
            'claim': 'best may be seed luck',
            'check_kind': 'seed_stability',
            'check_params': {'n_seeds': 3},
            'threshold': 0.02,
        }]}))
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected, [])
        self.assertEqual(accepted[0]['check_kind'], 'seed_stability')

    def test_rejects_vague_claim_without_check(self):
        team = RedTeam()
        accepted, rejected = team.parse(json.dumps({'claims': [{
            'claim': 'I feel this overfits', 'threshold': 0.02,
        }]}))
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]['reason'], 'rejected_vague')

    def test_rejects_unknown_check_kind(self):
        team = RedTeam()
        accepted, rejected = team.parse(json.dumps({'claims': [{
            'claim': 'run it 100 times', 'check_kind': 'monte_carlo_1000',
        }]}))
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]['reason'], 'rejected_vague')

    def test_rejects_non_json(self):
        team = RedTeam()
        accepted, rejected = team.parse('我觉得不太行')
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]['reason'], 'not_json')

    def test_clamps_bad_threshold(self):
        team = RedTeam()
        accepted, _ = team.parse(json.dumps({'claims': [{
            'claim': 'x', 'check_kind': 'seed_stability', 'threshold': 'big',
        }]}))
        self.assertEqual(accepted[0]['threshold'], 0.02)


class CheckTests(unittest.TestCase):
    def test_seed_stability_detects_unstable(self):
        session = FakeSession([0.50, 0.70, 0.90])
        result = seed_stability(session, {'model': 'rf', 'parameters': {}}, n_seeds=3, threshold=0.02)
        self.assertTrue(result.passed)              # 波动大 -> 质疑成立
        self.assertGreater(result.measured, 0.02)

    def test_seed_stability_accepts_stable(self):
        session = FakeSession([0.600, 0.601, 0.599])
        result = seed_stability(session, {'model': 'rf', 'parameters': {}}, n_seeds=3, threshold=0.02)
        self.assertFalse(result.passed)

    def test_split_shift_detects_drift(self):
        class SplitSession(FakeSession):
            def __init__(self):
                super().__init__([0.60, 0.80])
                self.y = pd.Series([0, 1] * 40)
                self.seed = 42
                self.train_indices = list(range(48))
                self.validation_indices = list(range(48, 64))

        session = SplitSession()
        original_train = list(session.train_indices)
        original_validation = list(session.validation_indices)
        result = split_shift(session, {'model': 'rf', 'parameters': {}}, threshold=0.02)
        self.assertTrue(result.passed)
        self.assertEqual(session.train_indices, original_train)
        self.assertEqual(session.validation_indices, original_validation)
        self.assertEqual(session.seed, 42)


class BlueTeamTests(unittest.TestCase):
    def test_upheld_claim_means_rollback(self):
        blue = BlueTeam()
        session = FakeSession([0.50, 0.70, 0.90])
        claim = {'claim': 'unstable', 'check_kind': 'seed_stability',
                 'check_params': {'n_seeds': 3, 'proposal': {'model': 'rf', 'parameters': {}}},
                 'threshold': 0.02}
        verdict = blue.verify(claim, session)
        self.assertTrue(verdict['supported'])
        summary = blue.summarize([verdict])
        self.assertTrue(summary['should_rollback'])
        self.assertEqual(summary['conclusion'], 'claims_upheld')

    def test_rejected_claim_keeps_best(self):
        blue = BlueTeam()
        session = FakeSession([0.600, 0.601, 0.599])
        claim = {'claim': 'unstable', 'check_kind': 'seed_stability',
                 'check_params': {'n_seeds': 3, 'proposal': {'model': 'rf', 'parameters': {}}},
                 'threshold': 0.02}
        verdict = blue.verify(claim, session)
        self.assertFalse(verdict['supported'])
        summary = blue.summarize([verdict])
        self.assertFalse(summary['should_rollback'])

    def test_unknown_check_is_not_supported(self):
        blue = BlueTeam()
        verdict = blue.verify({'claim': 'x', 'check_kind': 'nope'}, FakeSession())
        self.assertIsNone(verdict['supported'])

    def test_empty_verdicts_do_not_rollback(self):
        summary = BlueTeam().summarize([])
        self.assertFalse(summary['should_rollback'])
        self.assertEqual(summary['conclusion'], 'no_check_ran')

    def test_split_shift_without_split_controls_is_unsupported(self):
        verdict = BlueTeam().verify({
            'claim': 'split sensitive',
            'check_kind': 'split_shift',
            'check_params': {'proposal': {'model': 'rf'}},
        }, FakeSession())
        self.assertIsNone(verdict['supported'])


if __name__ == '__main__':
    unittest.main()
