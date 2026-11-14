"""第四步验收测试：候选必须有假设、有否证条件、只动白名单，且不动正式策略。"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from candidate_proposer import (CandidateRejected, build_prompt, propose_from_failures,
                                simulated_client, validate_candidate)
from strategy_store import StrategyStore

ACTIVE = {
    'classifier_prompt': '按类别分布选择指标。',
    'planning_prompt': '先看缺失与唯一值。',
    'candidate_selection_rules': [{'when': 'imbalanced', 'prefer': 'class_weight'}],
}

FAILURES = {
    'insufficient': False,
    'records': [{
        'failure_type': 'repeated_proposal',
        'run_id': 'run_stuck',
        'event_refs': ['event_007', 'event_013'],
        'measurements': {'duplicate_count': 2},
    }],
    'insufficient_reasons': [],
}

GOOD = {
    'changes': {'planning_prompt': '先查事后字段与 ID 列，再定指标。'},
    'hypothesis': {
        'what': '规划里加排雷步骤',
        'why': '重复提案集中在同一模型族调参',
        'expected_improvement': '重复提案率下降',
        'falsifier': '独立任务上重复率不降反升',
        'side_effects': '单次运行多一次列扫描',
    },
    'target_failure_types': ['repeated_proposal'],
    'evidence_run_ids': ['run_stuck'],
}


def with_changes(**overrides) -> dict:
    payload = json.loads(json.dumps(GOOD))
    payload.update(overrides)
    return payload


def without_hypothesis_key(key: str) -> dict:
    payload = json.loads(json.dumps(GOOD))
    payload['hypothesis'].pop(key)
    return payload


class ValidationTests(unittest.TestCase):
    def test_good_candidate_passes(self):
        ok, errors = validate_candidate(GOOD, ACTIVE)
        self.assertTrue(ok, errors)

    def test_forbidden_field_rejected(self):
        bad = with_changes(changes={'scoring_function': 'accuracy'})
        ok, errors = validate_candidate(bad, ACTIVE)
        self.assertFalse(ok)
        self.assertTrue(any('scoring_function' in e for e in errors))

    def test_each_hypothesis_field_is_required(self):
        for key in ('what', 'why', 'expected_improvement', 'falsifier', 'side_effects'):
            ok, errors = validate_candidate(without_hypothesis_key(key), ACTIVE)
            self.assertFalse(ok, key)
            self.assertTrue(any(key in e for e in errors), (key, errors))

    def test_blank_falsifier_rejected(self):
        payload = json.loads(json.dumps(GOOD))
        payload['hypothesis']['falsifier'] = '   '
        ok, errors = validate_candidate(payload, ACTIVE)
        self.assertFalse(ok)
        self.assertTrue(any('falsifier' in e for e in errors))

    def test_multiple_fields_rejected(self):
        payload = json.loads(json.dumps(GOOD))
        payload['changes']['classifier_prompt'] = '顺便把分类提示词也改了'
        ok, errors = validate_candidate(payload, ACTIVE)
        self.assertFalse(ok)
        self.assertTrue(any('修改方向' in e for e in errors))

    def test_identical_to_active_rejected(self):
        payload = with_changes(changes={'planning_prompt': ACTIVE['planning_prompt']})
        ok, errors = validate_candidate(payload, ACTIVE)
        self.assertFalse(ok)
        self.assertTrue(any('一模一样' in e for e in errors))

    def test_wrong_type_rejected(self):
        payload = with_changes(changes={'candidate_selection_rules': 'not-a-list'})
        ok, errors = validate_candidate(payload, ACTIVE)
        self.assertFalse(ok)


class ProposeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='candidate-')
        self.store = StrategyStore(Path(self.tmp) / 'store.json')
        self.active = self.store.bootstrap(ACTIVE, reason='基线')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_candidate_created_and_active_untouched(self):
        version = propose_from_failures(self.store, FAILURES, simulated_client(GOOD),
                                        proposed_by='deepseek', simulated=True)
        self.assertEqual(version['status'], 'candidate')
        self.assertIsNotNone(version['hypothesis'])
        self.assertTrue(version['simulated'])
        self.assertEqual(self.store.active()['strategy_id'], self.active['strategy_id'])
        self.assertEqual(self.store.active()['changes'], ACTIVE)

        diff = self.store.diff(self.active['strategy_id'], version['strategy_id'])
        self.assertEqual(diff['changed_fields'], ['planning_prompt'])

    def test_insufficient_evidence_refuses(self):
        thin = {'insufficient': True, 'records': [], 'insufficient_reasons': ['实验只有 1 次']}
        with self.assertRaises(CandidateRejected):
            propose_from_failures(self.store, thin, simulated_client(GOOD))

    def test_no_failure_records_refuses(self):
        empty = {'insufficient': False, 'records': [], 'insufficient_reasons': []}
        with self.assertRaises(CandidateRejected):
            propose_from_failures(self.store, empty, simulated_client(GOOD))

    def test_bad_payload_does_not_touch_store(self):
        bad = json.loads(json.dumps(GOOD))
        bad['hypothesis'].pop('falsifier')
        with self.assertRaises(CandidateRejected):
            propose_from_failures(self.store, FAILURES, simulated_client(bad))
        self.assertEqual(len(self.store.versions()), 1)
        self.assertEqual(self.store.events()[-1]['type'], 'bootstrap')

    def test_garbage_output_refused(self):
        def junk(messages, timeout=None):
            return '我想想……还是算了吧'
        with self.assertRaises(CandidateRejected):
            propose_from_failures(self.store, FAILURES, junk)

    def test_prompt_carries_evidence_and_limits(self):
        messages = build_prompt(FAILURES, self.active)
        text = messages[0]['content']
        self.assertIn('event_007', text)
        self.assertIn('repeated_proposal', text)
        self.assertIn('classifier_prompt', text)
        self.assertIn('falsifier', text)


if __name__ == '__main__':
    unittest.main()
