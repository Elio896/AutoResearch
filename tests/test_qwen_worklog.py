"""千问结构化工作记录的验收测试：引用校验、字段缺失、工具白名单、真实检查。"""
from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

import qwen_worklog
from qwen_worklog import (WorklogStore, build_record, evidence_from_session,
                          parse_model_record, run_checks, run_operation,
                          validate_record)


class _Session:
    """最小可用的 ResearchSession 替身：只要 x / y / train_indices / target。"""

    def __init__(self, frame: pd.DataFrame, target: str = 'target', imbalance: float = 0.5):
        self.target = target
        self.x = frame.drop(columns=[target])
        rng = np.random.default_rng(3)
        self.y = pd.Series((rng.random(len(frame)) < imbalance).astype(int))
        self.train_indices = np.arange(len(frame))


def make_frame(n: int = 120, missing: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({
        'a': rng.normal(size=n),
        'b': rng.normal(size=n),
        'c': rng.integers(0, 3, n),
        'target': (rng.random(n) < 0.3).astype(int),
    })
    if missing:
        frame.loc[frame.sample(frac=0.2, random_state=1).index, 'a'] = np.nan
    return frame


def valid_record(evidence: dict) -> dict:
    rows = evidence['profile.rows']['value']
    return {
        'task_kind': 'basic_operation',
        'observations': [
            {'fact': f'训练集共 {rows} 行', 'evidence_ref': 'profile.rows'},
            {'fact': '目标列只有两个取值', 'evidence_ref': 'profile.class_count'},
        ],
        'operation': {'tool': 'profile.missing', 'arguments': {}},
        'brief_rationale': '按已给定规则执行基础操作',
        'verification_checks': ['rows_readable', 'no_columns_dropped',
                                'target_column_untouched', 'operation_returned_rows'],
        'needs_specialist': False,
    }


class EvidenceTests(unittest.TestCase):
    def test_evidence_is_computed_and_carries_source(self):
        session = _Session(make_frame())
        evidence = evidence_from_session(session)
        self.assertEqual(evidence['profile.rows']['value'], 120)
        self.assertEqual(evidence['profile.rows']['source'], 'tool')
        self.assertIn('profile.missing_fraction', evidence)
        self.assertIn('profile.unique_top', evidence)

    def test_evidence_follows_the_input(self):
        small = evidence_from_session(_Session(make_frame(120)))
        big = evidence_from_session(_Session(make_frame(400)))
        self.assertNotEqual(small['profile.rows']['value'], big['profile.rows']['value'])

        clean = evidence_from_session(_Session(make_frame(120, missing=False)))
        dirty = evidence_from_session(_Session(make_frame(120, missing=True)))
        self.assertGreater(dirty['profile.missing_fraction']['value'],
                           clean['profile.missing_fraction']['value'])


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.session = _Session(make_frame())
        self.evidence = evidence_from_session(self.session)

    def test_valid_record_passes(self):
        ok, errors = validate_record(valid_record(self.evidence), self.evidence)
        self.assertTrue(ok, errors)

    def test_invented_number_is_rejected(self):
        record = valid_record(self.evidence)
        record['observations'] = [{'fact': '训练集共 999 行', 'evidence_ref': 'profile.rows'}]
        ok, errors = validate_record(record, self.evidence)
        self.assertFalse(ok)
        self.assertTrue(any('999' in e for e in errors), errors)

    def test_number_may_be_written_as_percent(self):
        session = _Session(make_frame(200, missing=True))
        evidence = evidence_from_session(session)
        rate = evidence['profile.missing_fraction']['value']
        fact = f'缺失比例约 {rate * 100:.1f}%'
        record = valid_record(evidence)
        record['observations'] = [{'fact': fact, 'evidence_ref': 'profile.missing_fraction'}]
        ok, errors = validate_record(record, evidence)
        self.assertTrue(ok, errors)

    def test_missing_fields_are_reported_not_filled(self):
        record = valid_record(self.evidence)
        del record['needs_specialist']
        del record['task_kind']
        ok, errors = validate_record(record, self.evidence)
        self.assertFalse(ok)
        self.assertTrue(any('task_kind' in e for e in errors))
        self.assertTrue(any('needs_specialist' in e for e in errors))

    def test_unknown_tool_and_check_rejected(self):
        record = valid_record(self.evidence)
        record['operation'] = {'tool': 'os.system', 'arguments': {'cmd': 'rm -rf /'}}
        record['verification_checks'] = ['looks_good']
        ok, errors = validate_record(record, self.evidence)
        self.assertFalse(ok)
        self.assertTrue(any('os.system' in e for e in errors))
        self.assertTrue(any('looks_good' in e for e in errors))

    def test_needs_specialist_must_be_boolean(self):
        record = valid_record(self.evidence)
        record['needs_specialist'] = 'yes please'
        ok, errors = validate_record(record, self.evidence)
        self.assertFalse(ok)
        self.assertTrue(any('needs_specialist' in e for e in errors))


class ParseAndRunTests(unittest.TestCase):
    def setUp(self):
        self.session = _Session(make_frame(150))

    def test_parse_wraps_json_in_chatter(self):
        evidence = evidence_from_session(self.session)
        raw = '好的，这是记录：' + json.dumps(valid_record(evidence), ensure_ascii=False) + ' 完成。'
        result = parse_model_record(raw, evidence)
        self.assertEqual(result['status'], 'ok', result['errors'])
        self.assertIsNotNone(result['model_output'])

    def test_non_json_is_invalid_without_fabrication(self):
        result = parse_model_record('我不太确定该怎么写。', evidence_from_session(self.session))
        self.assertEqual(result['status'], 'invalid')
        self.assertIsNone(result['model_output'])
        self.assertTrue(result['errors'])

    def test_empty_output_is_invalid(self):
        result = parse_model_record('', evidence_from_session(self.session))
        self.assertEqual(result['status'], 'invalid')
        self.assertIn('没有返回任何内容', result['errors'][0])

    def test_operation_and_checks_really_run(self):
        evidence = evidence_from_session(self.session)
        record = valid_record(evidence)
        tool = run_operation(record, self.session)
        self.assertTrue(tool['ok'], tool)
        self.assertEqual(tool['source'], 'tool')
        self.assertEqual(tool['evidence_ref'], 'profile.missing_top')

        checks = run_checks(record, self.session, tool)
        self.assertEqual(len(checks), 4)
        self.assertTrue(all(item['passed'] for item in checks), checks)

    def test_check_fails_when_target_leaks_into_features(self):
        frame = make_frame(80)
        leaked = frame.copy()
        leaked['target_copy'] = leaked['target']
        session = _Session(leaked)
        session.target = 'target_copy'          # 目标列还在特征里，这条检查必须失败
        evidence = evidence_from_session(session)
        record = valid_record(evidence)
        checks = run_checks(record, session, {'ok': True})
        verdict = {c['check']: c['passed'] for c in checks}
        self.assertFalse(verdict['target_column_untouched'])

    def test_build_record_separates_model_and_tool(self):
        evidence = evidence_from_session(self.session)
        raw = json.dumps(valid_record(evidence), ensure_ascii=False)
        result = build_record(raw, self.session)
        self.assertEqual(result['status'], 'ok')
        self.assertIn('model_output', result)
        self.assertIn('tool_computation', result)
        self.assertEqual(result['tool_computation']['source'], 'tool')
        self.assertTrue(all(c['passed'] for c in result['check_results']))

    def test_worklog_appends_and_reads_back(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            store = WorklogStore(Path(tmp) / 'log.jsonl')
            evidence = evidence_from_session(self.session)
            record = build_record(json.dumps(valid_record(evidence), ensure_ascii=False), self.session)
            store.append(record, run_id='run_1')
            store.append(record, run_id='run_2')
            rows = store.all()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['run_id'], 'run_1')
            self.assertEqual(rows[1]['status'], 'ok')


if __name__ == '__main__':
    unittest.main()
