import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from trajectory import TrajectoryWriter, canonical_json, record_hash


def minimal_record(run_id, round_number, round_id):
    return {
        'schema_version': '1.0',
        'run_id': run_id,
        'round': round_number,
        'round_id': round_id,
    }


class TrajectoryWriterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_id = 'test_run'
        self.run_dir = Path(self.temp.name) / self.run_id

    def test_round_records_and_trace_form_a_reopenable_hash_chain(self):
        writer = TrajectoryWriter(self.run_dir, self.run_id)
        first = writer.append(minimal_record(self.run_id, 0, 'r000_baseline'))
        second = writer.append(minimal_record(self.run_id, 1, 'r001_candidate'))

        self.assertIsNone(first['chain']['prev_hash'])
        self.assertEqual(second['chain']['prev_hash'], first['chain']['hash'])
        self.assertEqual(first['chain']['hash'], record_hash(first))
        lines = [json.loads(line) for line in writer.trace_path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(lines, [
            {'round': 0, 'round_id': 'r000_baseline', 'prev_hash': None,
             'hash': first['chain']['hash']},
            {'round': 1, 'round_id': 'r001_candidate', 'prev_hash': first['chain']['hash'],
             'hash': second['chain']['hash']},
        ])

        reopened = TrajectoryWriter(self.run_dir, self.run_id)
        third = reopened.append(minimal_record(self.run_id, 2, 'r002_candidate'))
        self.assertEqual(third['chain']['prev_hash'], second['chain']['hash'])
        self.assertEqual(reopened.next_round, 3)

    def test_rejects_a_modified_persisted_record(self):
        writer = TrajectoryWriter(self.run_dir, self.run_id)
        writer.append(minimal_record(self.run_id, 0, 'r000_baseline'))
        path = self.run_dir / 'rounds' / 'r000_baseline' / 'round_record.json'
        record = json.loads(path.read_text(encoding='utf-8'))
        record['run_id'] = 'tampered'
        path.write_text(json.dumps(record), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            TrajectoryWriter(self.run_dir, self.run_id)

    def test_record_hash_uses_canonical_utf8_json_without_hash_field(self):
        record = {'chain': {'algo': 'sha256', 'prev_hash': None, 'hash': 'ignored'},
                  'hypothesis': '假设'}
        expected = 'sha256:' + hashlib.sha256(canonical_json({
            'chain': {'algo': 'sha256', 'prev_hash': None},
            'hypothesis': '假设',
        })).hexdigest()
        self.assertEqual(record_hash(record), expected)

    def test_rejects_nonsequential_rounds_and_invalid_ids(self):
        writer = TrajectoryWriter(self.run_dir, self.run_id)
        with self.assertRaisesRegex(ValueError, 'next sequence'):
            writer.append(minimal_record(self.run_id, 1, 'r001_candidate'))
        with self.assertRaisesRegex(ValueError, 'round_id'):
            writer.append(minimal_record(self.run_id, 0, 'bad-id'))


class ResearchLoopTrajectoryTests(unittest.TestCase):
    def test_loop_records_hypotheses_verdicts_budgets_and_validation_only_evidence(self):
        import pandas as pd
        from sklearn.datasets import make_classification

        from agents import Budget
        from executor import ResearchSession
        from research_loop import ResearchLoop

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            features, labels = make_classification(
                n_samples=160, n_features=6, n_informative=4, n_redundant=0,
                random_state=17,
            )
            frame = pd.DataFrame(features, columns=[f'x{i}' for i in range(features.shape[1])])
            frame['target'] = labels
            csv_path = path / 'binary.csv'
            frame.to_csv(csv_path, index=False)
            session = ResearchSession(csv_path, 'target')
            replies = iter([
                '{"model":"random_forest","parameters":{"n_estimators":20,"max_depth":4},'
                '"hypothesis":"A bounded forest improves validation F1.",'
                '"expected_direction":"up","falsifier":"Validation F1 does not improve."}',
                '{"model":"hist_gradient_boosting","parameters":{"max_depth":3,"max_iter":30},'
                '"hypothesis":"A shallow boosted model improves validation F1.",'
                '"expected_direction":"up","falsifier":"Validation F1 does not improve."}',
            ])

            def proposer(messages, timeout=None):
                return {
                    'message': {'content': next(replies)},
                    'prompt_eval_count': 11,
                    'eval_count': 7,
                }

            run_id = 'binary_test_run'
            run_dir = path / 'trajectory'
            loop = ResearchLoop(
                session,
                proposer=proposer,
                budget=Budget(max_experiments=3, max_seconds=120, max_local_calls=2),
                trajectory_dir=run_dir,
                run_id=run_id,
            )
            result = loop.run()

            records = sorted((run_dir / 'rounds').glob('*/round_record.json'))
            self.assertEqual(len(records), 3)
            parsed = [json.loads(record.read_text(encoding='utf-8')) for record in records]
            self.assertEqual([record['round'] for record in parsed], [0, 1, 2])
            self.assertEqual(parsed[1]['hypothesis']['text'],
                             'A bounded forest improves validation F1.')
            self.assertIn(parsed[1]['status'], ('kept', 'reverted'))
            self.assertEqual(parsed[1]['budget']['tokens_in'], 11)
            self.assertEqual(parsed[1]['budget']['tokens_out'], 7)
            self.assertTrue(parsed[1]['validation']['gates'])
            self.assertFalse(parsed[1]['validation']['gates'][-2]['pass'])
            self.assertEqual(result['summary']['experiments'], 3)
            self.assertEqual(len((run_dir / 'trace.jsonl').read_text(encoding='utf-8').splitlines()), 3)
