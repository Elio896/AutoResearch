"""经验进化账的测试：用过要记账，失效要降权，但永不删除。"""
import tempfile
import unittest
from pathlib import Path

from experience import ExperienceStore


class UsageAccountingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ExperienceStore(Path(self.tmp.name) / 'exp.jsonl')
        self.rid = self.store.add(
            condition={'imbalanced': True},
            advice='优先试 hist_gradient_boosting',
            evidence={'task': 't2', 'delta': 0.36},
            source_task='t2',
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_new_record_has_zero_counters(self):
        rec = self.store.all()[0]
        self.assertEqual(rec['used_count'], 0)
        self.assertEqual(rec['success_count'], 0)
        self.assertEqual(rec['contradicted_count'], 0)

    def test_record_usage_increments_used(self):
        self.store.record_usage(self.rid, improved=True)
        rec = self.store.all()[0]
        self.assertEqual(rec['used_count'], 1)
        self.assertEqual(rec['success_count'], 1)

    def test_failed_use_increments_contradiction(self):
        self.store.record_usage(self.rid, improved=False)
        rec = self.store.all()[0]
        self.assertEqual(rec['used_count'], 1)
        self.assertEqual(rec['contradicted_count'], 1)
        self.assertEqual(rec['success_count'], 0)

    def test_two_failures_retire_from_active_pool(self):
        self.store.record_usage(self.rid, improved=False)
        self.assertEqual(len(self.store.active_pool()), 1)
        self.store.record_usage(self.rid, improved=False)
        self.assertEqual(len(self.store.active_pool()), 0)   # 降权

    def test_retired_record_is_not_deleted(self):
        self.store.record_usage(self.rid, improved=False)
        self.store.record_usage(self.rid, improved=False)
        self.assertEqual(len(self.store.all()), 1)           # 还在，只是沉底
        self.assertEqual(self.store.all()[0]['contradicted_count'], 2)

    def test_unknown_id_returns_none(self):
        self.assertIsNone(self.store.record_usage('nope', improved=True))

    def test_usage_does_not_corrupt_other_records(self):
        other = self.store.add(condition={'small_sample': True}, advice='小心 GBDT',
                               evidence={'task': 't3'})
        self.store.record_usage(self.rid, improved=True)
        records = {r['id']: r for r in self.store.all()}
        self.assertEqual(records[other]['used_count'], 0)
        self.assertEqual(records[self.rid]['used_count'], 1)


class LineageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ExperienceStore(Path(self.tmp.name) / 'exp.jsonl')

    def tearDown(self):
        self.tmp.cleanup()

    def test_lineage_counts(self):
        a = self.store.add(condition={'imbalanced': True}, advice='A', evidence={'t': 1})
        b = self.store.add(condition={'small_sample': True}, advice='B', evidence={'t': 2})
        self.store.record_usage(a, improved=True)
        self.store.record_usage(a, improved=True)
        self.store.record_usage(b, improved=False)

        report = self.store.lineage()
        self.assertEqual(report['observed'], 2)
        self.assertEqual(report['ever_used'], 2)
        self.assertEqual(report['total_uses'], 3)
        self.assertEqual(report['successes'], 2)
        self.assertEqual(report['contradictions'], 1)
        self.assertAlmostEqual(report['success_rate'], 2 / 3, places=3)
        self.assertEqual(report['active'], 2)

    def test_lineage_with_no_uses(self):
        self.store.add(condition={'imbalanced': True}, advice='A', evidence={'t': 1})
        report = self.store.lineage()
        self.assertEqual(report['total_uses'], 0)
        self.assertIsNone(report['success_rate'])

    def test_retired_counted_separately(self):
        rid = self.store.add(condition={'imbalanced': True}, advice='A', evidence={'t': 1})
        for _ in range(2):
            self.store.record_usage(rid, improved=False)
        report = self.store.lineage()
        self.assertEqual(report['retired'], 1)
        self.assertEqual(report['active'], 0)

    def test_lineage_is_json_serializable(self):
        import json
        self.store.add(condition={'imbalanced': True}, advice='A', evidence={'t': 1})
        json.dumps(self.store.lineage())     # 不抛异常即可


if __name__ == '__main__':
    unittest.main()
