"""EvoUndo 版本链的测试：回滚不重训、不删记录、树能读。"""
import json
import tempfile
import unittest
from pathlib import Path

from evoundo import ACTIVE, ROLLED_BACK, VersionChain


class VersionChainTests(unittest.TestCase):
    def setUp(self):
        self.chain = VersionChain(run_id='test_run')

    def _seed(self):
        v1 = self.chain.add({'model': 'logistic_regression', 'parameters': {}}, 0.30)
        v2 = self.chain.add({'model': 'hist_gradient_boosting', 'parameters': {'max_depth': 5}}, 0.66)
        v3 = self.chain.add({'model': 'random_forest', 'parameters': {}}, 0.43)
        return v1, v2, v3

    def test_add_and_best(self):
        v1, v2, v3 = self._seed()
        self.assertEqual(self.chain.best().version_id, v2.version_id)
        self.assertEqual(self.chain.current.version_id, v3.version_id)

    def test_parent_links_form_a_chain(self):
        v1, v2, v3 = self._seed()
        self.assertIsNone(v1.parent_id)
        self.assertEqual(v2.parent_id, v1.version_id)
        self.assertEqual(v3.parent_id, v2.version_id)
        self.assertEqual(len(self.chain.active_path()), 3)

    def test_rollback_returns_to_best_and_marks_abandoned(self):
        v1, v2, v3 = self._seed()
        self.chain.rollback(v2.version_id, reason='v3 regressed')
        self.assertEqual(self.chain.current_id, v2.version_id)
        self.assertEqual(v2.status, ACTIVE)
        self.assertEqual(v3.status, ROLLED_BACK)
        self.assertIn('v3 regressed', v3.note)

    def test_rollback_keeps_the_abandoned_version(self):
        """被回滚的版本不能消失——它是"假设被否证"的证据。"""
        v1, v2, v3 = self._seed()
        before = len(self.chain.versions)
        self.chain.rollback(v2.version_id)
        self.assertEqual(len(self.chain.versions), before)
        self.assertIsNotNone(self.chain.get(v3.version_id))

    def test_rollback_to_arbitrary_history(self):
        v1, v2, v3 = self._seed()
        self.chain.rollback(v1.version_id, reason='back to baseline')
        self.assertEqual(self.chain.current_id, v1.version_id)
        self.assertEqual(v1.status, ACTIVE)
        self.assertEqual(v3.status, ROLLED_BACK)
        # 回滚到任意历史版本后，有效路径只剩 v1
        self.assertEqual([v.version_id for v in self.chain.active_path()], [v1.version_id])

    def test_rollback_to_unknown_raises(self):
        self._seed()
        with self.assertRaises(ValueError):
            self.chain.rollback('v_does_not_exist')

    def test_best_survives_rollbacks(self):
        v1, v2, v3 = self._seed()
        self.chain.rollback(v2.version_id)
        self.chain.rollback(v1.version_id)
        self.assertEqual(self.chain.best().version_id, v2.version_id)

    def test_to_dict_structure(self):
        v1, v2, v3 = self._seed()
        self.chain.rollback(v2.version_id)
        data = self.chain.to_dict()
        self.assertEqual(data['run_id'], 'test_run')
        self.assertEqual(data['current_id'], v2.version_id)
        self.assertEqual(data['best_id'], v2.version_id)
        self.assertEqual(data['rollbacks'], 1)
        self.assertEqual(len(data['versions']), 3)
        for key in ('version_id', 'parent_id', 'proposal', 'score', 'status', 'created_at'):
            self.assertIn(key, data['versions'][0])

    def test_dump_writes_json(self):
        self._seed()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'versions.json'
            self.chain.dump(path)
            loaded = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(len(loaded['versions']), 3)

    def test_render_html_contains_all_versions(self):
        v1, v2, v3 = self._seed()
        self.chain.rollback(v2.version_id)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'versions.html'
            self.chain.render_html(path)
            text = path.read_text(encoding='utf-8')
            for v in (v1, v2, v3):
                self.assertIn(v.version_id, text)
            self.assertIn('rolled_back', text)
            self.assertIn('0.6600', text)

    def test_rejects_bad_score(self):
        with self.assertRaises(ValueError):
            self.chain.add({'model': 'x'}, 99.0)

    def test_rejects_non_dict_proposal(self):
        with self.assertRaises(ValueError):
            self.chain.add('not a dict', 0.5)


if __name__ == '__main__':
    unittest.main()
