import tempfile
import unittest
from pathlib import Path

from capability_memory import CapabilityMemory, bucket

BAD = {'rows': 6000, 'features': 26, 'class_count': 5, 'minority_fraction': 0.08, 'missing_fraction': 0.0}
GOOD = {'rows': 3000, 'features': 20, 'class_count': 2, 'minority_fraction': 0.27, 'missing_fraction': 0.0}


class CapabilityRouteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mem = CapabilityMemory(Path(self.tmp.name) / 'cap.sqlite3')

    def tearDown(self):
        self.mem.close()
        self.tmp.cleanup()

    def test_unseen_bucket_is_explored_on_the_cheap_tier(self):
        verdict = self.mem.route('qwen', 'model_search', BAD)
        self.assertEqual(verdict['tier'], 'low')
        self.assertEqual(verdict['reason'], 'unproven_explore')

    def test_proven_success_keeps_work_on_the_cheap_tier(self):
        for _ in range(2):
            self.mem.record('qwen', 'model_search', True, profile=GOOD)
        verdict = self.mem.route('qwen', 'model_search', GOOD)
        self.assertEqual(verdict['tier'], 'low')
        self.assertEqual(verdict['reason'], 'proven_capable')

    def test_proven_failure_moves_work_to_the_expensive_tier(self):
        for _ in range(2):
            self.mem.record('qwen', 'model_search', False, 'InvalidParameterError', profile=BAD)
        verdict = self.mem.route('qwen', 'model_search', BAD)
        self.assertEqual(verdict['tier'], 'high')
        self.assertEqual(verdict['reason'], 'proven_incapable')

    def test_evidence_is_bucket_scoped(self):
        for _ in range(2):
            self.mem.record('qwen', 'model_search', False, 'InvalidParameterError', profile=BAD)
        # same task type, different data bucket -> still explored
        verdict = self.mem.route('qwen', 'model_search', GOOD)
        self.assertEqual(verdict['tier'], 'low')

    def test_mixed_record_stays_on_the_cheap_tier(self):
        self.mem.record('qwen', 'model_search', True, profile=GOOD)
        self.mem.record('qwen', 'model_search', False, 'ValueError', profile=GOOD)
        verdict = self.mem.route('qwen', 'model_search', GOOD)
        self.assertEqual(verdict['tier'], 'low')

    def test_bucket_is_coarse_and_stable(self):
        self.assertEqual(bucket(GOOD), 'medium|cls2|featmid|clean')
        self.assertEqual(bucket(BAD), 'medium|cls5+|featmany|clean')
        self.assertEqual(bucket({}), 'small|cls2|featfew|clean')


if __name__ == '__main__':
    unittest.main()
