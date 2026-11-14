import tempfile
import unittest
from pathlib import Path

from capability_memory import CapabilityMemory


class CapabilityMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mem = CapabilityMemory(Path(self.tmp.name) / 'cap.sqlite3')

    def tearDown(self):
        self.mem.close()
        self.tmp.cleanup()

    def test_no_skip_before_enough_evidence(self):
        self.mem.record('qwen-local', 'model_search', False, 'InvalidParameterError')
        self.assertIsNone(self.mem.should_skip('qwen-local', 'model_search'))

    def test_skip_after_repeated_failures(self):
        self.mem.record('qwen-local', 'model_search', False, 'InvalidParameterError')
        self.mem.record('qwen-local', 'model_search', False, 'ValueError')
        verdict = self.mem.should_skip('qwen-local', 'model_search')
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict['attempts'], 2)
        self.assertEqual(verdict['failures'], 2)
        self.assertEqual(verdict['last_error'], 'ValueError')

    def test_success_ratio_keeps_a_usable_model(self):
        for _ in range(3):
            self.mem.record('qwen-local', 'model_search', True)
        self.mem.record('qwen-local', 'model_search', False, 'ValueError')
        self.assertIsNone(self.mem.should_skip('qwen-local', 'model_search'))

    def test_one_bad_model_does_not_poison_another(self):
        self.mem.record('qwen-local', 'model_search', False, 'ValueError')
        self.mem.record('qwen-local', 'model_search', False, 'ValueError')
        self.assertIsNotNone(self.mem.should_skip('qwen-local', 'model_search'))
        self.assertIsNone(self.mem.should_skip('deepseek-flash', 'model_search'))

    def test_failure_in_one_category_does_not_block_another(self):
        self.mem.record('qwen-local', 'model_search', False, 'ValueError')
        self.mem.record('qwen-local', 'model_search', False, 'ValueError')
        self.assertIsNone(self.mem.should_skip('qwen-local', 'preprocessing'))

    def test_summary_is_compact_and_ordered(self):
        for _ in range(3):
            self.mem.record('qwen-local', 'model_search', False, 'ValueError')
        self.mem.record('deepseek-flash', 'model_search', True)
        rows = self.mem.summary()
        self.assertEqual(rows[0]['model'], 'qwen-local')
        self.assertEqual(rows[0]['failure_rate'], 1.0)
        self.assertLessEqual(len(rows), 6)

    def test_survives_reopen(self):
        self.mem.record('qwen-local', 'model_search', False, 'ValueError')
        self.mem.record('qwen-local', 'model_search', False, 'ValueError')
        self.mem.close()
        again = CapabilityMemory(Path(self.tmp.name) / 'cap.sqlite3')
        try:
            self.assertEqual(again.stats('qwen-local', 'model_search')['attempts'], 2)
        finally:
            again.close()


if __name__ == '__main__':
    unittest.main()
