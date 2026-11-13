"""自迭代对照实验的测试：只测可测的部分（判据逻辑、字段完整性）。

真跑模型那段在 tools/run_selfiter.py 里，由人工跑一次并留档，
不放进单元测试——单元测试不该依赖本地模型和分钟级耗时。
"""
import unittest

from tools.run_selfiter import trend


class TrendTests(unittest.TestCase):
    def test_decreasing(self):
        rows = [{'local_tokens': 1000}, {'local_tokens': 800}, {'local_tokens': 600}]
        self.assertAlmostEqual(trend(rows, 'local_tokens'), -400.0, places=4)

    def test_increasing(self):
        rows = [{'local_tokens': 600}, {'local_tokens': 800}, {'local_tokens': 1000}]
        self.assertAlmostEqual(trend(rows, 'local_tokens'), 400.0, places=4)

    def test_flat(self):
        rows = [{'x': 1.0}, {'x': 1.0}]
        self.assertEqual(trend(rows, 'x'), 0.0)

    def test_not_enough_rounds(self):
        self.assertIsNone(trend([{'x': 1}], 'x'))
        self.assertIsNone(trend([], 'x'))

    def test_missing_key_ignored(self):
        rows = [{'x': 1}, {'y': 2}, {'x': 3}]
        # 只有两个有效值 1 和 3 -> delta = 2
        self.assertAlmostEqual(trend(rows, 'x'), 2.0, places=4)


class JudgementLogicTests(unittest.TestCase):
    """复刻 run_selfiter 里"哪条判据算成立"的规则，确保语义没被改歪。"""

    @staticmethod
    def judge(treatment_delta, control_delta):
        return treatment_delta < control_delta

    def test_treatment_dropped_more_is_improved(self):
        self.assertTrue(self.judge(-400, -100))

    def test_treatment_rose_while_control_fell_is_not_improved(self):
        self.assertFalse(self.judge(+300, -50))

    def test_both_flat_is_not_improved(self):
        self.assertFalse(self.judge(0, 0))


if __name__ == '__main__':
    unittest.main()
