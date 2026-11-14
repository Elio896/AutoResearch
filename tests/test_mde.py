"""MDE 判定的测试。

二德子给的算法：
    d_i = score_c(seed_i) − score_b(seed_i)
    MDE = t(0.95, S−1) × std(d)/sqrt(S)
    improved / regressed / flat
"""
import math
import unittest

from mde import compare, required_seeds, t_critical


class MCETests(unittest.TestCase):
    def test_t_critical_matches_table(self):
        """单侧和双侧是两张表，别再混用。

        猫 2026-11-14 指出：同时判升和降两个方向时用单侧临界值，
        合计误判率约 10%，不是 5%。所以默认走双侧。
        """
        self.assertAlmostEqual(t_critical(4, two_sided=True), 2.776, places=3)   # S=5
        self.assertAlmostEqual(t_critical(4, two_sided=False), 2.132, places=3)  # S=5
        self.assertAlmostEqual(t_critical(2, two_sided=True), 4.303, places=3)   # S=3
        self.assertAlmostEqual(t_critical(2, two_sided=False), 2.920, places=3)  # S=3

    def test_two_sided_is_stricter(self):
        """双侧比单侧更严——这是改它的原因。"""
        for dof in (2, 4, 9):
            self.assertGreater(t_critical(dof, two_sided=True),
                               t_critical(dof, two_sided=False))

    def test_same_data_more_conservative_under_two_sided(self):
        """同一组数据，双侧判 flat 的，单侧可能判 improved。"""
        cand = [0.951, 0.948, 0.955, 0.949, 0.952]
        base = [0.949, 0.947, 0.953, 0.948, 0.951]
        one = compare(cand, base, two_sided=False)
        two = compare(cand, base, two_sided=True)
        self.assertGreater(two['mde'], one['mde'])
        self.assertTrue(two['two_sided'])

    def test_consistent_small_gain_is_improved(self):
        """每个种子都略好一点——差值稳定，算真提升。"""
        cand = [0.951, 0.948, 0.955, 0.949, 0.952]
        base = [0.949, 0.947, 0.953, 0.948, 0.951]
        result = compare(cand, base)
        self.assertEqual(result['verdict'], 'improved')
        self.assertTrue(result['delta_significant'])
        self.assertGreater(result['delta_vs_best'], 0)
        self.assertLess(result['mde'], result['delta_vs_best'])

    def test_noise_around_zero_is_flat(self):
        """候选偶尔高偶尔低，均值接近 0——不许当提升。"""
        cand = [0.95, 0.94, 0.96, 0.93, 0.97]
        base = [0.95, 0.95, 0.95, 0.95, 0.95]
        result = compare(cand, base)
        self.assertEqual(result['verdict'], 'flat')
        self.assertFalse(result['delta_significant'])

    def test_real_regression_is_detected(self):
        cand = [0.90, 0.89, 0.91, 0.88, 0.90]
        base = [0.95, 0.94, 0.96, 0.93, 0.95]
        result = compare(cand, base)
        self.assertEqual(result['verdict'], 'regressed')

    def test_small_drop_within_noise_does_not_rollback(self):
        """差得少也算 flat——不因噪声回滚（对称性）。

        注意差值必须有波动：如果 5 个种子都稳定低同样的量，std=0、MDE=0，
        那确实该判 regressed——稳定变差就是变差。
        """
        cand = [0.948, 0.944, 0.951, 0.945, 0.950]
        base = [0.949, 0.947, 0.951, 0.948, 0.950]
        result = compare(cand, base)
        self.assertEqual(result['verdict'], 'flat')
        self.assertLess(result['delta_vs_best'], 0)     # 确实略低
        self.assertFalse(result['delta_significant'])   # 但不显著

    def test_stable_consistent_small_drop_is_regressed(self):
        """反过来：所有种子都稳定低同样一截，那是真退步。"""
        cand = [0.947, 0.945, 0.949, 0.946, 0.948]
        base = [0.949, 0.947, 0.951, 0.948, 0.950]
        result = compare(cand, base)
        self.assertEqual(result['verdict'], 'regressed')

    def test_single_seed_cannot_judge(self):
        result = compare([0.95], [0.94])
        self.assertEqual(result['verdict'], 'flat')
        self.assertIsNone(result['mde'])
        self.assertIn('只有一个种子', result['note'])

    def test_mismatched_lengths_rejected(self):
        with self.assertRaises(ValueError):
            compare([0.9, 0.91], [0.9])

    def test_fields_match_trajectory_spec(self):
        """字段名要能直接喂给《轨迹规范》的 per_fold/std/mde/delta_significant。"""
        result = compare([0.951, 0.948, 0.955, 0.949, 0.952],
                         [0.949, 0.947, 0.953, 0.948, 0.951])
        for key in ('per_seed', 'seed_mean', 'seed_std', 'mde',
                    'delta_vs_best', 'delta_significant', 'seeds'):
            self.assertIn(key, result)

    def test_required_seeds_is_monotone(self):
        """想发现越小的差异，需要越多种子。"""
        small = required_seeds(0.001, 0.01)
        large = required_seeds(0.01, 0.01)
        self.assertGreater(small, large)

    def test_mde_shrinks_with_more_seeds(self):
        """种子越多，可检测阈值越低（这是 S=5 比 S=3 好的量化理由）。"""
        deltas = [0.002, 0.001, 0.003, 0.002, 0.001, 0.002, 0.003, 0.001]
        base = [0.95] * 8
        cand5 = [b + d for b, d in zip(base[:5], deltas[:5])]
        cand8 = [b + d for b, d in zip(base, deltas)]
        mde5 = compare(cand5, base[:5])['mde']
        mde8 = compare(cand8, base)['mde']
        self.assertLess(mde8, mde5)


if __name__ == '__main__':
    unittest.main()
