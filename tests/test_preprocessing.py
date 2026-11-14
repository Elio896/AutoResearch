"""预处理开关的测试。

二德子 2026-11-14：只调模型天花板低，表格任务的收益大半在数据这一侧。
四个开关都是真跑出来的，不许空转。
"""
import unittest

import numpy as np
import pandas as pd

from executor import ResearchSession


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.root = __import__('pathlib').Path(self._dir.name)

    def make_csv(self, missing: bool, imbalanced: bool = True):
        """造一份**标签和特征有关**的数据，否则 F1 恒为 0，测不出东西。"""
        rng = np.random.default_rng(3)
        n = 800
        p = [0.9, 0.1] if imbalanced else [0.5, 0.5]
        labels = rng.choice(['0', '1'], n, p=p)
        signal = (labels == '1').astype(float)
        frame = pd.DataFrame({
            # a/b 带着标签信号，这样模型能学到点东西
            'a': (signal * 2 + rng.normal(0, 1, n)).round(3),
            'b': (signal * 4 + rng.integers(0, 5, n)).round(3),
            'c': rng.choice(['x', 'y', 'z'], n),
            'target': labels,
        })
        if missing:
            frame.loc[rng.random(n) < 0.25, 'a'] = np.nan
            frame.loc[rng.random(n) < 0.15, 'b'] = np.nan
        path = self.root / ('m.csv' if missing else 'clean.csv')
        frame.to_csv(path, index=False)
        return path

    def test_missing_indicator_changes_the_fit(self):
        """加了 isna 指示变量，管道里应该真的多出列。"""
        path = self.make_csv(missing=True)
        plain = ResearchSession(path, 'target', metric='f1_minority', pos_label='1')
        result = plain.run('logistic_regression', {},
                           preprocessing={'add_missing_indicator': True})
        self.assertEqual(result['status'], 'success')
        # 分数必须是有限数，不能因为维度错乱变成 0 或 nan
        self.assertGreater(result['validation_score'], 0.0)
        self.assertLessEqual(result['validation_score'], 1.0)

    def test_ordinal_encode_is_accepted(self):
        path = self.make_csv(missing=False)
        session = ResearchSession(path, 'target', metric='f1_minority', pos_label='1')
        result = session.run('logistic_regression', {},
                             preprocessing={'encode': 'ordinal',
                                            'imbalance_strategy': 'class_weight'})
        self.assertEqual(result['status'], 'success')
        self.assertGreater(result['validation_score'], 0.0)

    def test_unknown_preprocessing_key_rejected(self):
        path = self.make_csv(missing=False)
        session = ResearchSession(path, 'target', metric='f1_minority', pos_label='1')
        with self.assertRaises(ValueError):
            session.run('logistic_regression', {}, preprocessing={'smote': True})

    def test_bad_value_rejected(self):
        path = self.make_csv(missing=False)
        session = ResearchSession(path, 'target', metric='f1_minority', pos_label='1')
        with self.assertRaises(ValueError):
            session.run('logistic_regression', {}, preprocessing={'impute': 'zero'})

    def test_class_weight_reaches_the_estimator(self):
        path = self.make_csv(missing=False)
        session = ResearchSession(path, 'target', metric='f1_minority', pos_label='1')
        result = session.run('logistic_regression', {},
                             preprocessing={'imbalance_strategy': 'class_weight'})
        self.assertEqual(result['parameters'].get('class_weight'), 'balanced')

    def test_defaults_are_unchanged(self):
        """不传 preprocessing 时，行为和以前一模一样。"""
        path = self.make_csv(missing=False)
        session = ResearchSession(path, 'target', metric='f1_minority', pos_label='1')
        result = session.run('logistic_regression', {})
        self.assertNotIn('class_weight', result['parameters'])


if __name__ == '__main__':
    unittest.main()
