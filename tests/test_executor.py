import importlib.util
import tempfile
import unittest
from pathlib import Path


class ExecutorTests(unittest.TestCase):
    def test_executor_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('executor'))

    def test_real_training_and_reserved_test_set(self):
        from sklearn.datasets import load_iris
        from executor import ResearchSession
        with tempfile.TemporaryDirectory() as directory:
            data = load_iris(as_frame=True)
            path = Path(directory) / 'iris.csv'
            data.frame.to_csv(path, index=False)
            session = ResearchSession(path, 'target', seed=42)
            result = session.run('logistic_regression', {})
            self.assertGreater(result['validation_f1'], 0.7)
            self.assertNotIn('test_f1', result)
            self.assertEqual(sum(session.split_sizes.values()), 150)
            self.assertEqual(len(set(session.train_indices) & set(session.test_indices)), 0)
            self.assertEqual(len(set(session.validation_indices) & set(session.test_indices)), 0)
            final = session.finalize(result['experiment_id'])
            self.assertGreater(final['test_f1'], 0.7)
            with self.assertRaises(RuntimeError):
                session.finalize(result['experiment_id'])
            with self.assertRaises(RuntimeError):
                session.run('random_forest', {})

    def test_missing_target_rejected(self):
        from executor import ResearchSession
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.csv'
            path.write_text('feature,label\n1,yes\n2,\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                ResearchSession(path, 'label')

    def test_balanced_random_forest_with_string_labels(self):
        from sklearn.datasets import load_iris
        from executor import ResearchSession
        with tempfile.TemporaryDirectory() as directory:
            frame = load_iris(as_frame=True).frame
            path = Path(directory) / 'iris.csv'
            for mapping in [{0: '0', 1: '1', 2: '2'},
                            {0: 'setosa', 1: 'versicolor', 2: 'virginica'}]:
                with self.subTest(mapping=mapping):
                    named = frame.copy()
                    named['target'] = named['target'].map(mapping)
                    named.to_csv(path, index=False)
                    session = ResearchSession(path, 'target')
                    result = session.run('random_forest', {'n_estimators': 100, 'max_depth': 10,
                                                         'min_samples_leaf': 1, 'class_weight': 'balanced'})
                    self.assertGreater(result['validation_f1'], 0.7)
                    self.assertFalse(session.closed)


if __name__ == '__main__':
    unittest.main()
