import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TaskStoreTests(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from task_store import TaskStore
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.csv = Path(self.temp.name) / 'iris.csv'
        load_iris(as_frame=True).frame.to_csv(self.csv, index=False)
        self.db = Path(self.temp.name) / 'tasks.sqlite3'
        self.store = TaskStore(self.db)

    def test_resume_history_and_budget_with_new_store_instance(self):
        from task_store import TaskStore
        task = self.store.create(self.csv, 'target', max_experiments=2)
        first = self.store.run(task, 'logistic_regression', {})
        second_store = TaskStore(self.db)
        second = second_store.run(task, 'random_forest', {'n_estimators': 10})
        self.assertNotEqual(first['experiment_id'], second['experiment_id'])
        self.assertEqual(len(second_store.show(task)['experiments']), 2)
        with self.assertRaises(RuntimeError):
            second_store.run(task, 'logistic_regression', {'C': 0.5})

    def test_final_test_cannot_be_repeated_by_another_process(self):
        task = self.store.create(self.csv, 'target')
        self.store.run(task, 'logistic_regression', {})
        final = self.store.finalize(task)
        self.assertGreater(final['test_f1'], 0.7)
        script = 'from task_store import TaskStore; import sys; TaskStore(sys.argv[1]).finalize(sys.argv[2])'
        process = subprocess.run([sys.executable, '-c', script, str(self.db), task],
                                 capture_output=True, text=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('not open', process.stderr)
        with self.assertRaises(RuntimeError):
            self.store.run(task, 'random_forest', {})

    def test_changed_data_is_rejected_without_consuming_final_test(self):
        task = self.store.create(self.csv, 'target')
        self.store.run(task, 'logistic_regression', {})
        with self.csv.open('a') as file:
            file.write('\n')
        with self.assertRaises(ValueError):
            self.store.finalize(task)
        self.assertEqual(self.store.show(task)['state'], 'open')

    def test_failed_experiment_is_recorded_and_consumes_budget(self):
        task = self.store.create(self.csv, 'target', max_experiments=1)
        # Valid whitelist, but the synthetic executor error simulates a fit failure.
        from unittest.mock import patch
        with patch('executor.ResearchSession.run', side_effect=ValueError('do not log raw details')):
            with self.assertRaises(RuntimeError):
                self.store.run(task, 'logistic_regression', {})
        data = self.store.show(task)
        self.assertEqual(data['experiments'][0]['status'], 'failed')
        self.assertNotIn('do not log raw details', json.dumps(data))
        with self.assertRaises(RuntimeError):
            self.store.run(task, 'random_forest', {})

    def test_orchestrated_experiments_use_persistent_session(self):
        from task_store import StoredResearchSession
        from orchestrator import ResearchOrchestrator
        task = self.store.create(self.csv, 'target', max_experiments=2)
        def propose(messages, timeout):
            return {'message': {'content': '{"model":"random_forest","parameters":{"max_depth":3}}'}}
        result = ResearchOrchestrator(StoredResearchSession(self.store, task), propose,
            max_experiments=2, emit=lambda event: self.store.add_event(task, event)).run()
        self.store.save_report(task, result)
        restored = self.store.show(task)
        self.assertEqual(len(restored['experiments']), 2)
        self.assertEqual(restored['research_report']['best_experiment_id'], result['best_experiment_id'])
        self.assertTrue(restored['events'])

    def test_failed_final_evaluation_remains_sealed_after_restart(self):
        from unittest.mock import patch
        from task_store import TaskStore
        task = self.store.create(self.csv, 'target')
        self.store.run(task, 'logistic_regression', {})
        with patch('executor.ResearchSession.finalize', side_effect=RuntimeError('simulated crash')):
            with self.assertRaises(RuntimeError):
                self.store.finalize(task)
        self.assertEqual(self.store.show(task)['state'], 'final_failed')
        with self.assertRaises(RuntimeError):
            TaskStore(self.db).finalize(task)
