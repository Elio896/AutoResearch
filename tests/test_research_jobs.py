import tempfile
import unittest
from pathlib import Path


class ResearchJobTests(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from task_store import TaskStore
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        csv = self.path / 'iris.csv'
        load_iris(as_frame=True).frame.to_csv(csv, index=False)
        self.store = TaskStore(self.path / 'research.sqlite3')
        self.task = self.store.create(csv, 'target', max_experiments=2)

    def test_job_claim_blocks_manual_actions_and_duplicate_start(self):
        from research_jobs import ResearchJobs
        jobs = ResearchJobs(self.store)
        jobs.start(self.task, max_calls=1, max_seconds=30)
        with self.assertRaises(RuntimeError):
            jobs.start(self.task, max_calls=1, max_seconds=30)
        with self.assertRaises(RuntimeError):
            self.store.run(self.task)
        with self.assertRaises(RuntimeError):
            self.store.finalize(self.task)

    def test_real_experiments_and_report_are_persisted(self):
        from research_jobs import ResearchJobs
        def candidate(messages, timeout):
            return {'message': {'content': '{"model":"random_forest","parameters":{"max_depth":3}}'}}
        jobs = ResearchJobs(self.store, proposer_factory=lambda: candidate)
        jobs.start(self.task, max_calls=1, max_seconds=30)
        jobs.execute(self.task)
        task = self.store.show(self.task)
        self.assertEqual(task['research_job']['state'], 'completed')
        self.assertEqual(len(task['experiments']), 2)
        self.assertEqual(task['research_report']['cloud_calls'], 0)
        self.assertIsNone(task['final_result'])
        self.assertTrue(task['events'])

    def test_api_starts_local_research_and_serves_workbench(self):
        from fastapi.testclient import TestClient
        from api import create_app
        from research_jobs import ResearchJobs
        from unittest.mock import patch
        def candidate(messages, timeout):
            return {'message': {'content': '{"model":"random_forest","parameters":{"max_depth":3}}'}}
        with patch('research_jobs.LocalQwenClient', return_value=candidate):
            with TestClient(create_app(self.path)) as client:
                self.assertEqual(client.get('/').status_code, 200)
                self.assertEqual(client.get('/static/app.js').status_code, 200)
                response = client.post(f'/tasks/{self.task}/research', json={'max_calls': 1, 'max_seconds': 30})
                self.assertEqual(response.status_code, 202, response.text)
                task = client.get(f'/tasks/{self.task}').json()
                self.assertEqual(len(task['experiments']), 2)
                self.assertIsNone(task['final_result'])
