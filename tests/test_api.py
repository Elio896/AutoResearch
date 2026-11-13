import tempfile
import unittest
from pathlib import Path


class APITests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from api import create_app
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.client = TestClient(create_app(Path(self.temp.name)))
        self.addCleanup(self.client.close)

    def create_task(self):
        from sklearn.datasets import load_iris
        response = self.client.post('/tasks', json={
            'csv_text': load_iris(as_frame=True).frame.to_csv(index=False),
            'target': 'target', 'max_experiments': 2})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()['task_id']

    def test_create_train_inspect_finalize_and_block_repeated_test(self):
        task = self.create_task()
        run = self.client.post(f'/tasks/{task}/experiments', json={
            'model': 'random_forest', 'parameters': {'n_estimators': 10}})
        self.assertEqual(run.status_code, 200, run.text)
        self.assertGreater(run.json()['validation_f1'], 0.7)
        self.assertNotIn('test_f1', run.text)
        inspect = self.client.get(f'/tasks/{task}').json()
        self.assertEqual(len(inspect['experiments']), 1)
        self.assertEqual(self.client.post(f'/tasks/{task}/finalize').status_code, 200)
        self.assertEqual(self.client.post(f'/tasks/{task}/finalize').status_code, 409)
        self.assertEqual(self.client.post(f'/tasks/{task}/experiments', json={}).status_code, 409)

    def test_invalid_schema_and_unknown_task(self):
        self.assertEqual(self.client.get('/tasks/missing').status_code, 404)
        self.assertEqual(self.client.post('/tasks', json={
            'csv_text': 'x,y\n1,a\n2,b\n', 'target': 'missing'}).status_code, 400)
        task = self.create_task()
        self.assertEqual(self.client.post(f'/tasks/{task}/experiments', json={
            'model': 'logistic_regression', 'parameters': {'max_depth': 3}}).status_code, 400)
        self.assertEqual(self.client.post('/tasks', json={'csv_path': 'C:\\private.csv'}).status_code, 422)


if __name__ == '__main__':
    unittest.main()
