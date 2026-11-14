"""三步交互流程的测试：分析 → 人审批 → 执行（带两支分支）。"""
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api import create_app


def _csv() -> str:
    rows = ['sepal_length,sepal_width,petal_length,petal_width,species']
    for i in range(25):
        rows.append(f'{5.0 + i * 0.04:.2f},3.4,{1.4 + i * 0.01:.2f},0.2,setosa')
        rows.append(f'{6.0 + i * 0.04:.2f},2.9,{4.5 + i * 0.02:.2f},1.4,versicolor')
        rows.append(f'{6.5 + i * 0.04:.2f},3.0,{5.8 + i * 0.02:.2f},2.1,virginica')
    return '\n'.join(rows)


class ThreeStepDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app())

    def _analyze(self):
        return self.client.post('/demo/analyze', json={
            'csv_text': _csv(), 'target': 'species', 'metric': 'f1_macro',
            'pos_label': '1', 'use_local_model': False,
        })

    def test_analyze_returns_thought_and_suggestion(self):
        response = self._analyze()
        self.assertEqual(response.status_code, 201, response.text[:300])
        data = response.json()
        self.assertTrue(data['session_id'])
        self.assertIn('thought', data)
        self.assertTrue(data['thought']['text'])
        self.assertIn('suggested', data)
        self.assertIn('metric', data['suggested'])

    def test_execute_before_approval_is_blocked(self):
        """人没点同意之前，不许跑实验。"""
        session_id = self._analyze().json()['session_id']
        response = self.client.post('/demo/execute', json={
            'session_id': session_id, 'offline': True})
        self.assertEqual(response.status_code, 409)

    def test_approve_then_execute_returns_two_branches(self):
        session_id = self._analyze().json()['session_id']
        approve = self.client.post('/demo/approve', json={
            'session_id': session_id,
            'overrides': {'max_experiments': 2, 'metric': 'f1_macro'}})
        self.assertEqual(approve.status_code, 200)
        self.assertTrue(approve.json()['approved'])

        exec_response = self.client.post('/demo/execute', json={
            'session_id': session_id, 'offline': True})
        self.assertEqual(exec_response.status_code, 201, exec_response.text[:300])
        data = exec_response.json()
        self.assertIn('branches', data)
        self.assertIn('jev', data['branches'])
        self.assertIn('cloud', data['branches'])
        self.assertIn('verdict', data['branches']['jev'])
        self.assertIn('status', data['branches']['cloud'])
        # 人改过的预算要生效
        self.assertEqual(data['config']['max_experiments'], 2)

    def test_unknown_session_404(self):
        response = self.client.post('/demo/approve', json={
            'session_id': 'nope', 'overrides': {}})
        self.assertEqual(response.status_code, 404)


if __name__ == '__main__':
    unittest.main()
