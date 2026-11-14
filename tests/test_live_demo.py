"""现场演示端点的测试：离线模式必须能在没有 Ollama 的情况下跑完整条链。"""
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api import create_legacy_app as create_app


def _csv() -> str:
    path = Path('bench/tasks/t2_imbalanced/data.csv')
    if not path.is_file():
        path = Path(__file__).resolve().parent.parent / 'bench/tasks/t2_imbalanced/data.csv'
    return path.read_text(encoding='utf-8')


class LiveDemoEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app())

    def test_offline_run_completes(self):
        response = self.client.post('/research/five-roles', json={
            'csv_text': _csv(),
            'target': 'target',
            'metric': 'f1_minority',
            'pos_label': '1',
            'max_experiments': 3,
            'offline': True,
        })
        self.assertEqual(response.status_code, 201, response.text[:300])
        data = response.json()
        summary = data['summary']
        self.assertEqual(summary['status'], 'completed')
        self.assertIsNotNone(summary['best_score'])
        self.assertTrue(data['offline'])
        # 版本树必须被写出来（演示要能点开看）
        self.assertGreaterEqual(summary.get('evoundo_versions', 0), 1)

    def test_version_tree_endpoint_reads_what_the_run_wrote(self):
        run = self.client.post('/research/five-roles', json={
            'csv_text': _csv(), 'target': 'target', 'metric': 'f1_minority',
            'pos_label': '1', 'max_experiments': 2, 'offline': True,
        }).json()
        run_id = run['summary'].get('evoundo_dir', '').rstrip('\\/').split('\\')[-1].split('/')[-1]
        self.assertTrue(run_id)
        listing = self.client.get('/evoundo/runs')
        self.assertEqual(listing.status_code, 200)
        self.assertGreaterEqual(listing.json()['count'], 1)
        detail = self.client.get(f'/evoundo/{run_id}')
        self.assertEqual(detail.status_code, 200)
        self.assertIn('versions', detail.json())

    def test_experience_endpoint_never_500s(self):
        response = self.client.get('/experience')
        self.assertEqual(response.status_code, 200)
        self.assertIn('lineage', response.json())


if __name__ == '__main__':
    unittest.main()
