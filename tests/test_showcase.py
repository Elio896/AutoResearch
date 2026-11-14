import tempfile
import unittest

from fastapi.testclient import TestClient
from api import create_legacy_app as create_app
from tools.asgi_test_runtime import asgi_test_runtime


class ShowcaseTests(unittest.TestCase):
    def test_real_feedback_next_session_revoke_and_strategy_restore(self):
        with tempfile.TemporaryDirectory() as directory, asgi_test_runtime():
            with TestClient(create_app(directory)) as client:
                self.assertEqual(client.get('/showcase').status_code,200)
                self.assertEqual(client.get('/static/showcase.js').status_code,200)
                initial=client.post('/showcase/runs',json={})
                self.assertEqual(initial.status_code,201,initial.text)
                run=initial.json(); base='/showcase/runs/'+run['run_id']
                self.assertEqual(run['before']['duplicate_fits'],1)
                self.assertEqual(client.post(base+'/next',json={}).status_code,409)
                pending=client.post(base+'/feedback',json={'text':'请做得好一些'}).json()
                self.assertEqual(pending['feedback']['status'],'pending')
                self.assertEqual(pending['phase'],'before')
                active=client.post(base+'/feedback',json={'text':'不要重复运行相同的实验'}).json()
                self.assertEqual(active['feedback']['status'],'active')
                after=client.post(base+'/next',json={}).json()
                self.assertEqual(after['after']['duplicate_fits'],0)
                self.assertEqual(after['after']['blocked'],1)
                self.assertEqual(after['after']['score'],after['before']['score'])
                self.assertEqual(after['after']['fits'],2)
                revoked=client.post(base+'/revoke',json={}).json()
                self.assertEqual(revoked['revoked']['duplicate_fits'],1)
                restored=client.post(base+'/strategy-rollback',json={})
                self.assertEqual(restored.status_code,200,restored.text)
                evidence=restored.json()['strategy_rollback']
                self.assertNotEqual(evidence['from'],evidence['to'])
                self.assertEqual(client.post(base+'/strategy-rollback',json={}).status_code,409)
                self.assertEqual(client.get(base+'/evidence').status_code,200)
                for arm in ('before','after','revoked'):
                    self.assertFalse(revoked[arm]['test_evaluated'])
                    self.assertEqual(revoked[arm]['model_calls'],0)
                self.assertEqual(client.get('/system/feedback').json()['records'],[])

    def test_invalid_run_ids_and_schema_rejected(self):
        with tempfile.TemporaryDirectory() as directory, asgi_test_runtime():
            with TestClient(create_app(directory)) as client:
                self.assertEqual(client.get('/showcase/runs/unknown').status_code,404)
                self.assertEqual(client.post('/showcase/runs/'+'a'*32+'/feedback',json={'text':'x'}).status_code,422)
