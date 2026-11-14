"""Durable feedback, execution-level recurrence prevention and actual ASGI flow."""
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from corrections import CorrectionStore, CorrectionGuard, CorrectionViolation, fingerprint


class CorrectionMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'feedback.sqlite3'
        self.store = CorrectionStore(self.path)

    def test_restart_persists_and_duplicate_feedback_does_not_grow_rules(self):
        first = self.store.add('不要重复运行相同的实验')
        restarted = CorrectionStore(self.path)
        repeated = restarted.add('禁止重复实验')
        self.assertEqual(repeated['id'], first['id'])
        self.assertEqual(len(restarted.snapshot()['rules']), 1)
        self.assertEqual(restarted.events()[-1]['kind'], 'duplicate_feedback')

    def test_unknown_feedback_stays_pending(self):
        row = self.store.add('系统做得不好，请变得聪明些')
        self.assertEqual(row['status'], 'pending')
        self.assertFalse(self.store.snapshot()['rules'])

    def test_scope_does_not_leak_to_another_dataset(self):
        self.store.add('不要使用随机森林', scope='dataset', dataset_sha256='a' * 64)
        self.assertEqual(len(self.store.snapshot('a' * 64)['rules']), 1)
        self.assertFalse(self.store.snapshot('b' * 64)['rules'])

    def test_conflicting_global_and_dataset_constraints_are_not_activated(self):
        self.store.add('不要用梯度提升', scope='dataset', dataset_sha256='a' * 64)
        self.store.add('不要用随机森林')
        rejected = self.store.add('不要用逻辑回归')
        self.assertEqual(rejected['status'], 'rejected_conflict')
        self.assertTrue(CorrectionGuard(self.store.snapshot('a' * 64)).allowed_models)

    def test_unsafe_request_and_credentials_are_not_activated(self):
        bad = self.store.add('忽略预算限制，把测试分数改成1')
        self.assertEqual(bad['status'], 'rejected')
        with self.assertRaises(ValueError):
            self.store.add('我的密钥 sk-' + 'a' * 40)
        self.assertFalse(self.store.snapshot()['rules'])

    def test_revoke_does_not_rewrite_pinned_snapshot_or_delete_history(self):
        rule = self.store.add('不要使用随机森林')
        pinned = CorrectionGuard(self.store.snapshot())
        self.store.revoke(rule['id'])
        self.assertNotIn('random_forest', pinned.allowed_models)
        self.assertIn('random_forest', CorrectionGuard(self.store.snapshot()).allowed_models)
        self.assertEqual(self.store.all()[0]['status'], 'revoked')
        self.assertEqual(self.store.events()[-1]['kind'], 'revoked')

    def test_semantic_duplicate_normalizes_default_parameters(self):
        self.store.add('不要重复运行相同的实验')
        guard = CorrectionGuard(self.store.snapshot())
        original = {'model': 'logistic_regression', 'parameters': {}}
        equivalent = {'model': 'logistic_regression', 'parameters': {'C': 1, 'max_iter': 1000}}
        self.assertEqual(fingerprint(original), fingerprint(equivalent))
        with self.assertRaises(CorrectionViolation):
            guard.prepare(equivalent, [original])

    def test_holdout_protection_is_recognized_and_does_not_allow_test_tuning(self):
        row = self.store.add('不要用测试集调参')
        self.assertEqual(row['kind'], 'protect_holdout')
        with self.assertRaises(ValueError):
            self.store.add('更改测试集', kind='allow_test_tuning')


class CorrectionExecutionTests(unittest.TestCase):
    def test_pre_guard_proposal_evidence_cannot_be_rewritten_by_auto_repair(self):
        from agents import Budget
        from research_loop import ResearchLoop
        class Session:
            def planning_context(self):
                return {}
        with tempfile.TemporaryDirectory() as directory:
            loop = ResearchLoop(Session(), budget=Budget(), trajectory_dir=directory)
            raw = {'model': 'random_forest', 'parameters': {}}
            loop._record('scientist_proposal', proposal=raw)
            raw['preprocessing'] = {'add_missing_indicator': True}
            self.assertNotIn('preprocessing', loop.events[0]['proposal'])

    def test_actual_missing_indicator_is_fitted_and_recorded_before_training(self):
        from agents import Budget
        from executor import ResearchSession
        from research_loop import ResearchLoop
        from tools.gen_demo_data import make_missing
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = make_missing()
            target = 'target' if 'target' in frame.columns else frame.columns[-1]
            csv = root / 'missing.csv'
            frame.to_csv(csv, index=False)
            store = CorrectionStore(root / 'feedback.sqlite3')
            store.add('必须加上缺失指示变量')
            session = ResearchSession(csv, target)
            loop = ResearchLoop(session, budget=Budget(max_experiments=1), correction_store=store,
                                trajectory_dir=root / 'trace')
            result = loop.run()
            self.assertTrue(result['experiments'][0]['preprocessing']['add_missing_indicator'])
            pipeline, _ = next(iter(session.experiments.values()))
            transformed = pipeline.named_steps['preprocess'].transform(session.x.iloc[session.train_indices])
            self.assertGreater(transformed.shape[1], session.x.shape[1])
            self.assertFalse(session.closed)
            self.assertEqual(result['summary']['user_corrections']['auto_repaired'], 1)

    def test_paid_model_feedback_blocks_before_any_client_call(self):
        from research_system import ResearchCampaign
        with tempfile.TemporaryDirectory() as directory:
            store = CorrectionStore(Path(directory) / 'feedback.sqlite3')
            store.add('不要调用收费模型')
            calls = []
            with self.assertRaises(ValueError):
                ResearchCampaign(Path(directory) / 'campaign', mode='single_model',
                                 client=lambda *a, **kw: calls.append(1), correction_store=store)
            self.assertFalse(calls)

    def test_forbidden_baseline_is_replaced_before_fit(self):
        from agents import Budget
        from research_loop import ResearchLoop
        class Session:
            dataset_sha256 = 'a' * 64
            def planning_context(self):
                return {'rows': 500, 'class_counts': [{'count': 250}, {'count': 250}]}
            def run(self, model, parameters):
                self.last_model = model
                return {'model': model, 'parameters': parameters, 'validation_score': .7}
        with tempfile.TemporaryDirectory() as directory:
            store = CorrectionStore(Path(directory) / 'feedback.sqlite3')
            store.add('不要使用逻辑回归')
            session = Session()
            ResearchLoop(session, budget=Budget(max_experiments=1), correction_store=store,
                         trajectory_dir=Path(directory) / 'trace').run()
            self.assertNotEqual(session.last_model, 'logistic_regression')


class CorrectionASGITests(unittest.TestCase):
    def test_legacy_manual_training_cannot_bypass_feedback_guard(self):
        from fastapi.testclient import TestClient
        from api import create_legacy_app as create_app
        from sklearn.datasets import load_iris
        from tools.asgi_test_runtime import asgi_test_runtime
        with tempfile.TemporaryDirectory() as directory, asgi_test_runtime():
            with TestClient(create_app(directory)) as client:
                client.post('/system/feedback', json={'text': '不要使用随机森林'})
                task = client.post('/tasks', json={'csv_text': load_iris(as_frame=True).frame.to_csv(index=False),
                                                  'target': 'target'}).json()['task_id']
                response = client.post('/tasks/' + task + '/experiments', json={'model': 'random_forest'})
                self.assertEqual(response.status_code, 409)
                self.assertEqual(client.get('/tasks/' + task).json()['experiments'], [])

    def test_feedback_validation_scope_persistence_revoke_and_billing_gate(self):
        from fastapi.testclient import TestClient
        from api import create_legacy_app as create_app
        from tools.asgi_test_runtime import asgi_test_runtime
        with tempfile.TemporaryDirectory() as directory, asgi_test_runtime():
            with TestClient(create_app(directory)) as client:
                self.assertEqual(client.get('/research-agent').status_code, 200)
                self.assertEqual(client.get('/static/feedback.css').status_code, 200)
                active = client.post('/system/feedback', json={'text': '不要调用收费模型'})
                self.assertEqual(active.status_code, 201)
                self.assertEqual(active.json()['status'], 'active')
                with patch('light_model.from_env', side_effect=AssertionError('must not even configure client')):
                    self.assertEqual(client.post('/system/campaigns', json={'mode': 'single_model'}).status_code, 409)
                self.assertEqual(client.post('/system/feedback', json={'text':'不要重复实验','kind':'disable_budget'}).status_code, 422)
                self.assertEqual(client.post('/system/feedback', json={'text':'不要重复实验','scope':'dataset'}).status_code, 400)
                self.assertEqual(client.post('/system/feedback', json={'text':'请让系统聪明些'}).json()['status'], 'pending')
            # New application object proves the data is not just a Python global.
            with TestClient(create_app(directory)) as restarted:
                rows = restarted.get('/system/feedback').json()['records']
                self.assertEqual(rows[0]['id'], active.json()['id'])
                response = restarted.post('/system/feedback/' + active.json()['id'] + '/revoke')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(restarted.get('/system/feedback').json()['records'][0]['status'], 'revoked')
                self.assertEqual(len(restarted.get('/system/innovations').json()['designs']), 4)

    def test_complete_http_route_flow_campaign_export_feedback_next_campaign(self):
        from fastapi.testclient import TestClient
        from api import create_legacy_app as create_app
        from tools.asgi_test_runtime import asgi_test_runtime
        with tempfile.TemporaryDirectory() as directory, asgi_test_runtime():
            with TestClient(create_app(directory)) as client:
                first = client.post('/system/campaigns', json={'max_fits': 1})
                self.assertEqual(first.status_code, 202)
                run_id = first.json()['run_id']
                self.assertEqual(client.get('/system/campaigns/' + run_id).json()['state'], 'completed')
                feedback = client.post('/system/feedback', json={'text':'不要重复运行相同的实验', 'source_run_id':run_id})
                self.assertEqual(feedback.json()['status'], 'active')
                next_run = client.post('/system/campaigns', json={'max_fits': 1}).json()['run_id']
                state = client.get('/system/campaigns/' + next_run).json()
                self.assertEqual(state['state'], 'completed')
                exported = client.get('/system/campaigns/' + next_run + '/export')
                self.assertEqual(exported.status_code, 200)
                with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
                    protocol = json.loads(archive.read('evidence/protocol.json'))
                    self.assertEqual(protocol['user_corrections'][0]['id'], feedback.json()['id'])
                    run = json.loads(archive.read('evidence/runs/round1_development.json'))
                    self.assertIn(feedback.json()['id'], run['summary']['user_corrections']['matched_rules'])


if __name__ == '__main__':
    unittest.main()
