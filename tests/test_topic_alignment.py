"""Regression tests for the competition workflow; no sockets or paid models."""
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from agents import Budget, Scientist
from model_budget import ModelBudgetExceeded, ModelLimits, SharedModel
from research_loop import DEFAULT_ALLOWED, ResearchLoop
from research_system import ResearchCampaign
from strategy_eval import compare, freeze_criteria
from strategy_store import StrategyStore


def proposal(model='random_forest', **extra):
    return {'model': model, 'parameters': {}, 'hypothesis': 'Test a nonlinear boundary.',
            'expected_direction': 'up', 'falsifier': 'Validation F1 does not improve.', **extra}


class SharedBudgetTests(unittest.TestCase):
    def client(self, usage=True):
        class Client:
            max_output_tokens = 32
            last_usage = {}

            def __call__(self, messages, timeout):
                self.last_usage = {'prompt_tokens': 20, 'completion_tokens': 10} if usage else {}
                return json.dumps(proposal())
        return Client()

    def test_multiple_roles_share_one_call_cap_and_provider_usage(self):
        shared = SharedModel(self.client(), ModelLimits(2, 5000, 60))
        shared.for_role('planning')([], timeout=5)
        shared.for_role('diagnosis')([], timeout=5)
        with self.assertRaises(ModelBudgetExceeded):
            shared.for_role('review')([], timeout=5)
        self.assertEqual(shared.snapshot()['calls'], 2)
        self.assertEqual(shared.snapshot()['known_tokens'], 60)
        self.assertEqual(shared.snapshot()['unreconciled_calls'], 0)
        self.assertIsNone(shared.snapshot()['cost_cny'])

    def test_unknown_usage_is_reserved_not_zero(self):
        shared = SharedModel(self.client(False), ModelLimits(5, 10000, 60))
        shared.for_role('planning')([{'role': 'user', 'content': 'evidence'}])
        usage = shared.snapshot()
        self.assertEqual(usage['unreconciled_calls'], 1)
        self.assertGreater(usage['budget_tokens'], 0)

    def test_failed_attempt_consumes_call_and_unknown_reservation(self):
        def failed(messages, timeout):
            raise RuntimeError('PRIVATE PROVIDER ERROR')
        shared = SharedModel(failed, ModelLimits(1, 5000, 60))
        with self.assertRaises(RuntimeError):
            shared.for_role('diagnosis')([])
        with self.assertRaises(ModelBudgetExceeded):
            shared.for_role('planning')([])
        self.assertNotIn('PRIVATE', json.dumps(shared.snapshot()))
        self.assertEqual(shared.snapshot()['calls'], 1)

    def test_token_admission_blocks_before_transport(self):
        calls = []
        shared = SharedModel(lambda *a, **k: calls.append(1), ModelLimits(2, 1, 60))
        with self.assertRaises(ModelBudgetExceeded):
            shared.for_role('planning')([])
        self.assertFalse(calls)

    def test_deadline_blocks_before_transport(self):
        clock = [0]
        shared = SharedModel(self.client(), ModelLimits(2, 5000, 2), clock=lambda: clock[0])
        clock[0] = 3
        with self.assertRaises(ModelBudgetExceeded):
            shared.for_role('planning')([])

    def test_scientist_records_string_adapter_usage(self):
        shared = SharedModel(self.client())
        scientist = Scientist(shared.for_role('scientist'), DEFAULT_ALLOWED)
        scientist.propose([], {})
        self.assertEqual(scientist.last_tokens, {'prompt': 20, 'completion': 10})


class ProviderAccountingTests(unittest.TestCase):
    def test_failed_provider_attempts_cannot_bypass_call_cap(self):
        from light_model import LightModelClient, LightModelError
        client = LightModelClient('test-key', max_calls=1)
        with patch.object(client, '_post', side_effect=LightModelError('timeout')) as post:
            with self.assertRaises(LightModelError):
                client([], timeout=1)
            with self.assertRaises(LightModelError):
                client([], timeout=1)
            self.assertEqual(post.call_count, 1)
        self.assertEqual(client.usage['calls'], 1)
        self.assertFalse(client.usage_summary()['usage_complete'])
        self.assertIsNone(client.usage_summary()['cost_cny'])

    def test_usage_is_reconciled_and_missing_usage_stays_unknown(self):
        from light_model import LightModelClient
        client = LightModelClient('test-key')
        payload = {'choices': [{'message': {'content': 'ok'}}],
                   'usage': {'prompt_tokens': 11, 'completion_tokens': 5}}
        with patch.object(client, '_post', return_value=payload):
            client([])
        self.assertEqual(client.last_usage, {'prompt_tokens': 11, 'completion_tokens': 5})
        with patch.object(client, '_post', return_value={'choices': payload['choices']}):
            client([])
        self.assertEqual(client.last_usage, {})
        self.assertFalse(client.usage_complete)

    def test_credential_bearing_endpoint_requires_https_or_loopback(self):
        from light_model import LightModelClient, LightModelError
        for url in ('http://example.com', 'https://user:pass@example.com', 'https://example.com?key=private'):
            with self.assertRaises(LightModelError):
                LightModelClient('test-key', base_url=url)


class FakeSession:
    def planning_context(self):
        return {'rows': 600, 'class_count': 2, 'class_counts': [{'count': 300}, {'count': 300}]}

    def run(self, model, parameters):
        return {'model': model, 'parameters': parameters, 'validation_score':
                {'logistic_regression': 0.6, 'random_forest': 0.8, 'hist_gradient_boosting': 1.0}[model]}


class ResearchBindingTests(unittest.TestCase):
    def test_strategy_changes_actual_prompt_and_keeps_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StrategyStore(Path(directory) / 'strategy.json')
            store.bootstrap({'planning_prompt': 'UNIQUE_POLICY_INSTRUCTION'})
            seen = []
            def model(messages, timeout):
                seen.append(messages)
                return json.dumps(proposal())
            loop = ResearchLoop(FakeSession(), proposer=model, strategy_store=store,
                                budget=Budget(max_experiments=2, max_local_calls=1),
                                trajectory_dir=Path(directory) / 'trajectory')
            result = loop.run()
            self.assertIn('UNIQUE_POLICY_INSTRUCTION', seen[0][0]['content'])
            self.assertTrue(result['summary']['strategy_pin_intact'])

    def test_cloud_route_executes_proposal_without_replanning(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = ResearchLoop(FakeSession(), proposer=None,
                                cloud_proposer=lambda *a, **k: json.dumps(proposal()),
                                use_task_router=True, max_cloud_calls=1,
                                budget=Budget(max_experiments=2), trajectory_dir=directory)
            with patch('task_router.route', return_value={'executor': 'deepseek'}):
                result = loop.run()
            self.assertEqual(result['summary']['experiments'], 2)
            self.assertEqual(result['summary']['cloud_calls'], 1)
            self.assertEqual(result['summary']['best_model'], 'random_forest')

    def test_suspicious_result_is_not_final_best_or_rollback_target(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = ResearchLoop(FakeSession(), proposer=lambda *a, **k: json.dumps(proposal('hist_gradient_boosting')),
                                budget=Budget(max_experiments=2, max_local_calls=1), trajectory_dir=directory)
            result = loop.run()
            self.assertEqual(result['summary']['best_model'], 'logistic_regression')
            self.assertEqual(loop.chain.current.score, 0.6)
            self.assertTrue(any(e['event'] == 'rollback' for e in result['events']))

    def test_preprocessing_must_be_bounded_and_boolean(self):
        scientist = Scientist(None, DEFAULT_ALLOWED)
        with self.assertRaises(ValueError):
            scientist.parse({'message': {'content': json.dumps(proposal(preprocessing={'add_missing_indicator': 1}))}})
        parsed = scientist.parse({'message': {'content': json.dumps(proposal(preprocessing={'add_missing_indicator': True}))}})
        self.assertTrue(parsed['preprocessing']['add_missing_indicator'])

    def test_stale_activation_cannot_replace_new_active(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StrategyStore(Path(directory) / 's.json')
            store.bootstrap({'planning_prompt': 'initial'})
            c = store.propose({'planning_prompt': 'candidate'}, reason='test')
            with self.assertRaises(RuntimeError):
                store.activate(c['strategy_id'], checks={'tested': True}, jev={'decision': 'upgrade'}, expected_active_id='stale')
            with self.assertRaises(PermissionError):
                store.activate(c['strategy_id'], checks={'tested': 'yes'}, jev={'decision': 'upgrade'})


def arm(**patches):
    result = {'task_count': 2, 'completed': 2, 'legal_output_rate': 1.0, 'duplicate_rate': 0.0,
              'mean_best_score': 0.7, 'local_calls': 0, 'cloud_calls': 2, 'model_tokens': 100,
              'elapsed_seconds': 1.0, 'task_ids': ['a', 'b']}
    result.update(patches)
    return result


class EvaluationIntegrityTests(unittest.TestCase):
    def test_incomplete_tasks_cannot_win_on_easy_subset(self):
        r = compare(arm(), arm(completed=1, mean_best_score=0.9), freeze_criteria())
        self.assertEqual(r['verdict'], 'keep')
        self.assertIn('completion', r['failed_checks'])

    def test_cloud_call_inflation_is_counted(self):
        r = compare(arm(), arm(cloud_calls=10, mean_best_score=0.8), freeze_criteria())
        self.assertEqual(r['verdict'], 'keep')

    def test_unknown_tokens_is_insufficient(self):
        r = compare(arm(), arm(model_tokens=None, mean_best_score=0.8), freeze_criteria())
        self.assertEqual(r['verdict'], 'insufficient_evidence')

    def test_changed_criteria_or_task_set_is_rejected(self):
        frozen = freeze_criteria()
        frozen['criteria']['mean_best_score']['min_delta'] = -1
        r = compare(arm(), arm(task_ids=['a', 'c'], mean_best_score=0.8), frozen)
        self.assertEqual(r['verdict'], 'keep')
        self.assertIn('criteria_integrity', r['failed_checks'])
        self.assertIn('matched_tasks', r['failed_checks'])

    def test_nan_is_unknown_not_success(self):
        r = compare(arm(), arm(mean_best_score=float('nan')), freeze_criteria())
        self.assertEqual(r['verdict'], 'insufficient_evidence')


class CampaignIntegrationTests(unittest.TestCase):
    def test_real_two_rounds_isolated_data_final_test_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = ResearchCampaign(directory)
            report = campaign.run()
            self.assertEqual(len(report['rounds']), 2)
            self.assertEqual(report['model_usage']['calls'], 0)
            self.assertLessEqual(report['training_fits'], 80)
            self.assertIsNotNone(report['final_test'])
            first, second = report['rounds']
            self.assertEqual(first['decision'], 'upgrade')
            self.assertEqual(second['decision'], 'keep')
            self.assertNotEqual(first['evaluation']['old']['task_ids'], second['evaluation']['old']['task_ids'])
            self.assertEqual(campaign.store.get(second['candidate'])['status'], 'rejected')
            runs = list((Path(directory) / 'runs').glob('*.json'))
            self.assertTrue(all(not json.loads(p.read_text(encoding='utf-8'))['summary']['test_evaluated'] for p in runs))
            # A replay is a new campaign, not overwriting accepted evidence.
            with self.assertRaises(ValueError):
                ResearchCampaign(directory)

    def test_fit_budget_one_stops_without_unmeasured_upgrade(self):
        with tempfile.TemporaryDirectory() as directory:
            report = ResearchCampaign(directory, max_fits=1).run()
            self.assertEqual(report['training_fits'], 1)
            self.assertIsNone(report['final_test'])
            self.assertTrue(all(r['decision'] == 'keep' for r in report['rounds']))

    def test_direct_api_campaign_export_and_manual_restore(self):
        # Direct sync handler contract test: no TestClient/socketpair required.
        from fastapi import BackgroundTasks
        from api import create_legacy_app as create_app
        from research_service import CampaignInput
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(directory)
            route = lambda path, method: next(r.endpoint for r in app.routes if
                                              getattr(r, 'path', None) == path and method in r.methods)
            tasks = BackgroundTasks()
            started = route('/system/campaigns', 'POST')(CampaignInput(), tasks)
            queued = tasks.tasks[0]
            queued.func(*queued.args, **queued.kwargs)
            show = route('/system/campaigns/{run_id}', 'GET')
            report = show(started['run_id'])
            self.assertEqual(report['state'], 'completed', report)
            exported = route('/system/campaigns/{run_id}/export', 'GET')(started['run_id'])
            with zipfile.ZipFile(io.BytesIO(exported.body)) as archive:
                self.assertIn('evidence/report.json', archive.namelist())
                self.assertIn('source/research_system.py', archive.namelist())
                self.assertIn('source/tools/run_research_system.py', archive.namelist())
            restored = route('/system/campaigns/{run_id}/rollback', 'POST')(started['run_id'])
            self.assertTrue(restored['restored'])
            self.assertEqual(show(started['run_id'])['current_strategy']['strategy_id'], 'strategy_001')
            self.assertEqual(show(started['run_id'])['report']['final_test'], report['report']['final_test'])


if __name__ == '__main__':
    unittest.main()
