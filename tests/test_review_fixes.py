import json, ssl, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from ml_engine import configured_clients, TierBudget, _ClientSwitch, MLResearch
from model_budget import ModelLimits, ModelBudgetExceeded
from comparison_benchmark import aggregate
from research_context import is_followup
from tests.test_ml_pipeline import clients
from executor import ResearchSession
from sklearn.datasets import load_iris
from strategy_store import StrategyStore


class ReviewFixes(unittest.TestCase):
    def test_one_key_configures_without_jev(self):
        c=configured_clients({'DEEPSEEK_API_KEY':'fixture-not-a-real-key','LIGHT_MODEL_NAME':'same-model'})
        self.assertEqual(c['classifier'].source,'rules')
        self.assertEqual({c[k].model for k in ('source','low','high','reviewer')},{'same-model'})
        result=json.loads(c['classifier']([{'role':'user','content':json.dumps({'task':{'objective':'Diagnose a failed experiment'}})}]))
        self.assertEqual(result['category'],'failure_diagnosis')

    def test_explicit_single_model_uses_one_shared_client(self):
        c=configured_clients({'DEEPSEEK_API_KEY':'fixture-key','ML_SINGLE_MODEL':'1'})
        self.assertTrue(all(c[r] is c['high'] for r in ('source','planner','low','reviewer')))
        self.assertEqual(c['classifier'].source,'rules')

    def test_api_config_and_completed_report_disclose_policy(self):
        from ml_api import create_app
        from fastapi.testclient import TestClient
        from tools.asgi_test_runtime import asgi_test_runtime
        from ml_engine import RuleClassifier
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients();c['classifier']=RuleClassifier()
            with TestClient(create_app(tmp,client_factory=lambda:c)) as api:
                self.assertEqual(api.get('/ml/config').json()['classifier_source'],'rules')
                run=api.post('/ml/runs',json={'goal':'Compare iris classifiers','csv_text':load_iris(as_frame=True).frame.to_csv(index=False),'target':'target'}).json()['run_id']
                result=api.get('/ml/runs/'+run).json()
                self.assertEqual(result['status'],'completed',result)
                self.assertEqual(result['report']['strategy']['strategy_id'],'strategy_001')
                self.assertFalse(result['report']['auto_evolution_enabled'])

    def test_api_detects_changed_source_even_without_version_bump(self):
        from ml_api import create_app
        from fastapi.testclient import TestClient
        from tools.asgi_test_runtime import asgi_test_runtime
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime(),TestClient(create_app(tmp,client_factory=clients)) as api:
            self.assertFalse(api.get('/ml/version').json()['restart_required'])
            with patch('ml_api.source_fingerprint',return_value='changed'):
                self.assertTrue(api.get('/ml/version').json()['restart_required'])

    def test_tls_retries_once_and_keeps_unknown_usage_reservation(self):
        class Flaky:
            last_usage={};max_output_tokens=20
            def __init__(self): self.calls=0
            def __call__(self,messages,timeout):
                self.calls+=1
                if self.calls==1: raise ssl.SSLEOFError('unexpected EOF')
                self.last_usage={'prompt_tokens':5,'completion_tokens':2};return '{}'
        c=Flaky();g=TierBudget(_ClientSwitch({'reviewer':c}),ModelLimits(2,5000,30))
        self.assertEqual(g.call('reviewer',[{'role':'user','content':'json'}]),'{}')
        self.assertEqual(g.snapshot()['calls'],2)
        self.assertEqual(g.snapshot()['unreconciled_calls'],1)
        self.assertGreater(g.snapshot()['reserved_unknown_tokens'],0)

    def test_tls_retry_never_exceeds_call_budget(self):
        class Offline:
            last_usage={};max_output_tokens=20
            def __call__(self,messages,timeout): raise ssl.SSLEOFError('EOF')
        g=TierBudget(_ClientSwitch({'reviewer':Offline()}),ModelLimits(1,5000,30))
        with self.assertRaises(ModelBudgetExceeded):g.call('reviewer',[{'role':'user','content':'json'}])
        self.assertEqual(len(g.records),1)

    def test_permanent_errors_are_not_retried(self):
        for exc in (ValueError('bad JSON'), RuntimeError('HTTP 401')):
            class Broken:
                last_usage={};max_output_tokens=20
                def __call__(self,messages,timeout):raise exc
            g=TierBudget(_ClientSwitch({'reviewer':Broken()}),ModelLimits(3,5000,30))
            with self.assertRaises(type(exc)):g.call('reviewer',[{'role':'user','content':'json'}])
            self.assertEqual(len(g.records),1)

    def test_second_tls_failure_stops_and_failure_kind_is_network(self):
        from failure_recovery import failure_kind
        class Broken:
            last_usage={};max_output_tokens=20
            def __call__(self,messages,timeout):raise ssl.SSLEOFError('EOF')
        g=TierBudget(_ClientSwitch({'reviewer':Broken()}),ModelLimits(5,5000,30))
        with self.assertRaises(ssl.SSLEOFError) as caught:g.call('reviewer',[{'role':'user','content':'json'}])
        self.assertEqual(len(g.records),2)
        self.assertEqual(failure_kind(caught.exception),'network_failure')

    def test_rule_classifier_is_honest_and_does_not_spend_model_budget(self):
        from ml_engine import RuleClassifier
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/'data.csv';load_iris(as_frame=True).frame.to_csv(data,index=False)
            c=clients();c['classifier']=RuleClassifier()
            report=MLResearch(ResearchSession(data,'target'),c).run('Compare classifiers')
            self.assertEqual(report['research_status'],'approved')
            self.assertEqual(report['usage']['calls'],2)
            self.assertFalse(any(r['role']=='classifier' for r in report['usage']['records']))
            self.assertIn('rules',[e.get('classifier_source') for e in report['events']])

    def test_explanatory_question_defaults_to_followup(self):
        self.assertTrue(is_followup('这些结论有什么局限？'))
        self.assertFalse(is_followup('换一个数据集重新训练'))

    def test_paired_percentages_and_unknown_cost(self):
        def arm(mode,tokens,seconds,cost):
            return {'seed':42,'mode':mode,'metrics':{'status':'completed','research_status':'approved','baseline_only':False,'analysis_scope':'classification',
                'validation_macro_f1':.8,'known_tokens':tokens,'wall_seconds':seconds,'usage_complete':True,'costs':{'calculated_total_usd':cost}}}
        result=aggregate([arm('routed',80,5,.02),arm('high_only',100,10,.04)])
        pair=result['paired_deltas_routed_minus_high_only'][0]
        self.assertEqual(pair['token_savings_percent'],20)
        self.assertEqual(pair['cost_savings_percent'],50)
        self.assertEqual(pair['speedup'],2)
        self.assertFalse(result['relative_summary']['conclusive'])
        pair=aggregate([arm('routed',80,5,None),arm('high_only',100,10,.04)])['paired_deltas_routed_minus_high_only'][0]
        self.assertIsNone(pair['cost_savings_percent'])

    def test_pinned_strategy_reaches_actual_scientist_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/'data.csv';load_iris(as_frame=True).frame.to_csv(data,index=False)
            store=StrategyStore(Path(tmp)/'strategies.json')
            store.bootstrap({'planning_prompt':'POLICY_MARKER: prefer bounded diverse experiments'})
            c=clients();original=c['low']
            captured=[]
            def low(messages,timeout):
                captured.extend(messages);return original(messages,timeout)
            engine=MLResearch(ResearchSession(data,'target'),{**c,'low':low},strategy_store=store)
            report=engine.run('Compare classifiers')
            self.assertIn('POLICY_MARKER',json.dumps(captured))
            self.assertEqual(report['strategy']['strategy_id'],'strategy_001')
            self.assertTrue(any(e['event']=='strategy_pinned' for e in report['events']))

if __name__=='__main__':unittest.main()
