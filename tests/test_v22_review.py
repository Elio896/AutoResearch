"""Regressions found while reviewing the deployed V22 release."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from tools.asgi_test_runtime import asgi_test_runtime
from tests.test_ml_pipeline import clients,Reply
from ml_api import create_app,source_fingerprint,asset_fingerprint
from app_version import APP_VERSION
from research_context import is_followup,requires_experiment
from research_context import present_answer,ANSWER_RULES


class V22ReviewTests(unittest.TestCase):
    def post_csv(self,api):
        return api.post('/ml/runs',json={'goal':'Predict iris classes','csv_text':load_iris(as_frame=True).frame.to_csv(index=False),'target':'target'}).json()['run_id']

    def test_launcher_checks_shared_version_and_restart_flag(self):
        from tools import open_showcase
        for version,restart,expected in [(APP_VERSION,False,True),(APP_VERSION,True,False),('ml-chain-19',False,False)]:
            with self.subTest(version=version,restart=restart),patch.object(open_showcase.opener,'open',return_value=io.BytesIO(json.dumps({'version':version,'restart_required':restart}).encode())):
                self.assertEqual(open_showcase.ready(),expected)

    def test_active_data_parser_and_catalog_changes_affect_fingerprint(self):
        for file in ('data_source.py','local_catalog.py','association_analysis.py','research_deliverables.py'):
            with self.subTest(file=file):
                with patch.object(Path,'read_bytes',autospec=True,side_effect=lambda p:b'original'):before=source_fingerprint()
                with patch.object(Path,'read_bytes',autospec=True,side_effect=lambda p:b'changed' if p.name==file else b'original'):after=source_fingerprint()
                self.assertNotEqual(before,after)

    def test_unsupported_first_question_is_not_misdiagnosed_as_followup(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients();c['source']=Reply([{'query':'unemployment'}],'fixture-source')
            with TestClient(create_app(tmp,client_factory=lambda:c)) as api:
                run=api.post('/ml/runs',json={'goal':'请收集公开青年失业数据并分析趋势。'}).json()['run_id']
                state=api.get('/ml/runs/'+run).json()
                self.assertEqual(state['status'],'needs_input',state)
                self.assertNotIn('追问',state['reason'])
                self.assertIn('当前可用数据集',state['reason'])
                self.assertEqual(c['source'].calls,1)

    def test_uploaded_dataset_business_name_does_not_trigger_new_research(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients()
            with TestClient(create_app(tmp,client_factory=lambda:c)) as api:
                parent=self.post_csv(api)
                c['reviewer'].replies.append({'decision':'approve','answer':'这些是验证记录的分类结果。','summary':'existing','experiment_ids':['exp_001']})
                before=c['source'].calls
                run=api.post('/ml/runs',json={'goal':'请解释鸢尾花的预测结果。','previous_run_id':parent}).json()['run_id']
                state=api.get('/ml/runs/'+run).json()
                self.assertEqual(state['report']['execution_kind'],'answer_followup',state)
                self.assertEqual(state['report']['training_fits'],0)
                self.assertEqual(c['source'].calls,before)

    def seed(self,root,outputs,status):
        run='a'*32;p=Path(root)/run;p.mkdir()
        load_iris(as_frame=True).frame.to_csv(p/'data.csv',index=False)
        report={'research_status':status,'source':{'provider':'local_catalog','dataset_id':'iris'},
                'protocol':{'target':'target'},'experiments':[{'experiment_id':'exp_001'}],
                'research_outputs':outputs,'review':{'answer':'先前研究结果'}}
        (p/'report.json').write_text(json.dumps(report),encoding='utf-8')
        (p/'state.json').write_text(json.dumps({'status':'partial_completed' if outputs else 'needs_input'}))
        return run

    def test_association_followup_keeps_partial_scope(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            parent=self.seed(tmp,{'scope':'training_association_only','records_predicted':0,'predictions':[],'prediction_counts':{},'labels':[],
                                  'association_evidence':{'observations':90,'features':[]}},'partial_completed')
            c=clients()
            with TestClient(create_app(tmp,client_factory=lambda:c)) as api:
                run=api.post('/ml/runs',json={'goal':'这些结果有哪些局限？','previous_run_id':parent}).json()['run_id']
                state=api.get('/ml/runs/'+run).json()
                self.assertEqual(state['status'],'partial_completed',state)
                self.assertEqual(state['report']['research_status'],'partial_completed')
                self.assertEqual(state['report']['training_fits'],0)
                self.assertNotIn('/predictions',state['report']['review']['answer'])

    def test_explaining_missing_data_never_promotes_research_to_approved(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            parent=self.seed(tmp,None,'needs_clarification');c=clients()
            with TestClient(create_app(tmp,client_factory=lambda:c)) as api:
                run=api.post('/ml/runs',json={'goal':'那需要我补充什么？','previous_run_id':parent}).json()['run_id']
                state=api.get('/ml/runs/'+run).json()
                self.assertEqual(state['status'],'needs_input',state)
                self.assertIsNone(state['report']['research_outputs'])

    def test_terminal_state_is_published_only_after_billing_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients();seen=[]
            def finish(_):
                paths=list(Path(tmp).glob('*/state.json'));self.assertEqual(len(paths),1)
                seen.append(json.loads(paths[0].read_text())['status'])
                return {'task_actual_charge':None,'observations':[],'test_snapshot':True}
            with patch('ml_api.BalanceMeter.finish',autospec=True,side_effect=finish),TestClient(create_app(tmp,client_factory=lambda:c)) as api:
                run=self.post_csv(api);state=api.get('/ml/runs/'+run).json()
                self.assertEqual(seen,['running'])
                self.assertEqual(state['status'],'completed',state)
                self.assertTrue(state['report']['account_billing']['test_snapshot'])

    def test_balance_failure_preserves_completed_research_and_unknown_charge(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients()
            with patch('ml_api.BalanceMeter.finish',side_effect=ValueError('fixture failure')),TestClient(create_app(tmp,client_factory=lambda:c)) as api:
                run=self.post_csv(api);state=api.get('/ml/runs/'+run).json()
                self.assertEqual(state['status'],'completed',state)
                bill=state['report']['account_billing']
                self.assertIsNone(bill['task_actual_charge']);self.assertEqual(bill['observations'],[])

    def test_negated_retraining_is_not_an_experiment_request(self):
        for goal in ('解释刚才的预测，不要重新训练。','不重新训练，只解释校准。','无需重训，说明局限。','Explain the result, do not retrain.'):
            with self.subTest(goal=goal):self.assertFalse(requires_experiment(goal))

    def test_independent_positive_request_after_negation_is_preserved(self):
        self.assertTrue(requires_experiment('不要重新训练，但请重新预测这批记录。'))

    def test_negated_data_change_keeps_explicit_context(self):
        self.assertTrue(is_followup('不要换数据，只解释当前结果。'))
        self.assertFalse(is_followup('换数据，重新研究葡萄酒。'))

    def test_calibration_explanation_distinguishes_predicted_label_from_churn_risk(self):
        review=present_answer({'answer':'没有校准。'},'预测校准了吗？',{})
        self.assertIn('不是统一的流失风险分数',review['answer'])
        self.assertIn('predicted_label',ANSWER_RULES)

    def test_static_style_changes_do_not_require_backend_restart(self):
        with patch.object(Path,'read_bytes',autospec=True,side_effect=lambda p:b'original'):
            before=source_fingerprint();assets_before=asset_fingerprint()
        with patch.object(Path,'read_bytes',autospec=True,side_effect=lambda p:b'new style' if p.name=='pixel-theme.css' else b'original'):
            self.assertEqual(before,source_fingerprint())
            self.assertNotEqual(assets_before,asset_fingerprint())


if __name__=='__main__':unittest.main()
