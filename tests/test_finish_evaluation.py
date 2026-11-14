import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from executor import ResearchSession
from ml_api import create_app
from tools.asgi_test_runtime import asgi_test_runtime
from tests.test_ml_pipeline import clients,Reply
from association_analysis import target_group_comparison
from tools.replay_ml_run import replay
from research_context import requires_experiment,needs_new_analysis,target_from_question,research_target


class FinishEvaluationTests(unittest.TestCase):
    def test_plain_improve_command_with_context_really_runs_a_new_experiment(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients()
            with TestClient(create_app(tmp,client_factory=lambda:c)) as api:
                parent=api.post('/ml/runs',json={'goal':'Compare classifiers','csv_text':load_iris(as_frame=True).frame.to_csv(index=False),'target':'target'}).json()['run_id']
                c['classifier'].replies.append({'category':'model_search','confidence':.95})
                c['low'].replies.append({'model':'random_forest','parameters':{'n_estimators':30,'max_depth':4},'hypothesis':'Different boundary','expected_direction':'up','falsifier':'No validation gain'})
                c['reviewer'].replies.append({'decision':'approve','answer':'新实验已实际执行，保留验证最佳方案。','summary':'checked','experiment_ids':['exp_001','exp_002']})
                rid=api.post('/ml/runs',json={'goal':'请在历史最佳方案基础上改进分类预测。','previous_run_id':parent}).json()['run_id']
                state=api.get('/ml/runs/'+rid).json()
                self.assertEqual(state['status'],'completed',state)
                self.assertEqual(state['report']['execution_kind'],'research')
                self.assertEqual(state['report']['training_fits'],2)

    def test_prediction_request_after_association_executes_instead_of_explaining_missing_file(self):
        from local_catalog import LocalCatalog
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            csv,target,_=LocalCatalog().download('customer_churn');c=clients()
            with TestClient(create_app(tmp,client_factory=lambda:c)) as api:
                parent=api.post('/ml/runs',json={'goal':'请比较流失客户与未流失客户的差异。','csv_text':csv,'target':target}).json()['run_id']
                c['classifier'].replies.append({'category':'model_search','confidence':.95})
                c['reviewer'].replies.append({'decision':'approve','answer':'实际预测已生成。','summary':'checked','experiment_ids':['exp_001']})
                run=api.post('/ml/runs',json={'goal':'请给出客户流失预测结果，并说明针对哪些样本。','previous_run_id':parent}).json()['run_id']
                state=api.get('/ml/runs/'+run).json()
                self.assertEqual(state['status'],'completed',state)
                self.assertGreater(state['report']['training_fits'],0)
                self.assertEqual(state['report']['research_outputs']['records_predicted'],1000)
                self.assertEqual(api.get('/ml/runs/'+run+'/predictions').status_code,200)

    def test_predictive_routing_cannot_silently_downgrade_to_association(self):
        from ml_engine import MLResearch
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.csv';load_iris(as_frame=True).frame.to_csv(p,index=False);c=clients(confidence=.2)
            c['high'].replies=[{'decision':'partial','reason':'Missing prior information','questions':[],'association_columns':['sepal length (cm)']},
                               {'decision':'dispatch','category':'model_search','tier':'low','reason':'Current incumbent is supplied','questions':[]}]
            engine=MLResearch(ResearchSession(p,'target'),c)
            result=engine.run('请改进分类预测并给出预测结果。')
            self.assertEqual(result['research_status'],'approved')
            self.assertEqual(result['training_fits'],2)
            self.assertTrue(any(e['event']=='route_recheck_scope_conflict' for e in result['events']))

    def test_uncomputed_significance_is_not_presented_as_a_test_result(self):
        from research_context import present_answer
        review={'decision':'approve','answer':'该组显著高于另一组。'}
        rendered=present_answer(review,'比较两组',{'scope':'training_association_only','association_evidence':{'observations':10,'features':[]},'limitations':[]})
        self.assertNotIn('显著高于',rendered['answer'])
        self.assertIn('未进行显著性检验',rendered['answer'])

    def test_seven_existing_association_fields_are_not_discarded(self):
        from ml_engine import resolve_association_columns
        columns=['feature_'+str(i) for i in range(7)]
        self.assertEqual(resolve_association_columns(columns,columns),columns)
        self.assertEqual(resolve_association_columns(columns+['missing'],columns),columns)

    def test_computed_statistics_can_be_reviewed_without_fitted_experiment_refs(self):
        from local_catalog import LocalCatalog
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            csv,target,_=LocalCatalog().download('customer_churn');c=clients()
            c['reviewer'].replies=[{'decision':'approve','answer':'The computed outcome groups differ; this does not establish causation.',
                                    'summary':'Actual computed group statistics checked.','experiment_ids':[]}]
            with TestClient(create_app(tmp,client_factory=lambda:c)) as api:
                run=api.post('/ml/runs',json={'goal':'\u8bf7\u6bd4\u8f83\u6d41\u5931\u5ba2\u6237\u4e0e\u672a\u6d41\u5931\u5ba2\u6237\u7684\u5dee\u5f02\u3002','csv_text':csv,'target':target}).json()['run_id']
                state=api.get('/ml/runs/'+run).json()
                self.assertEqual(state['status'],'partial_completed',state)
                self.assertTrue(state['report']['research_outputs']['target_group_comparison'])

    def request(self,api,c,namespace,use_memory=True,seed=42):
        c['classifier'].replies.append({'category':'model_search','confidence':.95})
        c['low'].replies.append({'model':'random_forest','parameters':{'n_estimators':20,'max_depth':3},'hypothesis':'tree boundary','expected_direction':'up','falsifier':'no gain'})
        c['reviewer'].replies.append({'decision':'approve','answer':'已完成验证预测。','summary':'checked','experiment_ids':['exp_001']})
        response=api.post('/ml/runs',json={'csv_text':load_iris(as_frame=True).frame.to_csv(index=False),'target':'target','goal':'Compare classifiers',
            'memory_namespace':namespace,'use_memory':use_memory,'seed':seed})
        self.assertEqual(response.status_code,202,response.text)
        state=api.get('/ml/runs/'+response.json()['run_id']).json()
        self.assertEqual(state['status'],'completed',state)
        return state

    def test_memory_ablations_have_isolated_history_and_revalidated_champions(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients();c['classifier'].replies=[];c['low'].replies=[];c['reviewer'].replies=[]
            with TestClient(create_app(tmp,client_factory=lambda:c)) as api:
                first=self.request(api,c,'a'*32,False)
                warm=self.request(api,c,'a'*32,True)
                independent=self.request(api,c,'b'*32,True)
                ignored=self.request(api,c,'a'*32,False)
                self.assertFalse(first['report']['experience']['reused'])
                self.assertTrue(warm['report']['experience']['reused'])
                self.assertGreaterEqual(warm['report']['training_fits'],1)
                self.assertFalse(independent['report']['experience']['reused'])
                self.assertFalse(ignored['report']['experience']['reused'])
                self.assertFalse(any(e['event']=='lessons_consulted' for e in ignored['events']))
                mirrored=json.loads((Path(tmp)/'memory-experiments'/('a'*32)/first['run_id']/'report.json').read_text(encoding='utf-8'))
                self.assertNotIn('account_billing',mirrored);self.assertNotIn('research_outputs',mirrored)

    def test_namespace_cannot_escape_history_root(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime(),TestClient(create_app(tmp,client_factory=clients)) as api:
            self.assertEqual(api.post('/ml/runs',json={'goal':'Compare classifiers','memory_namespace':'../other'}).status_code,422)

    def test_group_comparison_is_actual_training_group_means(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.csv';frame=load_iris(as_frame=True).frame;frame.to_csv(p,index=False)
            session=ResearchSession(p,'target');result=target_group_comparison(session)
            self.assertEqual(result['observations'],len(session.train_indices))
            self.assertFalse(result['causal_identification_performed'])
            for feature in result['comparison']['numeric']:
                self.assertEqual(sum(g['n'] for g in feature['groups']),len(session.train_indices))
                for group in feature['groups']:
                    training=frame.iloc[session.train_indices]
                    actual=training.loc[training.target.astype(str)==group['label'],feature['feature']].mean()
                    self.assertAlmostEqual(actual,group['mean'])

    def test_churn_difference_question_executes_comparison_without_predictions(self):
        from local_catalog import LocalCatalog
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            csv,target,_=LocalCatalog().download('customer_churn');c=clients()
            with TestClient(create_app(tmp,client_factory=lambda:c)) as api:
                run=api.post('/ml/runs',json={'goal':'请比较流失客户与未流失客户的主要差异。','csv_text':csv,'target':target,'max_rounds':3}).json()['run_id']
                state=api.get('/ml/runs/'+run).json();out=state['report']['research_outputs']
                self.assertEqual(state['status'],'partial_completed',state)
                self.assertTrue(out['target_group_comparison']['numeric'])
                self.assertEqual(out['records_predicted'],0)
                self.assertEqual(c['low'].calls,0)

    def test_replay_recomputes_saved_validation_scores_without_test_access(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients()
            with TestClient(create_app(tmp,client_factory=lambda:c)) as api:
                run=api.post('/ml/runs',json={'goal':'Compare classifiers','csv_text':load_iris(as_frame=True).frame.to_csv(index=False),'target':'target'}).json()['run_id']
            result=replay(Path(tmp)/run)
            self.assertTrue(result['all_passed']);self.assertFalse(result['test_evaluated']);self.assertEqual(result['llm_calls'],0)

    def test_new_predict_request_and_new_feature_analysis_are_not_just_reformatting(self):
        for command in ('请在历史最佳方案基础上改进客户流失预测。','帮我优化这个预测模型。','给我预测哪些客户会流失。','能不能改进预测方案。'):
            self.assertTrue(requires_experiment(command),command)
        self.assertFalse(requires_experiment('请说明是否取得改进。'))
        self.assertFalse(requires_experiment('请在原方案基础上不要改进。'))
        self.assertTrue(requires_experiment('请预测哪些客户会流失。'))
        self.assertTrue(requires_experiment('请用本地数据预测客户流失。'))
        old={'association_evidence':{'features':[{'feature':'number_customer_service_calls'}]}}
        self.assertTrue(needs_new_analysis('国际套餐与流失的关联如何？',old,['international_plan','number_customer_service_calls']))
        self.assertFalse(needs_new_analysis('这些结果有哪些局限？',old,['international_plan']))
        self.assertEqual(research_target('国际套餐会导致客户流失吗？',['class','international_plan'],'international_plan'),'class')


if __name__=='__main__':unittest.main()
