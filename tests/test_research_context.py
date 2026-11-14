import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from tools.asgi_test_runtime import asgi_test_runtime
from tests.test_ml_pipeline import clients, Reply
from ml_api import create_app
from local_catalog import LocalCatalog
from research_context import present_answer, scope_gap, target_from_question


class ConversationTests(unittest.TestCase):
    def first(self, client):
        result=client.post('/ml/runs',json={'goal':'Compare iris classifiers and provide predictions.',
            'csv_text':load_iris(as_frame=True).frame.to_csv(index=False),'target':'target'})
        self.assertEqual(result.status_code,202,result.text)
        run=result.json()['run_id']
        self.assertEqual(client.get('/ml/runs/'+run).json()['status'],'completed')
        return run

    def test_followup_is_one_review_no_search_or_training_and_real_csv_link(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients()
            with TestClient(create_app(tmp,client_factory=lambda:c)) as client:
                parent=self.first(client)
                counts={k:v.calls for k,v in c.items()}
                c['reviewer'].replies.append({'decision':'approve','answer':'本次分数没有校准，只适合排序。',
                    'summary':'Existing evidence only','experiment_ids':['exp_001']})
                run=client.post('/ml/runs',json={'goal':'你刚才的预测百分比经过概率校准了吗？','previous_run_id':parent}).json()['run_id']
                state=client.get('/ml/runs/'+run).json();report=state['report']
                self.assertEqual(state['status'],'completed',state)
                self.assertEqual(report['training_fits'],0)
                self.assertEqual(report['execution_kind'],'answer_followup')
                self.assertFalse(report['test_evaluated'])
                self.assertEqual(report['usage']['calls'],1)
                for role in ('source','classifier','low','high'):self.assertEqual(c[role].calls,counts[role])
                self.assertIn('没有经过概率校准',report['review']['answer'])
                self.assertEqual(report['context']['evidence_run_id'],parent)
                link=report['research_outputs']['prediction_download_url']
                self.assertEqual(client.get(link).status_code,200)
                self.assertFalse((Path(tmp)/run/'predictions.csv').exists())

    def test_chained_format_followups_keep_evidence_and_all_headings(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients()
            with TestClient(create_app(tmp,client_factory=lambda:c)) as client:
                original=self.first(client);parent=original
                for goal in ('请用三段话回答：发现了什么、证据是什么、有哪些局限。',
                             '请按直接回答、证据、预测结果、局限四部分输出。'):
                    c['reviewer'].replies.append({'decision':'approve','answer':'验证记录已有实际预测。','summary':'existing','experiment_ids':['exp_001']})
                    parent=client.post('/ml/runs',json={'goal':goal,'previous_run_id':parent}).json()['run_id']
                    report=client.get('/ml/runs/'+parent).json()['report']
                    self.assertEqual(report['context']['evidence_run_id'],original)
                    for heading in report['review']['answer_contract']['headings']:self.assertIn(heading+'：',report['review']['answer'])
                    self.assertEqual(report['training_fits'],0)

    def test_explicit_new_experiment_reuses_csv_without_collection(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients()
            with TestClient(create_app(tmp,client_factory=lambda:c)) as client:
                parent=self.first(client)
                c['classifier'].replies.append({'category':'model_search','confidence':.99})
                c['low'].replies.append({'model':'random_forest','parameters':{'n_estimators':12,'max_depth':2},
                    'hypothesis':'Try another tree size','expected_direction':'up','falsifier':'score fails to improve'})
                c['reviewer'].replies.append({'decision':'approve','answer':'已复用数据并验证候选。','summary':'checked','experiment_ids':['exp_001']})
                before=c['source'].calls
                run=client.post('/ml/runs',json={'goal':'请沿用上一轮数据继续尝试改进预测。','previous_run_id':parent}).json()['run_id']
                state=client.get('/ml/runs/'+run).json()
                self.assertIn(state['status'],('completed','needs_revision'),state)
                self.assertGreater(state['report']['training_fits'],0)
                self.assertEqual(c['source'].calls,before)
                self.assertTrue(any(e['event']=='context_loaded' for e in state['events']))

    def test_missing_context_is_clarification_with_available_catalog_no_paid_calls(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients()
            with TestClient(create_app(tmp,client_factory=lambda:c)) as client:
                run=client.post('/ml/runs',json={'goal':'你刚才的预测校准了吗？'}).json()['run_id']
                state=client.get('/ml/runs/'+run).json()
                self.assertEqual(state['status'],'needs_input')
                self.assertIn('当前可用数据集',state['reason'])
                self.assertIn('上一轮',state['reason'])
                self.assertEqual(sum(x.calls for x in c.values()),0)

    def test_no_silent_global_latest_and_invalid_parent_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients()
            with TestClient(create_app(tmp,client_factory=lambda:c)) as client:
                self.first(client)
                self.assertEqual(client.post('/ml/runs',json={'goal':'解释刚才结果','previous_run_id':'../secret'}).status_code,422)
                self.assertEqual(client.post('/ml/runs',json={'goal':'解释刚才结果','previous_run_id':'f'*32}).status_code,404)
                run=client.post('/ml/runs',json={'goal':'解释刚才结果'}).json()['run_id']
                self.assertEqual(client.get('/ml/runs/'+run).json()['status'],'needs_input')

    def test_revision_cannot_be_reported_as_completed(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients(review='revise')
            with TestClient(create_app(tmp,client_factory=lambda:c)) as client:
                run=client.post('/ml/runs',json={'goal':'Compare classifiers','csv_text':load_iris(as_frame=True).frame.to_csv(index=False),'target':'target'}).json()['run_id']
                self.assertEqual(client.get('/ml/runs/'+run).json()['status'],'needs_revision')

    def test_wrong_domain_never_trains_or_attaches_predictions(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients()
            with TestClient(create_app(tmp,client_factory=lambda:c)) as client:
                run=client.post('/ml/runs',json={'goal':'请用当前客户流失数据回答青年失业率是多少。'}).json()['run_id']
                state=client.get('/ml/runs/'+run).json()
                self.assertEqual(state['status'],'needs_input',state)
                self.assertIn('不能回答',state['report']['review']['answer'])
                self.assertIsNone(state['report']['research_outputs'])
                self.assertEqual(sum(x.calls for x in c.values()),0)

    def test_changed_target_uses_existing_field_instead_of_claiming_no_data(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients();parent='c'*32;p=Path(tmp)/parent;p.mkdir()
            csv,target,source=LocalCatalog().download('customer_churn')
            (p/'data.csv').write_text(csv,encoding='utf-8')
            (p/'state.json').write_text(json.dumps({'status':'completed'}))
            (p/'report.json').write_text(json.dumps({'source':source,'protocol':{'target':target},'experiments':[]}),encoding='utf-8')
            with TestClient(create_app(tmp,client_factory=lambda:c)) as client:
                result=client.post('/ml/runs',json={'goal':'请预测数据中的另一个目标：是否开通国际套餐。','previous_run_id':parent})
                state=client.get('/ml/runs/'+result.json()['run_id']).json()
                self.assertEqual(state['status'],'completed',state)
                self.assertEqual(state['report']['protocol']['target'],'international_plan')
                self.assertGreater(state['report']['research_outputs']['records_predicted'],0)
                self.assertEqual(c['source'].calls,0)


class ContractTests(unittest.TestCase):
    def test_domain_mismatch_and_missing_income_are_explicit(self):
        columns=['class','international_plan','number_customer_service_calls','total_day_charge']
        source={'dataset_id':'customer_churn'}
        self.assertIn('不能回答',scope_gap('用客户流失数据回答青年失业率',columns,source))
        self.assertIn('没有收入字段',scope_gap('分析收入对流失的影响',columns,source))
        self.assertIsNone(scope_gap('分析国际套餐与流失的关联',columns,source))

    def test_business_field_target_changes_only_on_explicit_request(self):
        columns=['class','international_plan']
        self.assertEqual(target_from_question('预测数据中的另一个目标：是否开通国际套餐',columns),'international_plan')
        self.assertIsNone(target_from_question('国际套餐是否影响流失',columns))

    def test_catalog_search_understands_existing_business_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog=LocalCatalog(tmp)
            catalog.add('customer_churn','international_plan,class\nyes,1\nno,0\n','class',['customer churn'],{'name':'churn'})
            self.assertEqual(catalog.search('是否开通国际套餐')[0]['id'],'customer_churn')
            self.assertEqual(catalog.search('international plan')[0]['id'],'customer_churn')
            self.assertEqual(catalog.search('收入'),[])

    def test_operations_language_character_budget_and_real_download(self):
        review={'answer':'随机森林的宏 F1 为 0.9。total_day_minutes 较高的客户值得关注。'}
        out={'records_predicted':1000}
        text=present_answer(review,'面向没有机器学习背景的运营人员，不超过500字，提供详细预测下载',out,'/ml/runs/'+'a'*32+'/predictions')['answer']
        self.assertLessEqual(len(text),500)
        self.assertNotIn('F1',text);self.assertNotIn('total_day_minutes',text)
        self.assertIn('日间通话分钟数',text);self.assertIn('/predictions',text)

    def test_absent_prediction_ci_is_not_confused_with_group_wilson(self):
        text=present_answer({'answer':'未计算。'},'请提供预测结果的置信区间',{'records_predicted':0})['answer']
        self.assertIn('没有计算',text)

    def test_model_supplied_headings_are_not_duplicated(self):
        body='直接回答：预测 867/133。\n\n证据：3000 行训练。\n\n预测结果：1000 条。\n\n局限：未校准。'
        review={'answer':body,'decision':'approve','summary':'s','experiment_ids':['exp_001']}
        text=present_answer(review,'请按四部分重写：直接回答、证据、预测结果、局限。',
                            {'records_predicted':1000,'limitations':['未校准']},'/ml/runs/'+'a'*32+'/predictions')['answer']
        for heading in ('直接回答','证据','预测结果','局限'):
            self.assertEqual(text.count(heading),1,heading)
        self.assertIn('867/133',text)


if __name__=='__main__':unittest.main()
