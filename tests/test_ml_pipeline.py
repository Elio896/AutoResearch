import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from executor import ResearchSession
from ml_engine import MLResearch, JevClassifier, CATEGORIES
from ml_api import create_app
from data_source import OpenMLCatalog
from tools.asgi_test_runtime import asgi_test_runtime


class Reply:
    def __init__(self,replies,model):
        self.replies=list(replies); self.model=model; self.last_usage={}; self.calls=0
    def __call__(self,messages,timeout):
        self.calls+=1; self.last_usage={'prompt_tokens':20,'completion_tokens':10}
        if not self.replies: raise AssertionError('Unexpected extra API call to '+self.model)
        raw=self.replies.pop(0)
        return json.dumps(raw) if isinstance(raw,dict) else raw


def clients(natural=False,low_bad=False,review='approve',category='model_search',confidence=.95):
    plan={'tasks':[{'id':'task1','objective':'Compare a bounded random forest classifier with the baseline.'}]}
    source=Reply(([{'query':'iris'},{'dataset_id':'61','reason':'matches requested classification data'}] if natural else [])+[plan],'fixture-source')
    proposal={'model':'random_forest','parameters':{'n_estimators':20,'max_depth':3},
              'hypothesis':'Tree boundaries may improve classification.','expected_direction':'up',
              'falsifier':'Validation macro F1 does not improve.'}
    return {'source':source,'classifier':Reply([{'category':category,'confidence':confidence}],'fixture-classifier'),
            'low':Reply(['invalid JSON' if low_bad else proposal],'fixture-low'),
            'high':Reply([proposal],'fixture-high'),
            'reviewer':Reply([{'decision':review,'summary':'Evidence checked against the supplied experiments.','answer':'根据提供的数据和真实验证结果，本轮已完成分类预测；结果仅适用于当前验证样本，不能据此证明因果。',
                               'experiment_ids':['exp_001']}],'fixture-reviewer')}


class CatalogFixture:
    def search(self,query):
        assert query=='iris'
        return [{'id':'61','name':'iris-fixture'}]
    def download(self,dataset_id):
        assert dataset_id=='61'
        return load_iris(as_frame=True).frame.to_csv(index=False),'target',{'provider':'test_catalog','dataset_id':'61'}


class MLChainTests(unittest.TestCase):
    def test_pricing_endpoints_schedule_refresh_without_model_calls(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime(),patch('price_feed.PriceFeed.refresh_async') as refresh:
            with TestClient(create_app(tmp)) as client:
                response=client.get('/ml/pricing')
                self.assertEqual(response.status_code,200)
                self.assertEqual(response.json()['refresh_interval_seconds'],300)
                self.assertIn('deepseek',response.json()['prices'])
                response=client.post('/ml/pricing/refresh',json={})
                self.assertEqual(response.status_code,202)
                self.assertEqual(refresh.call_count,2)
                self.assertEqual(refresh.call_args.kwargs,{'force':True})

    def test_price_refresh_parses_official_rates_and_preserves_last_good_on_failure(self):
        from price_feed import PriceFeed,parse_deepseek,parse_jev
        html='<p>01:00 04:00 06:00 10:00 Monday Friday</p><table><tr><th>MODEL</th><th>deepseek-flash</th><th>deepseek-v4-pro</th></tr>'
        for label,off,peak in [('1M INPUT TOKENS CACHE HIT',('0.003','0.022'),('0.006','0.044')),('1M INPUT TOKENS CACHE MISS',('0.15','0.66'),('0.3','1.32')),('1M OUTPUT TOKENS',('0.6','1.98'),('1.2','3.96'))]:
            html+='<tr><td>'+label+'</td><td>OFF-PEAK $'+off[0]+'</td><td>$'+off[1]+'</td></tr><tr><td>PEAK</td><td>$'+peak[0]+'</td><td>$'+peak[1]+'</td></tr>'
        html+='</table>'
        self.assertEqual(parse_deepseek(html)['rates']['deepseek-flash'],[.003,.15,.6])
        self.assertEqual(parse_jev('<p>Jev costs $0.042 per million input tokens. Output tokens are free.</p>')['input_per_million'],.042)
        with self.assertRaises(ValueError):parse_deepseek(html.replace('1M','1K'))
        with tempfile.TemporaryDirectory() as tmp:
            feed=PriceFeed(Path(tmp)/'prices.json')
            def page(url):return (html if 'deepseek' in url else '<p>Jev costs $0.042 per million input tokens. Output tokens are free.</p>','digest')
            with patch('price_feed.read_page',side_effect=page):feed.refresh()
            first=feed.snapshot_for_request()
            self.assertEqual(first['deepseek']['status'],'live')
            with patch('price_feed.read_page',side_effect=OSError('offline')):feed.refresh()
            second=PriceFeed(Path(tmp)/'prices.json').snapshot_for_request()
            self.assertEqual(second['deepseek']['rates'],first['deepseek']['rates'])
            self.assertEqual(second['deepseek']['status'],'stale')

    def test_request_price_snapshot_cannot_change_after_refresh(self):
        from price_feed import PriceFeed
        from ml_engine import TierBudget,_ClientSwitch
        from model_budget import ModelLimits
        from model_costs import request_cost
        class Call:
            model='deepseek-flash';base_url='https://api.deepseek.com/v1';max_output_tokens=20
            last_usage={};last_billing_usage={}
            def __call__(self,messages,timeout):
                self.last_usage={'prompt_tokens':100,'completion_tokens':10}
                self.last_billing_usage={'prompt_cache_hit_tokens':0,'prompt_cache_miss_tokens':100}
                return '{}'
        with tempfile.TemporaryDirectory() as tmp:
            feed=PriceFeed(Path(tmp)/'prices.json')
            gateway=TierBudget(_ClientSwitch({'low':Call()}),ModelLimits(3,5000,30),price_feed=feed)
            gateway.call('low',[{'role':'user','content':'JSON'}])
            record=gateway.records[0]
            before=request_cost(record)
            feed.prices['deepseek']['rates']['deepseek-flash']=[1.,2.,3.]
            self.assertEqual(request_cost(record),before)
            gateway.call('low',[{'role':'user','content':'JSON'}])
            self.assertNotEqual(gateway.records[1]['price_snapshot'],record['price_snapshot'])

    def test_official_costs_use_cache_and_do_not_claim_invoice_charges(self):
        from model_costs import cost_breakdown
        records=[{'provider':'deepseek','model':'deepseek-v4-pro','request_utc':'2026-11-14T12:00:00+00:00','usage':{'prompt_tokens':1000,'completion_tokens':100},'billing_usage':{'prompt_cache_hit_tokens':800,'prompt_cache_miss_tokens':200}},
                 {'provider':'typesafe','model':'jev-latest','usage':{'prompt_tokens':1000,'completion_tokens':50}}]
        costs=cost_breakdown(records)
        expected=(800*.022+200*.66+100*1.98+1000*.042)/1000000
        self.assertAlmostEqual(costs['calculated_total_usd'],expected)
        self.assertIsNone(costs['invoice_actual_usd'])
        self.assertEqual(costs['models'][1]['calculated_usd'],.000042)

    def test_missing_cache_or_usage_cannot_be_invented_as_exact_cost(self):
        from model_costs import cost_breakdown
        record={'provider':'deepseek','model':'deepseek-flash','request_utc':'2026-11-14T12:00:00+00:00','usage':{'prompt_tokens':1000,'completion_tokens':100}}
        costs=cost_breakdown([record])
        self.assertIsNone(costs['calculated_total_usd'])
        self.assertLess(costs['calculated_min_usd'],costs['calculated_max_usd'])
        costs=cost_breakdown([record,{'provider':'deepseek','model':'deepseek-flash','usage':None}])
        self.assertIsNone(costs['calculated_max_usd'])
        self.assertEqual(costs['models'][0]['unknown_usage_calls'],1)
        self.assertIsNone(cost_breakdown([{'provider':'custom','model':'deepseek-flash','usage':record['usage']}])['calculated_total_usd'])

    def test_cache_details_do_not_double_count_tokens(self):
        from ml_engine import TierBudget,_ClientSwitch
        from model_budget import ModelLimits
        class WithCache:
            model='deepseek-flash';base_url='https://api.deepseek.com/v1';max_output_tokens=20
            last_usage={};last_billing_usage={}
            def __call__(self,messages,timeout):
                self.last_usage={'prompt_tokens':100,'completion_tokens':10}
                self.last_billing_usage={'prompt_cache_hit_tokens':80,'prompt_cache_miss_tokens':20}
                return '{}'
        gateway=TierBudget(_ClientSwitch({'low':WithCache()}),ModelLimits(2,4000,30))
        gateway.call('low',[{'role':'user','content':'JSON'}])
        snapshot=gateway.snapshot()
        self.assertEqual(snapshot['known_tokens'],110)
        self.assertEqual(snapshot['records'][0]['billing_usage']['prompt_cache_hit_tokens'],80)
        self.assertEqual(snapshot['costs']['models'][0]['known_input_tokens'],100)

    def test_failed_output_with_real_usage_remains_counted(self):
        from ml_engine import TierBudget,_ClientSwitch
        from model_budget import ModelLimits
        class Truncated:
            last_usage={}
            max_output_tokens=100
            def __call__(self,messages,timeout):
                self.last_usage={'prompt_tokens':10,'completion_tokens':100}
                raise ValueError('output truncated')
        gateway=TierBudget(_ClientSwitch({'high':Truncated()}),ModelLimits(3,4000,30))
        with self.assertRaises(ValueError):gateway.call('high',[{'role':'user','content':'JSON'}])
        state=gateway.snapshot()
        self.assertEqual(state['known_tokens'],110)
        self.assertEqual(state['unreconciled_calls'],0)
        self.assertEqual(state['records'][0]['status'],'failed_with_usage')

    def test_comparison_uses_same_data_and_high_only_never_calls_router(self):
        class Generated:
            max_output_tokens=768
            def __init__(self,model,classifier=False): self.model=model;self.classifier=classifier;self.last_usage={}
            def __call__(self,messages,timeout):
                self.last_usage={'prompt_tokens':20,'completion_tokens':15}
                if self.classifier:return json.dumps({'category':'model_search','confidence':.95})
                if messages[0]['content'].startswith('Audit'):
                    return json.dumps({'decision':'approve','answer':'根据当前验证数据给出分类预测；不作因果解释。','summary':'checked','experiment_ids':['exp_001','exp_002']})
                return json.dumps({'model':'random_forest','parameters':{'n_estimators':20,'max_depth':3},'hypothesis':'nonlinear boundaries','expected_direction':'up','falsifier':'no validation gain'})
        def factory():
            return {'source':Generated('fixture-high'),'low':Generated('fixture-low'),'high':Generated('fixture-high'),'reviewer':Generated('fixture-high'),'classifier':Generated('fixture-jev',True)}
        csv=load_iris(as_frame=True).frame.to_csv(index=False)
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            with TestClient(create_app(tmp,client_factory=factory)) as client:
                r=client.post('/ml/comparisons',json={'goal':'Compare iris classifiers','csv_text':csv,'target':'target','seeds':[42],'max_fits':2})
                self.assertEqual(r.status_code,202,r.text)
                comparison_id=r.json()['comparison_id']
                result=client.get('/ml/comparisons/'+comparison_id).json()
                self.assertEqual(result['status'],'completed',result)
                report=result['report']
                self.assertEqual(len(report['arms']),2)
                self.assertTrue(all(a['metrics']['status']=='completed' for a in report['arms']),report)
                metrics={a['mode']:a['metrics'] for a in report['arms']}
                self.assertEqual(metrics['routed']['calls'],3)
                self.assertEqual(metrics['high_only']['calls'],2)
                self.assertFalse(metrics['high_only']['test_evaluated'])
                self.assertEqual(report['protocol']['memory'],'disabled_for_both')
                base=Path(tmp)/'comparisons'/comparison_id
                high=json.loads((base/'42-high_only/report.json').read_text(encoding='utf-8'))
                routed=json.loads((base/'42-routed/report.json').read_text(encoding='utf-8'))
                self.assertEqual(set(high['model_roles'].values()),{'fixture-high'})
                self.assertEqual(high['best']['dataset_sha256'],routed['best']['dataset_sha256'])
                self.assertEqual(high['best']['split_sizes'],routed['best']['split_sizes'])
                self.assertEqual([r['role'] for r in high['usage']['records']],['high','reviewer'])
                self.assertFalse((Path(tmp)/'experiment-lessons.sqlite3').exists())
                self.assertEqual(client.get('/ml/comparisons/'+comparison_id+'/export').status_code,200)
                self.assertEqual(client.post('/ml/comparisons',json={'goal':'compare','seeds':[42,42]}).status_code,400)

    def test_comparison_incomplete_usage_is_not_reported_as_savings(self):
        from comparison_benchmark import aggregate
        base={'status':'completed','research_status':'approved','validation_macro_f1':.8,'minority_f1':.6,'known_tokens':100,'wall_seconds':1,'calls':2,'training_fits':2,'usage_complete':False}
        result=aggregate([{'mode':'routed','seed':42,'metrics':base},{'mode':'high_only','seed':42,'metrics':{**base,'usage_complete':True}}])
        self.assertIsNone(result['paired_deltas_routed_minus_high_only'][0]['known_tokens_delta'])
        failure=aggregate([{'mode':'high_only','seed':42,'metrics':{'status':'failed','wall_seconds':3}}])
        self.assertEqual(failure['by_mode']['high_only']['runs'],1)
        self.assertEqual(failure['by_mode']['high_only']['completed'],0)
        self.assertIsNone(failure['by_mode']['high_only']['known_tokens']['mean'])

    def test_comparison_blocked_arms_are_not_averaged_as_results(self):
        from comparison_benchmark import aggregate
        base={'status':'completed','research_status':'blocked','baseline_only':True,'analysis_scope':None,
              'validation_macro_f1':.5969,'minority_f1':.26,'known_tokens':668,'wall_seconds':1,
              'calls':3,'training_fits':1,'usage_complete':True,'costs':{'calculated_total_usd':.001}}
        result=aggregate([{'mode':'routed','seed':42,'metrics':base},
                          {'mode':'high_only','seed':42,'metrics':{**base,'known_tokens':0,'calls':1}}])
        self.assertIsNone(result['by_mode']['routed']['validation_macro_f1']['mean'])
        self.assertIsNone(result['by_mode']['high_only']['minority_f1']['mean'])
        self.assertEqual(result['by_mode']['routed']['not_research_ready'],1)
        self.assertEqual(result['by_mode']['routed']['runs'],1)
        pair=result['paired_deltas_routed_minus_high_only'][0]
        self.assertFalse(pair['comparable'])
        self.assertEqual(pair['reason'],'unfinished_arm')
        self.assertIsNone(pair['macro_f1_delta'])
        self.assertIsNone(pair['calculated_usd_delta'])
        self.assertIsNone(pair['known_tokens_delta'])

    def test_encrypted_credentials_survive_restart_and_partial_edit(self):
        from credential_store import CredentialStore
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            with TestClient(create_app(tmp)) as client:
                response=client.post('/ml/credentials',json={'deepseek':'test-deepseek-original','jev':'test-jev-original'})
                self.assertEqual(response.status_code,200,response.text)
                self.assertTrue(response.json()['persisted'])
            raw=(Path(tmp)/'credentials.dpapi').read_bytes()
            self.assertNotIn(b'test-deepseek-original',raw)
            self.assertNotIn(b'test-jev-original',raw)
            with TestClient(create_app(tmp)) as restarted:
                state=restarted.get('/ml/credentials/status')
                self.assertTrue(state.json()['deepseek_configured'])
                self.assertTrue(state.json()['jev_configured'])
                self.assertNotIn('test-deepseek-original',state.text)
                config=restarted.get('/ml/config')
                self.assertTrue(config.json()['configured'])
                self.assertNotIn('test-jev-original',config.text)
                response=restarted.post('/ml/credentials',json={'deepseek':'test-deepseek-updated'})
                self.assertEqual(response.status_code,200)
            restored=CredentialStore(Path(tmp)/'credentials.dpapi').load()
            self.assertEqual(restored['DEEPSEEK_API_KEY'],'test-deepseek-updated')
            self.assertEqual(restored['JEV_API_KEY'],'test-jev-original')

    def test_credential_save_error_keeps_previous_configuration(self):
        from credential_store import CredentialStore,CredentialStoreError
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            with TestClient(create_app(tmp)) as client:
                self.assertEqual(client.post('/ml/credentials',json={'deepseek':'old-deepseek-test','jev':'old-jev-test'}).status_code,200)
                with patch.object(CredentialStore,'save',side_effect=CredentialStoreError('Save failed')):
                    response=client.post('/ml/credentials',json={'deepseek':'new-deepseek-test'})
                    self.assertEqual(response.status_code,500)
                response=client.post('/ml/credentials',json={'deepseek':{'unexpected':'sensitive-test-string'}})
                self.assertEqual(response.status_code,400)
                self.assertNotIn('sensitive-test-string',response.text)
                response=client.post('/ml/credentials',json={'deepseek':'new-test-key'},headers={'Origin':'http://untrusted.invalid'})
                self.assertEqual(response.status_code,403)
            self.assertEqual(CredentialStore(Path(tmp)/'credentials.dpapi').load()['DEEPSEEK_API_KEY'],'old-deepseek-test')

    def test_research_answer_precedes_collapsed_developer_controls(self):
        html=(Path(__file__).resolve().parents[1]/'static/ml-app.html').read_text(encoding='utf-8')
        panel=html.index('<details class="card developer-panel"')
        self.assertLess(html.index('id="direct-answer"'),panel)
        for name in ('calls','tokens','fits','experts','score','usage','details','feature-answer'):
            self.assertGreater(html.index('id="'+name+'"'),panel)
        self.assertNotIn('open',html[panel:html.index('>',panel)])

    def test_windows_locked_state_uses_journal_and_poll_reads_latest(self):
        from research_system import save,read_saved_state
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/('b'*32);folder.mkdir()
            save(folder/'state.json',{'status':'queued'})
            original=Path.replace
            def locked(source,target):
                if Path(target).name=='state.json': raise PermissionError('simulated Windows lock')
                return original(source,target)
            with patch.object(Path,'replace',locked),patch('research_system.time.sleep'):
                save(folder/'state.json',{'status':'running'})
            self.assertEqual(read_saved_state(folder/'state.json')['status'],'running')
            with asgi_test_runtime(),TestClient(create_app(tmp)) as client:
                self.assertEqual(client.get('/ml/runs/'+'b'*32).json()['status'],'running')

    def test_compact_poll_preserves_full_report_and_limits_preview(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            root=Path(tmp);run_id='a'*32;folder=root/run_id;folder.mkdir()
            (folder/'state.json').write_text(json.dumps({'status':'completed'}),encoding='utf-8')
            report={'research_outputs':{'predictions':[{'source_row':i} for i in range(1000)]}}
            (folder/'report.json').write_text(json.dumps(report),encoding='utf-8')
            with TestClient(create_app(root)) as client:
                short=client.get('/ml/runs/'+run_id+'?compact=true').json()['report']['research_outputs']
                self.assertEqual(len(short['predictions']),20)
                self.assertEqual(short['predictions_total'],1000)
                full=client.get('/ml/runs/'+run_id).json()['report']['research_outputs']
                self.assertEqual(len(full['predictions']),1000)
                self.assertEqual(len(json.loads((folder/'report.json').read_text())['research_outputs']['predictions']),1000)

    def test_lessons_persist_and_do_not_apply_to_other_targets(self):
        from experiment_lessons import LessonStore
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);engine=self.engine(root,clients())
            path=root/'lessons.sqlite3'
            proposal={'model':'random_forest','parameters':{'n_estimators':20,'max_depth':3}}
            store=LessonStore(path,engine.session)
            store.record(proposal,'no_gain',.5,.8)
            restored=LessonStore(path,engine.session)
            self.assertTrue(restored.should_skip(proposal,.8))
            self.assertFalse(restored.should_skip(proposal,.4))
            self.assertFalse(restored.should_skip({'model':'random_forest','parameters':{'n_estimators':30}},.8))
            engine.session.target='other'
            self.assertEqual(LessonStore(path,engine.session).snapshot(),[])

    def test_worker_consults_lessons_and_skips_measured_duplicate(self):
        from experiment_lessons import LessonStore
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);engine=self.engine(root,clients())
            engine.lessons=LessonStore(root/'lessons.sqlite3',engine.session)
            proposal={'model':'random_forest','parameters':{'n_estimators':20,'max_depth':3}}
            engine.lessons.record(proposal,'no_gain',0.0,.9)
            report=engine.run('Compare classifiers')
            self.assertEqual(report['training_fits'],1)
            self.assertTrue(any(e['event']=='lessons_consulted' for e in report['events']))
            self.assertTrue(any(e['event']=='lesson_prevented_repeat' for e in report['events']))
            self.assertTrue(report['lessons'])

    def test_prediction_delivery_matches_validation_rows_only(self):
        from research_deliverables import deliver
        from sklearn.metrics import f1_score
        with tempfile.TemporaryDirectory() as tmp:
            engine=self.engine(Path(tmp),clients())
            result=engine.session.run('random_forest',{'n_estimators':20})
            out=deliver(engine.session,result,'给出分类预测和影响因素')
            self.assertEqual(out['records_predicted'],len(engine.session.validation_indices))
            self.assertEqual(sum(out['prediction_counts'].values()),out['records_predicted'])
            rows=out['predictions']
            self.assertEqual({r['source_row']-1 for r in rows},set(engine.session.validation_indices))
            self.assertTrue({r['source_row']-1 for r in rows}.isdisjoint(engine.session.test_indices))
            self.assertAlmostEqual(f1_score([r['actual_label'] for r in rows],[r['predicted_label'] for r in rows],average='macro'),out['validation_macro_f1'])
            self.assertTrue(all(abs(sum(r['probabilities'].values())-1)<1e-6 for r in rows))
            self.assertFalse(engine.session.closed)

    def test_persisted_champion_starts_next_run_and_wrong_target_is_rejected(self):
        from research_deliverables import ExperienceBank
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);engine=self.engine(root,clients())
            result=engine.session.run('random_forest',{'n_estimators':20,'max_depth':3})
            saved=root/'previous';saved.mkdir()
            report={'research_status':'approved','best':result,'protocol':{'target':engine.session.target}}
            (saved/'report.json').write_text(json.dumps(report),encoding='utf-8')
            strategy=ExperienceBank(root).find(engine.session)
            self.assertEqual(strategy['model'],'random_forest')
            next_engine=self.engine(root,clients())
            next_engine.experience=strategy
            report2=next_engine.run('比较模型并给出预测')
            self.assertTrue(report2['experience']['reused'])
            self.assertEqual(report2['experiments'][0]['model'],'random_forest')
            self.assertEqual(len(report2['experiments']),1)
            self.assertEqual(report2['research_status'],'approved')
            engine.session.target='different_target'
            self.assertIsNone(ExperienceBank(root).find(engine.session))

    def test_csv_bypasses_source_planning(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=clients()
            c['source']=Reply([],'must-not-be-called')
            report=self.engine(Path(tmp),c).run('Compare classifiers')
            self.assertEqual(c['source'].calls,0)
            self.assertTrue(any(e['event']=='classification_started' for e in report['events']))

    def test_structured_model_payload_disables_deepseek_thinking(self):
        from light_model import LightModelClient
        client=LightModelClient('placeholder-key',model='deepseek-v4-pro',structured_output=True)
        response={'choices':[{'message':{'content':'{"ok":true}'},'finish_reason':'stop'}]}
        with patch.object(client,'_post',return_value=response) as post:
            client([{'role':'user','content':'Return JSON.'}])
            payload=post.call_args.args[0]
            self.assertEqual(payload['response_format'],{'type':'json_object'})
            self.assertEqual(payload['thinking'],{'type':'disabled'})

    def test_local_catalog_search_and_load_without_network(self):
        from local_catalog import LocalCatalog
        with tempfile.TemporaryDirectory() as tmp:
            catalog=LocalCatalog(tmp)
            with patch('urllib.request.build_opener',side_effect=AssertionError('No network permitted')):
                self.assertEqual(catalog.search('鸢尾花')[0]['id'],'iris')
                self.assertEqual(catalog.search('wine')[0]['id'],'wine')
                csv,target,provenance=catalog.download('iris')
                self.assertIn(target,csv.splitlines()[0])
                self.assertTrue(provenance['offline'])
                self.assertEqual(catalog.search('unrelated nonexistent domain'),[])

    def test_local_catalog_detects_corrupted_data(self):
        from local_catalog import LocalCatalog
        with tempfile.TemporaryDirectory() as tmp:
            catalog=LocalCatalog(tmp)
            (Path(tmp)/'iris.csv').write_text('tampered',encoding='utf-8')
            with self.assertRaises(ValueError): catalog.download('iris')

    def test_default_application_is_the_reduced_pipeline(self):
        from api import create_app as default_app
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            with TestClient(default_app(tmp)) as client:
                response=client.get('/')
                self.assertIn('/static/ml-app.js',response.text)
                self.assertEqual(client.get('/static/ml-app.css').status_code,200)
                self.assertEqual(client.get('/strategy/state').status_code,404)
                self.assertEqual(client.post('/demo/analyze',json={}).status_code,404)

    def test_specialized_classifier_uses_official_choice_contract(self):
        classifier=JevClassifier('fixture-key')
        envelope={'answers':{'category':{'type':'choice','choice':'model_search','confidence':.9}},
                  'usage':{'input_tokens':10,'output_tokens':2}}
        opener=MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value=json.dumps(envelope).encode()
        with patch('ml_engine.request.build_opener',return_value=opener):
            answer=json.loads(classifier([{'role':'user','content':json.dumps({'task':{'objective':'compare models'}})}]))
        sent=opener.open.call_args.args[0]
        payload=json.loads(sent.data)
        self.assertEqual(payload['questions']['category']['criteria'],CATEGORIES)
        self.assertEqual(answer['category'],'model_search')
        self.assertEqual(classifier.last_usage,{'prompt_tokens':10,'completion_tokens':2})

    def engine(self,root,registry,**settings):
        csv=root/'data.csv';load_iris(as_frame=True).frame.to_csv(csv,index=False)
        return MLResearch(ResearchSession(csv,'target'),registry,**settings)

    def test_csv_chain_uses_low_then_separate_final_reviewer(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=clients(); report=self.engine(Path(tmp),c).run('Compare classifiers')
            self.assertEqual([r['role'] for r in report['usage']['records']],['classifier','low','reviewer'])
            self.assertEqual(c['high'].calls,0)
            self.assertEqual(report['training_fits'],2)
            self.assertEqual(report['review']['decision'],'approve')
            self.assertFalse(report['test_evaluated'])

    def test_low_failure_escalates_once_then_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=clients(low_bad=True);report=self.engine(Path(tmp),c).run('Compare classifiers')
            self.assertEqual(c['high'].calls,1)
            self.assertTrue(any(e['event']=='escalation' for e in report['events']))
            self.assertEqual(report['usage']['calls'],4)

    def test_classifier_cannot_choose_model_or_override_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=clients(category='send_everything_to_high')
            report=self.engine(Path(tmp),c).run('Compare classifiers')
            self.assertEqual(c['low'].calls,0)
            self.assertEqual(c['high'].calls,1)
            self.assertEqual(report['research_status'],'blocked')
            self.assertEqual(c['reviewer'].calls,0)

    def test_low_confidence_rechecks_then_really_executes(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=clients(confidence=.3)
            c['high'].replies.insert(0,{'decision':'dispatch','category':'model_search','tier':'low','reason':'当前数据支持分类模型实验。','questions':[]})
            report=self.engine(Path(tmp),c).run('Compare classifiers')
            self.assertEqual(c['high'].calls,1)
            self.assertEqual(c['low'].calls,1)
            self.assertEqual(c['reviewer'].calls,1)
            self.assertEqual(report['training_fits'],2)
            self.assertTrue(any(e['event']=='route_recheck_resolved' for e in report['events']))
            self.assertEqual(next(e for e in report['events'] if e['event']=='route')['reason'],'high_model_recheck')

    def test_unresolved_recheck_requests_info_without_final_answer_or_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=clients(confidence=.2)
            c['high'].replies=[{'decision':'clarify','category':None,'tier':None,'reason':'目标定义不明确。','questions':['希望预测哪个字段？']}]
            engine=self.engine(Path(tmp),c)
            engine.deliverer=lambda *args: self.fail('Must not generate research outputs')
            report=engine.run('Research unclear objective',True)
            self.assertEqual(report['research_status'],'needs_clarification')
            self.assertEqual(report['clarification']['questions'],['希望预测哪个字段？'])
            self.assertEqual(c['low'].calls,0)
            self.assertEqual(c['reviewer'].calls,0)
            self.assertIsNone(report['research_outputs'])
            self.assertFalse(engine.session.closed)

    def test_partial_causal_review_runs_real_association_without_predictions_or_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=clients(confidence=.95)
            engine=self.engine(Path(tmp),c)
            column=str(engine.session.x.columns[0])
            c['high'].replies=[{'decision':'partial','category':None,'tier':None,'reason':'没有因果识别设计，但可以计算分组关联。','questions':[],'association_columns':[column]}]
            report=engine.run('现有数据能否支持因果结论？',True)
            self.assertEqual(report['research_status'],'partial_completed')
            self.assertTrue(any(e['event']=='association_completed' for e in report['events']))
            evidence=report['research_outputs']['association_evidence']
            self.assertEqual(evidence['observations'],len(engine.session.train_indices))
            self.assertFalse(evidence['causal_identification_performed'])
            groups=evidence['features'][0]['groups']
            self.assertEqual(sum(g['n'] for g in groups),len(engine.session.train_indices))
            for g in groups:
                self.assertAlmostEqual(g['rate'],g['positive_count']/g['n'])
                self.assertLessEqual(g['ci95_lower'],g['rate'])
                self.assertGreaterEqual(g['ci95_upper'],g['rate'])
            self.assertEqual(report['research_outputs']['records_predicted'],0)
            self.assertEqual(c['low'].calls,0)
            self.assertEqual(c['reviewer'].calls,1)
            self.assertFalse(engine.session.closed)

    def test_partial_association_api_completes_with_limits_and_no_prediction_export(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients(confidence=.95)
            column=str(load_iris(as_frame=True).frame.columns[0])
            c['high'].replies=[{'decision':'partial','reason':'只能支持关联，不能识别因果。','questions':[],'association_columns':[column]}]
            with TestClient(create_app(tmp,client_factory=lambda:c)) as client:
                response=client.post('/ml/runs',json={'goal':'现有数据能否证明因果？','csv_text':load_iris(as_frame=True).frame.to_csv(index=False),'target':'target','evaluate_test':True})
                self.assertEqual(response.status_code,202)
                run_id=response.json()['run_id']
                state=client.get('/ml/runs/'+run_id).json()
                self.assertEqual(state['status'],'partial_completed')
                self.assertEqual(state['report']['research_outputs']['scope'],'training_association_only')
                self.assertFalse(state['report']['test_evaluated'])
                self.assertEqual(client.get('/ml/runs/'+run_id+'/predictions').status_code,409)

    def test_partial_review_cannot_invent_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=clients(confidence=.1)
            c['high'].replies=[{'decision':'partial','reason':'可完成关联分析','questions':[],'association_columns':['invented-column']}]
            report=self.engine(Path(tmp),c).run('Can the data establish causality?')
            self.assertEqual(report['research_status'],'needs_clarification')
            self.assertEqual(c['reviewer'].calls,0)
            self.assertIsNone(report['research_outputs'])
            recorded=next(e for e in report['events'] if e['event']=='route_recheck_invalid_columns')
            self.assertIn('invented-column',recorded['returned'])

    def test_partial_review_drops_target_echo_and_keeps_the_real_feature(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=clients(confidence=.95)
            engine=self.engine(Path(tmp),c)
            column=str(engine.session.x.columns[0])
            c['high'].replies=[{'decision':'partial','reason':'只能支持关联分析，不能识别因果。','questions':[],
                                'association_columns':[column,'target']}]
            report=engine.run('现有数据能否支持因果结论？',True)
            self.assertEqual(report['research_status'],'partial_completed')
            evidence=report['research_outputs']['association_evidence']
            self.assertEqual([f['feature'] for f in evidence['features']],[column])
            adjusted=next(e for e in report['events'] if e['event']=='association_columns_adjusted')
            self.assertEqual(adjusted['used'],[column])
            self.assertIn('target',adjusted['requested'])

    def test_partial_review_accepts_case_and_single_string_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=clients(confidence=.95)
            engine=self.engine(Path(tmp),c)
            column=str(engine.session.x.columns[0])
            c['high'].replies=[{'decision':'partial','reason':'完成可支持的关联分析。','questions':[],
                                'association_columns':column.upper()}]
            report=engine.run('现有数据能否支持因果结论？',True)
            self.assertEqual(report['research_status'],'partial_completed')
            self.assertEqual(report['research_outputs']['association_evidence']['features'][0]['feature'],column)
            self.assertFalse(any(e['event']=='route_recheck_invalid_columns' for e in report['events']))

    def test_recheck_can_dispatch_high_worker_after_its_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=clients(confidence=.2)
            c['high'].replies.insert(0,{'decision':'dispatch','category':'failure_diagnosis','tier':'high','reason':'需要高级工作站实验。','questions':[]})
            report=self.engine(Path(tmp),c).run('Compare classifiers')
            self.assertEqual(c['high'].calls,2)
            self.assertEqual(c['low'].calls,0)
            self.assertEqual(report['training_fits'],2)
            self.assertEqual(report['expert_calls'],1)

    def test_api_unresolved_recheck_exposes_needs_input(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients(confidence=.1)
            c['high'].replies=[{'decision':'clarify','category':None,'tier':None,'reason':'缺少研究范围。','questions':['请说明研究对象。']}]
            with TestClient(create_app(tmp,client_factory=lambda:c)) as client:
                r=client.post('/ml/runs',json={'goal':'Research the dataset','csv_text':load_iris(as_frame=True).frame.to_csv(index=False),'target':'target','evaluate_test':True})
                self.assertEqual(r.status_code,202)
                state=client.get('/ml/runs/'+r.json()['run_id']).json()
                self.assertEqual(state['status'],'needs_input')
                self.assertIsNone(state['report']['research_outputs'])
                self.assertEqual(c['reviewer'].calls,0)

    def test_recheck_budget_insufficient_stops_without_paid_high_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=clients(confidence=.1)
            report=self.engine(Path(tmp),c,max_calls=3).run('Compare classifiers')
            self.assertEqual(report['research_status'],'blocked')
            self.assertEqual(c['high'].calls,0)
            self.assertEqual(c['reviewer'].calls,0)

    def test_classifier_error_can_be_recovered_by_high_recheck(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=clients()
            c['classifier']=Reply(['invalid JSON'],'fixture-classifier')
            c['high'].replies.insert(0,{'decision':'dispatch','category':'model_search','tier':'low','reason':'数据适合分类预测。','questions':[]})
            report=self.engine(Path(tmp),c).run('Compare classifiers')
            self.assertEqual(report['training_fits'],2)
            self.assertEqual(c['high'].calls,1)

    def test_unsupported_low_capability_routes_to_high_with_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=clients(category='failure_diagnosis');report=self.engine(Path(tmp),c).run('Diagnose classifier failure')
            self.assertEqual(c['low'].calls,0)
            self.assertEqual(c['high'].calls,1)
            self.assertEqual(next(e for e in report['events'] if e['event']=='route')['reason'],'low_unsupported_capability')

    def test_reviewer_reject_cannot_open_final_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=clients(review='reject');report=self.engine(Path(tmp),c).run('Compare classifiers',evaluate_test=True)
            self.assertIsNone(report['final_test'])
            self.assertEqual(report['research_status'],'needs_revision')

    def test_shared_call_budget_reserves_reviewer(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=clients();report=self.engine(Path(tmp),c,max_calls=3).run('Compare classifiers')
            self.assertLessEqual(report['usage']['calls'],3)
            self.assertEqual(c['reviewer'].calls,1)

    def test_no_generated_data_or_private_url_download(self):
        for url in ('http://www.openml.org/file','https://127.0.0.1/x','https://example.com/x','file:///secret'):
            with self.assertRaises(ValueError): OpenMLCatalog().fetch(url)

    def test_natural_language_full_api_chain_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients(natural=True)
            with TestClient(create_app(tmp,client_factory=lambda:c,catalog_factory=CatalogFixture)) as client:
                response=client.post('/ml/runs',json={'goal':'Find public iris data and compare classifiers.'})
                self.assertEqual(response.status_code,202,response.text)
                run_id=response.json()['run_id']
                state=client.get('/ml/runs/'+run_id).json()
                self.assertEqual(state['status'],'completed',state)
                self.assertEqual(state['report']['source']['provider'],'test_catalog')
                self.assertEqual(c['source'].calls,2)
                self.assertEqual(state['report']['usage']['calls'],5)
                self.assertEqual(client.get('/ml/runs/'+run_id+'/export').status_code,200)
                predictions=client.get('/ml/runs/'+run_id+'/predictions')
                self.assertEqual(predictions.status_code,200)
                self.assertIn('predicted_label',predictions.text)
                self.assertEqual(state['report']['research_outputs']['scope'],'validation_rows_only')
                self.assertEqual(client.get('/').status_code,200)
                self.assertEqual(client.get('/demo/analyze').status_code,404)

    def test_search_retries_core_keyword_without_extra_model_calls(self):
        attempts=[]
        class RetryCatalog(CatalogFixture):
            def search(self,query):
                attempts.append(query)
                return [{'id':'61','name':'customer-churn-fixture'}] if query=='churn' else []
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients(natural=True)
            c['source'].replies[0]={'query':'customer churn'}
            with TestClient(create_app(tmp,client_factory=lambda:c,catalog_factory=RetryCatalog)) as client:
                r=client.post('/ml/runs',json={'goal':'Research customer churn.'}).json()
                state=client.get('/ml/runs/'+r['run_id']).json()
                self.assertEqual(state['status'],'completed',state)
                self.assertEqual(attempts,['customer churn','customer_churn','churn'])
                self.assertEqual(c['source'].calls,2)
                self.assertTrue(any(e['event']=='source_search_retry' for e in state['events']))

    def test_no_search_results_stops_without_inventing_csv(self):
        class EmptyCatalog(CatalogFixture):
            def search(self,query): return []
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients(natural=True)
            with TestClient(create_app(tmp,client_factory=lambda:c,catalog_factory=EmptyCatalog)) as client:
                r=client.post('/ml/runs',json={'goal':'Find public iris data.'}).json()
                state=client.get('/ml/runs/'+r['run_id']).json()
                self.assertEqual(state['status'],'needs_input')
                self.assertEqual(state['report']['training_fits'],0)
                self.assertIsNone(state['report']['research_outputs'])
                self.assertEqual(c['reviewer'].calls,0)
                self.assertFalse((Path(tmp)/r['run_id']/'data.csv').exists())


if __name__=='__main__': unittest.main()
