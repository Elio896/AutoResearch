import json,tempfile,unittest
from pathlib import Path
from tests import test_autonomous_rounds as rounds
proposal=rounds.proposal
from ml_engine import compact_review_payload


class EconomyTests(unittest.TestCase):
    def test_prediction_reuses_revalidated_history_without_new_tuning(self):
        from fastapi.testclient import TestClient
        from sklearn.datasets import load_iris
        from ml_api import create_app
        from tools.asgi_test_runtime import asgi_test_runtime
        from tests.test_ml_pipeline import clients
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients();body={'goal':'Compare classifiers','csv_text':load_iris(as_frame=True).frame.to_csv(index=False),'target':'target'}
            with TestClient(create_app(tmp,client_factory=lambda:c)) as api:
                rid=api.post('/ml/runs',json=body).json()['run_id'];first=api.get('/ml/runs/'+rid).json()['report']
                calls=c['low'].calls
                c['classifier'].replies.append({'category':'model_search','confidence':.95})
                c['reviewer'].replies.append({'decision':'approve','answer':'已验证历史方案并提供逐条预测。','summary':'checked','experiment_ids':['exp_001']})
                rid=api.post('/ml/runs',json={**body,'goal':'请预测这批记录并提供预测文件。','economy':True,'max_rounds':3}).json()['run_id']
                state=api.get('/ml/runs/'+rid).json();r=state['report']
                self.assertEqual(state['status'],'completed',state)
                self.assertEqual(r['training_fits'],1)
                self.assertEqual(c['low'].calls,calls)
                self.assertEqual(r['best']['validation_score'],first['best']['validation_score'])
                self.assertEqual(r['iteration']['stop_reason'],'revalidated_history_delivery')
                self.assertEqual(api.get('/ml/runs/'+rid+'/predictions').status_code,200)

    def test_plateau_substitutes_one_high_candidate_instead_of_more_low_tuning(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=proposal(20,2)
            engine,c=rounds.AutonomousRoundTests().engine(Path(tmp),economy=True,proposals=[p,p,p],max_expert_calls=1)
            c['high'].replies=[proposal(25,3)]
            r=engine.run('Compare classifiers')
            self.assertEqual(c['high'].calls,1)
            self.assertTrue(any(e.get('reason')=='economy_quality_protection_after_low_marginal_gain' for e in r['events']))
            for row in r['iteration']['rounds']:self.assertGreaterEqual(row['incumbent_after'],row['incumbent_before'])

    def test_stable_dispatch_is_reused_and_repeated_proposal_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=proposal(20,2)
            engine,c=rounds.AutonomousRoundTests().engine(Path(tmp),economy=True,proposals=[p,p,p])
            r=engine.run('Compare classification experiments')
            self.assertEqual(c['classifier'].calls,1)
            self.assertEqual(c['reviewer'].calls,1)
            self.assertEqual(r['training_fits'],2)
            self.assertEqual(r['iteration']['stop_reason'],'low_marginal_return')
            self.assertTrue(any(e['event']=='route_reused' for e in r['events']))
            self.assertEqual(r['research_status'],'approved')

    def test_failed_worker_invalidates_reused_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine,c=rounds.AutonomousRoundTests().engine(Path(tmp),economy=True,max_expert_calls=1)
            c['low'].replies[0]='invalid JSON';c['high'].replies=[proposal(16,2)]
            r=engine.run('Compare classifiers')
            self.assertGreaterEqual(c['classifier'].calls,2)
            self.assertEqual(c['high'].calls,1)
            self.assertTrue(any(e['event']=='escalation' for e in r['events']))
            self.assertGreaterEqual(r['training_fits'],2)

    def test_uncertain_classification_still_gets_real_high_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine,c=rounds.AutonomousRoundTests().engine(Path(tmp),economy=True)
            c['classifier'].replies=[{'category':'unknown','confidence':.1}]
            c['high'].replies=[{'decision':'dispatch','category':'model_search','tier':'low','reason':'Supported task','questions':[]}]
            r=engine.run('Compare classifiers')
            self.assertEqual(c['high'].calls,1)
            self.assertTrue(any(e['event']=='route_recheck_resolved' for e in r['events']))
            self.assertGreaterEqual(r['training_fits'],2)

    def test_compaction_keeps_actual_statistics_and_limits(self):
        evidence={'association_evidence':{'observations':3000,'features':[{'feature':'plan','groups':[{'n':288,'rate':.392,'ci':[.338,.45]}]}]},
                  'limitations':['No causal identification'],'prediction_download_url':None}
        full={'goal':'Compare groups','research_deliverables':evidence,'best':{'experiment_id':'exp_001','feature_review':['large repeated audit']*100},
              'experiments':[{'experiment_id':'exp_001','model':'random_forest','parameters':{'max_depth':4},'validation_f1':.9,'feature_review':['audit']*100}],
              'iteration':{'stop_reason':'round_limit','rounds':[]}}
        short=compact_review_payload(full)
        self.assertEqual(short['research_deliverables'],evidence)
        self.assertEqual(short['experiments'][0]['validation_f1'],.9)
        self.assertEqual(short['experiments'][0]['parameters'],{'max_depth':4})
        self.assertLess(len(json.dumps(short)),len(json.dumps(full))*.5)


if __name__=='__main__':unittest.main()
