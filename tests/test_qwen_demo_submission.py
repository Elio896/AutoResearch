import tempfile,unittest,json,hashlib
from pathlib import Path
from unittest.mock import patch
from qwen_demo import DemoStore,run,Qwen,MODEL
from model_budget import ModelLimits
from tests.test_ml_pipeline import Reply

def candidate():return {'model':'random_forest','parameters':{'n_estimators':20,'max_depth':5},'hypothesis':'nonlinear boundary','expected_direction':'up','falsifier':'no validation gain'}

class QwenDemoSubmissionTests(unittest.TestCase):
    def data(self,root):
        from sklearn.datasets import load_iris
        p=Path(root)/'iris.csv';load_iris(as_frame=True).frame.rename(columns={'target':'label'}).to_csv(p,index=False)
        return [('iris',p,'label')]
    def test_original_qwen_and_original_six_datasets_kept(self):
        from qwen_demo import DATASETS,tasks_for
        self.assertEqual(Qwen().model,'qwen2.5:3b-instruct-q4_K_M')
        self.assertEqual(Qwen().base_url,'http://127.0.0.1:11434/v1')
        self.assertEqual(len(DATASETS),6)
        with tempfile.TemporaryDirectory() as d:
            rows=tasks_for(Path(__file__).resolve().parents[1],'mixed',Path(d))
            self.assertEqual(len(rows),6);self.assertTrue(all(p.exists() for _,p,_ in rows))
    def test_invalid_low_really_calls_high_and_fits_exports(self):
        with tempfile.TemporaryDirectory() as d:
            tasks=self.data(d);store=DemoStore(Path(d)/'demo');low=Reply(['bad'],MODEL);high=Reply([candidate()],'fixture-high')
            with patch('qwen_demo.tasks_for',return_value=tasks):r=run(store,d,rounds=1,low=low,high=high)
            self.assertEqual(low.calls,1);self.assertEqual(high.calls,1)
            self.assertEqual(r['usage']['calls'],2);self.assertEqual(r['summary']['tokens'],60)
            self.assertTrue(r['steps'][0]['high_attempted']);self.assertTrue(r['steps'][0]['delivered'])
            self.assertTrue((store.root/'runs'/r['run_id']/'step-1/predictions.csv').exists())
    def test_no_high_execution_never_counts_passed(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('qwen_demo.tasks_for',return_value=self.data(d)):
                r=run(DemoStore(Path(d)/'demo'),d,rounds=1,low=Reply(['bad'],MODEL),high=Reply(['bad'],'high'))
            self.assertEqual(r['summary']['passed'],0);self.assertEqual(r['summary']['completed'],0)
            self.assertEqual(r['state'],'failed')
    def test_snapshot_reset_and_rollback_restore_report_and_demo_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);production=root/'capability.sqlite3';production.write_bytes(b'production sentinel');store=DemoStore(root/'demo')
            with patch('qwen_demo.tasks_for',return_value=self.data(d)):
                first=run(store,d,rounds=1,low=Reply([candidate()],MODEL))
                second=run(store,d,rounds=1,low=Reply([candidate()],MODEL))
            self.assertNotEqual(first['run_id'],second['run_id']);store.rollback()
            self.assertEqual(store.report()['run_id'],first['run_id']);self.assertEqual(production.read_bytes(),b'production sentinel')
    def test_budget_stops_without_fake_completion(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('qwen_demo.tasks_for',return_value=self.data(d)):
                r=run(DemoStore(Path(d)/'demo'),d,rounds=3,low=Reply(['bad'],MODEL),high=Reply([candidate()],'high'),limits=ModelLimits(1,10000,30))
            self.assertEqual(r['usage']['calls'],1);self.assertGreater(r['summary']['unexecuted'],0);self.assertNotEqual(r['state'],'done')
    def test_api_keeps_qwen_and_exports_live_report(self):
        from fastapi.testclient import TestClient
        from tools.asgi_test_runtime import asgi_test_runtime
        from ml_api import create_app
        with tempfile.TemporaryDirectory() as d:
            with patch('qwen_demo.Qwen',return_value=Reply([candidate()],MODEL)),patch('qwen_demo.tasks_for',return_value=self.data(d)),asgi_test_runtime(),TestClient(create_app(Path(d)/'api',client_factory=lambda:{'high':Reply([candidate()],'high')})) as api:
                result=api.post('/ml/evolution/run',json={'rounds':1})
                self.assertEqual(result.status_code,200,result.text);self.assertEqual(result.json()['model'],MODEL)
                r=api.get('/ml/evolution/report').json();self.assertTrue(r['steps'][0]['delivered'])
                self.assertEqual(api.get('/ml/evolution/predictions/1').status_code,200)
                self.assertEqual(api.post('/ml/evolution/rollback').status_code,200)
                self.assertEqual(api.get('/ml/evolution/report').json()['steps'],[])

    def test_live_high_phase_is_persisted_after_budget_admission(self):
        with tempfile.TemporaryDirectory() as d:
            store=DemoStore(Path(d)/'demo');seen=[]
            class InspectHigh(Reply):
                def __call__(self,messages,timeout=30):
                    report=store.report()
                    seen.append((report['active']['event'],report['active']['tier'],report['usage']['records'][-1]['role']))
                    return super().__call__(messages,timeout)
            with patch('qwen_demo.tasks_for',return_value=self.data(d)):
                report=run(store,d,rounds=1,low=Reply(['bad'],MODEL),high=InspectHigh([candidate()],'high'))
            self.assertEqual(seen,[('worker_started','high','high')])
            self.assertTrue(any(e['event']=='escalation' for e in report['events']))
            self.assertEqual(report['events'][-1]['event'],'iteration_completed')
