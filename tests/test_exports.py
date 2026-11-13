import io,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
from sklearn.datasets import load_iris

class ExportTests(unittest.TestCase):
 def setUp(self):
  from fastapi.testclient import TestClient
  from api import create_app
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.client=TestClient(create_app(Path(self.temp.name)));self.addCleanup(self.client.close)
  self.csv=load_iris(as_frame=True).frame.to_csv(index=False)
  self.task=self.client.post('/tasks',json={'csv_text':self.csv,'target':'target','max_experiments':3}).json()['task_id']
 def test_draft_export_is_read_only_and_path_free(self):
  before=self.client.get('/tasks/'+self.task).json()
  with patch('executor.ResearchSession.finalize',side_effect=AssertionError('export must not finalize')):
   response=self.client.get(f'/tasks/{self.task}/export?format=json')
  self.assertEqual(response.status_code,200,response.text)
  result=response.json();self.assertFalse(result['test_evaluated'])
  self.assertNotIn('csv',result['task']);self.assertNotIn(self.temp.name,response.text)
  self.assertEqual(before,self.client.get('/tasks/'+self.task).json())
 def test_baseline_bundle_and_final_state(self):
  self.client.post(f'/tasks/{self.task}/research',json={'baseline_policy':'fixed_search','max_calls':0})
  response=self.client.get(f'/tasks/{self.task}/export?format=zip')
  self.assertEqual(response.status_code,200)
  with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
   manifest=json.loads(bundle.read('manifest.json'))
   self.assertEqual([x['source'] for x in manifest['experiments']],['rule_baseline']*3)
   self.assertFalse(manifest['test_evaluated'])
   self.assertIn('replay_validation.py',bundle.namelist())
   self.assertEqual(manifest['usage']['model_calls'],0)
  self.assertEqual(self.client.post(f'/tasks/{self.task}/finalize').status_code,200)
  one=self.client.get(f'/tasks/{self.task}/export?format=json').json()
  two=self.client.get(f'/tasks/{self.task}/export?format=json').json()
  self.assertTrue(one['test_evaluated']);self.assertEqual(one['final_result'],two['final_result'])
  self.assertEqual(self.client.post(f'/tasks/{self.task}/finalize').status_code,409)
 def test_unknown_task_and_invalid_format(self):
  self.assertEqual(self.client.get('/tasks/missing/export').status_code,404)
  self.assertEqual(self.client.get(f'/tasks/{self.task}/export?format=pdf').status_code,422)

 def test_bundle_script_replays_validation_without_finalizing(self):
  import subprocess,sys
  self.client.post(f'/tasks/{self.task}/research',json={'baseline_policy':'fixed_search','max_calls':0})
  response=self.client.get(f'/tasks/{self.task}/export?format=zip')
  directory=Path(self.temp.name)/'replay';directory.mkdir()
  with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:bundle.extractall(directory)
  csv=directory/'original.csv';csv.write_bytes(self.csv.encode('utf-8'))
  before=self.client.get('/tasks/'+self.task).json()
  run=subprocess.run([sys.executable,str(directory/'replay_validation.py'),'--project',str(Path(__file__).resolve().parents[1]),'--csv',str(csv)],capture_output=True,text=True,timeout=30)
  self.assertEqual(run.returncode,0,run.stderr)
  rows=[json.loads(line) for line in run.stdout.splitlines()]
  self.assertEqual(len(rows),3)
  for actual,recorded in zip(rows,before['experiments']):
   self.assertAlmostEqual(actual['replayed_validation']['validation_score'],recorded['result']['validation_score'])
  self.assertEqual(before,self.client.get('/tasks/'+self.task).json())

 def test_risky_history_has_no_best_conclusion(self):
  from research_export import build_manifest,markdown
  self.client.post(f'/tasks/{self.task}/experiments',json={})
  task=self.client.get('/tasks/'+self.task).json()
  task['legacy_results_at_risk']=True
  exported=build_manifest(task)
  self.assertTrue(exported['conclusion_blocked']);self.assertIsNone(exported['best_validation'])
  self.assertIn('不能作为可靠研究结论',markdown(exported))
  task['research_job']={'state':'running','settings':{}}
  with self.assertRaises(RuntimeError):build_manifest(task)
