import tempfile,unittest
from pathlib import Path
from sklearn.datasets import load_iris
from executor import ResearchSession
from orchestrator import ResearchOrchestrator

class BaselinePolicyTests(unittest.TestCase):
 def run_policy(self,budget):
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'iris.csv';load_iris(as_frame=True).frame.to_csv(path,index=False)
   session=ResearchSession(path,'target');calls=[]
   def propose(messages,timeout):
    calls.append(messages);return {'message':{'content':'{"model":"logistic_regression","parameters":{"C":5}}'}}
   result=ResearchOrchestrator(session,propose,max_experiments=budget,max_calls=1,baseline_policy='fixed_search').run()
   self.assertFalse(session.closed)
   self.assertEqual(result['baseline_policy'],'fixed_search')
   best=max(result['experiments'],key=lambda x:x['validation_score'])
   self.assertEqual(best['experiment_id'],result['best_experiment_id'])
   return result,calls
 def test_three_trials_use_entire_baseline_and_zero_model_calls(self):
  result,calls=self.run_policy(3)
  self.assertEqual([r['model'] for r in result['experiments']],['logistic_regression','random_forest','logistic_regression'])
  self.assertFalse(calls);self.assertEqual(result['model_calls'],0)
 def test_one_trial_never_exceeds_cap(self):
  result,calls=self.run_policy(1)
  self.assertEqual(len(result['experiments']),1);self.assertFalse(calls)
 def test_remaining_trial_is_available_to_agent(self):
  result,calls=self.run_policy(4)
  self.assertEqual(len(result['experiments']),4);self.assertEqual(len(calls),1)
