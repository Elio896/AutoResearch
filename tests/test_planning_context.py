import json
import tempfile
import unittest
from pathlib import Path
from sklearn.datasets import load_iris
from executor import ResearchSession
from orchestrator import ResearchOrchestrator

class PlanningContextTests(unittest.TestCase):
    def test_profile_never_reads_holdout_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'iris.csv'; load_iris(as_frame=True).frame.to_csv(path,index=False)
            session=ResearchSession(path,'target')
            before=session.planning_context()
            holdout=list(session.validation_indices)+list(session.test_indices)
            session.x.iloc[holdout,:]=float('nan')
            session.y.iloc[holdout]=999
            self.assertEqual(before,session.planning_context())
            self.assertEqual(before['scope'],'training_only')
            self.assertEqual(before['rows'],session.split_sizes['train'])

    def test_prompt_contains_profile_feedback_and_diversity_hint(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'iris.csv';load_iris(as_frame=True).frame.to_csv(path,index=False)
            session=ResearchSession(path,'target');captured=[]
            def proposer(messages,timeout):
                captured.append(json.dumps(messages))
                return {'message':{'content':'{"model":"random_forest","parameters":{"max_depth":3}}'}}
            report=ResearchOrchestrator(session,proposer,max_experiments=2,max_calls=1).run()
            self.assertIn('training_only',captured[0])
            self.assertIn('validation_class_metrics',captured[0])
            self.assertIn('untried model family',captured[0])
            self.assertNotIn('test_f1',captured[0])
            self.assertEqual(report['planning_policy'],'context_v2')
            self.assertEqual(report['model_calls'],1)
            self.assertFalse(session.closed)
