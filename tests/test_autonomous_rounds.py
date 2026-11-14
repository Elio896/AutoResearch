"""Real sklearn execution with deterministic model-response fixtures."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from executor import ResearchSession
from ml_engine import MLResearch
from research_deliverables import deliver
from tests.test_ml_pipeline import Reply


def proposal(trees,depth):
    return {'model':'random_forest','parameters':{'n_estimators':trees,'max_depth':depth},
            'hypothesis':'A bounded tree ensemble may capture nonlinear interactions.',
            'expected_direction':'up','falsifier':'Validation macro F1 fails to exceed the incumbent.'}


class CaptureReply(Reply):
    def __init__(self,replies,model):super().__init__(replies,model);self.messages=[]
    def __call__(self,messages,timeout):self.messages.append(messages);return super().__call__(messages,timeout)


class AutonomousRoundTests(unittest.TestCase):
    def engine(self,root,*,proposals=None,**settings):
        x,y=make_classification(n_samples=240,n_features=6,n_informative=4,n_redundant=1,flip_y=.08,random_state=19)
        frame=pd.DataFrame(x,columns=['f'+str(i) for i in range(6)]);frame['label']=y
        p=root/'data.csv';frame.to_csv(p,index=False)
        low=CaptureReply(proposals or [proposal(20,2),proposal(30,3),proposal(40,4)],'fixture-low')
        clients={'source':Reply([],'fixture-source'),'low':low,'high':Reply([],'fixture-high'),
                 'classifier':Reply([{'category':'model_search','confidence':.95}]*8,'fixture-classifier'),
                 'reviewer':Reply([{'decision':'approve','answer':'已验证候选方案，结论限于当前验证样本。','summary':'checked',
                                    'experiment_ids':['exp_001']}],'fixture-reviewer')}
        args={'max_rounds':3,'patience':5,'max_fits':4,'max_calls':12,'max_expert_calls':0,'deliverer':deliver}
        args.update(settings)
        return MLResearch(ResearchSession(p,'label'),clients,**args),clients

    def test_three_rounds_execute_feedback_validate_and_preserve_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine,c=self.engine(Path(tmp));report=engine.run('Compare classification experiments')
            self.assertEqual(report['training_fits'],4)
            self.assertEqual(report['iteration']['attempted_rounds'],3)
            self.assertEqual(c['reviewer'].calls,1)
            self.assertFalse(report['test_evaluated']);self.assertFalse(engine.session.closed)
            self.assertTrue(report['research_iteration_enabled'])
            self.assertTrue(any(e['event']=='research_replanned' for e in report['events']))
            self.assertNotIn('Research feedback',json.dumps(c['low'].messages[0]))
            self.assertIn('Research feedback',json.dumps(c['low'].messages[1]))
            for row in report['iteration']['rounds']:
                self.assertGreaterEqual(row['incumbent_after'],row['incumbent_before'])
                self.assertTrue(row['proposals'][0]['hypothesis'])
                self.assertTrue(row['proposals'][0]['falsifier'])
            best=report['best'];pipeline,_=engine.session.experiments[best['experiment_id']]
            actual=engine.session.y.iloc[engine.session.validation_indices]
            predicted=pipeline.predict(engine.session.x.iloc[engine.session.validation_indices])
            self.assertAlmostEqual(f1_score(actual,predicted,average='macro'),best['validation_score'])

    def test_calls_budget_preserves_final_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine,c=self.engine(Path(tmp),max_calls=3);report=engine.run('Compare classifiers')
            self.assertEqual(report['usage']['calls'],3)
            self.assertEqual(report['iteration']['attempted_rounds'],1)
            self.assertEqual(report['iteration']['stop_reason'],'reserve_final_reviewer')
            self.assertEqual(c['reviewer'].calls,1)

    def test_training_budget_stops_without_extra_provider_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine,c=self.engine(Path(tmp),max_fits=2);report=engine.run('Compare classifiers')
            self.assertEqual(report['training_fits'],2)
            self.assertEqual(c['low'].calls,1)
            self.assertEqual(report['iteration']['stop_reason'],'training_budget')

    def test_duplicate_proposals_are_observed_without_duplicate_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=proposal(20,2);engine,c=self.engine(Path(tmp),proposals=[p,p,p]);report=engine.run('Compare classifiers')
            self.assertEqual(report['training_fits'],2)
            self.assertEqual(report['iteration']['attempted_rounds'],3)
            self.assertTrue(any(e['event']=='duplicate_proposal_skipped' for e in report['events']))
            self.assertEqual(sum(r['training_fits'] for r in report['iteration']['rounds']),1)

    def test_patience_stops_repeated_no_novel_experiments(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=proposal(20,2);engine,c=self.engine(Path(tmp),proposals=[p]*8,max_rounds=8,patience=2,max_fits=10)
            report=engine.run('Compare classifiers')
            self.assertLess(report['iteration']['attempted_rounds'],8)
            self.assertEqual(report['iteration']['stop_reason'],'no_gain_patience')
            self.assertEqual(report['training_fits'],2)

    def test_association_analysis_is_not_repeated_to_fill_round_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine,c=self.engine(Path(tmp),max_expert_calls=1)
            c['classifier'].replies=[{'category':'unknown','confidence':.1}]
            c['high'].replies=[{'decision':'partial','category':'model_search','tier':'low','reason':'仅能计算观察关联。',
                                'questions':[],'association_columns':['f0']}]
            report=engine.run('Can f0 establish causality?')
            self.assertEqual(report['research_status'],'partial_completed')
            self.assertEqual(report['iteration']['attempted_rounds'],1)
            self.assertEqual(report['iteration']['stop_reason'],'supported_analysis_complete')
            self.assertEqual(c['low'].calls,0)
            self.assertEqual(report['research_outputs']['records_predicted'],0)

    def test_invalid_iteration_policy_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for settings in ({'max_rounds':0},{'max_rounds':True},{'patience':0}):
                with self.subTest(settings=settings),self.assertRaises(ValueError):self.engine(Path(tmp),**settings)


if __name__=='__main__':unittest.main()
