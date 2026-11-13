import tempfile
import unittest
from pathlib import Path
from sklearn.datasets import make_classification
import pandas as pd
from executor import ResearchSession
from task_store import TaskStore

class MetricTests(unittest.TestCase):
    def test_minority_metrics_persist_and_finalize(self):
        with tempfile.TemporaryDirectory() as directory:
            x,y=make_classification(n_samples=240,n_features=5,n_informative=3,weights=[.9,.1],random_state=17)
            frame=pd.DataFrame(x,columns=list('abcde')); frame['target']=y
            path=Path(directory)/'data.csv'; frame.to_csv(path,index=False)
            store=TaskStore(Path(directory)/'tasks.sqlite3')
            task=store.create(path,'target',metric='f1_minority',pos_label='1')
            result=store.run(task,'logistic_regression',{})
            self.assertEqual(result['validation_score'],result['validation_f1_minority'])
            self.assertIn('validation_minority_recall',result)
            import json
            second=store.run(task,'random_forest',{'n_estimators':10})
            # Controlled persisted validation evidence: objectives disagree.
            # Finalization must choose minority score, never macro score.
            with store.connection() as db:
                for number, record, macro, objective in [(1,result,.99,.1),(2,second,.4,.8)]:
                    record.update(validation_f1=macro,validation_score=objective)
                    db.execute('UPDATE experiments SET result=? WHERE task_id=? AND number=?',
                               (json.dumps(record),task,number))
            reopened=TaskStore(Path(directory)/'tasks.sqlite3')
            self.assertEqual(reopened.show(task)['metric'],'f1_minority')
            final=reopened.finalize(task)
            self.assertEqual(final['test_score'],final['test_f1_minority'])
            self.assertEqual(final['experiment_id'],'exp_002')
            with self.assertRaises(RuntimeError): reopened.finalize(task)
            with self.assertRaises(ValueError): ResearchSession(path,'target',metric='f1_minority',pos_label='0')
            with self.assertRaises(ValueError): ResearchSession(path,'target',metric='accuracy')

    def test_macro_compatibility_and_distinct_objective(self):
        # Same predictions have excellent majority performance but zero minority recall.
        session=object.__new__(ResearchSession)
        session.metric='f1_minority'; session.pos_label='1'; session.positive_id=1
        result=session._metrics([0]*9+[1],[0]*10,'validation')
        self.assertEqual(result['validation_score'],0)
        session.metric='f1_macro'
        self.assertGreater(session._metrics([0]*9+[1],[0]*10,'validation')['validation_score'],.4)
