import json,tempfile,unittest
from pathlib import Path
from sklearn.datasets import load_iris
from executor import ResearchSession
from ml_engine import MLResearch
from research_context import present_answer
from tests.test_ml_pipeline import clients

class AnswerSafety(unittest.TestCase):
    def test_regular_answer_adds_real_numbers_scope_and_download(self):
        url='/ml/runs/'+'a'*32+'/predictions'
        out={'scope':'validation_rows_only','records_predicted':100,'validation_macro_f1':.8123,'calibration_performed':False}
        r=present_answer({'decision':'approve','answer':'可将这些分类结果用于核查现有样本。'},'解释分类结果',out,url)
        for word in ('100','0.8123','未校准','方案选择','不是未来','不能证明因果',url):self.assertIn(word,r['answer'])
        self.assertEqual(r['answer_evidence']['download_url'],url)

    def test_does_not_manufacture_answer_or_approve_missing_direct_answer(self):
        r=present_answer({'decision':'approve','answer':''},'解释分类结果',{'scope':'validation_rows_only','records_predicted':100})
        self.assertEqual(r['decision'],'revise')
        self.assertIn('missing_direct_answer',r['unmet_requirements'])

    def test_association_has_no_prediction_link_and_is_not_causal(self):
        out={'scope':'training_association_only','records_predicted':0,'association_evidence':{'observations':60,'features':[]}}
        r=present_answer({'decision':'approve','answer':'当前只能提供关联描述。'},'解释关系',out,'/ml/runs/'+'a'*32+'/predictions')
        self.assertNotIn('/predictions',r['answer'])
        self.assertIn('60',r['answer']);self.assertIn('不能证明因果',r['answer'])

    def test_association_removes_model_invented_prediction_link(self):
        url='/ml/runs/'+'a'*32+'/predictions'
        out={'scope':'training_association_only','records_predicted':0,'association_evidence':{'observations':60,'features':[]}}
        r=present_answer({'decision':'approve','answer':'请下载：'+url},'解释关系',out)
        self.assertNotIn('/predictions',r['answer'])

    def test_missing_data_verdict_is_never_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.csv';load_iris(as_frame=True).frame.to_csv(p,index=False)
            c=clients(confidence=.2);col='sepal length (cm)'
            c['high'].replies=[{'decision':'missing_data','reason':'缺少干预数据','questions':['提供额外数据'],'association_columns':[col]}]
            report=MLResearch(ResearchSession(p,'target'),c).run('Can '+col+' establish causality?')
            self.assertEqual(report['research_status'],'needs_clarification')
            self.assertIsNone(report['research_outputs'])

    def test_short_answer_preserves_constraints_or_marks_unmet(self):
        out={'scope':'validation_rows_only','records_predicted':100,'calibration_performed':False}
        r=present_answer({'decision':'approve','answer':'这是结论。'*200},'最多100字',out,'/ml/runs/'+'a'*32+'/predictions')
        self.assertLessEqual(len(r['answer']),100)
        self.assertEqual(r['decision'],'revise')
        self.assertIn('character_budget_too_small',r['unmet_requirements'])

    def test_known_field_labels_keep_english_for_technical_audience(self):
        r=present_answer({'answer':'检查 total_day_minutes 的影响。'},'解释字段',{})
        self.assertIn('日间通话分钟数（total_day_minutes）',r['answer'])

    def test_relevant_explicit_association_can_recover_from_clarify(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.csv';load_iris(as_frame=True).frame.to_csv(p,index=False)
            session=ResearchSession(p,'target');col=str(session.x.columns[0]);c=clients(confidence=.2)
            c['high'].replies=[{'decision':'clarify','reason':'不能证明因果','questions':['是否需要关联分析？']}]
            report=MLResearch(session,c).run('Can '+col+' establish causality?',True)
            self.assertEqual(report['research_status'],'partial_completed')
            self.assertFalse(report['test_evaluated'])
            self.assertEqual(report['research_outputs']['records_predicted'],0)
            self.assertTrue(any(e['event']=='clarification_replaced_by_supported_analysis' for e in report['events']))

    def test_unrelated_columns_never_force_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.csv';load_iris(as_frame=True).frame.to_csv(p,index=False)
            c=clients(confidence=.2);c['high'].replies=[{'decision':'clarify','reason':'缺少收入字段','questions':['请提供收入数据'],'association_columns':['sepal length (cm)']}]
            report=MLResearch(ResearchSession(p,'target'),c).run('Does income cause unemployment?')
            self.assertEqual(report['research_status'],'needs_clarification')
            self.assertIsNone(report['research_outputs']);self.assertEqual(c['reviewer'].calls,0)

if __name__=='__main__':unittest.main()
