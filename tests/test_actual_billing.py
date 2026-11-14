import unittest
from model_costs import cost_breakdown

class ActualBillingTests(unittest.TestCase):
    def test_same_provider_key_is_one_account_across_base_paths(self):
        from light_model import LightModelClient
        a=LightModelClient('fixture-key',base_url='https://api.deepseek.com/v1')
        b=LightModelClient('fixture-key',base_url='https://api.deepseek.com')
        self.assertEqual(a.billing_account_identity,b.billing_account_identity)

    def test_account_delta_is_not_task_actual_charge(self):
        from billing_actual import account_observation
        result=account_observation([{'account':'account_1','currency':'CNY','balance':'12.0000'}],
                                   [{'account':'account_1','currency':'CNY','balance':'11.9983'}])
        self.assertEqual(result['observations'][0]['decrease'],'0.0017')
        self.assertIsNone(result['task_actual_charge'])
        self.assertEqual(result['attribution'],'account_window_not_task_charge')

    def test_recharge_or_currency_change_is_not_a_saving(self):
        from billing_actual import account_observation
        result=account_observation([{'account':'a','currency':'CNY','balance':'1'}],[{'account':'a','currency':'CNY','balance':'3'}])
        self.assertEqual(result['observations'][0]['decrease'],'-2')
        result=account_observation([{'account':'a','currency':'CNY','balance':'1'}],[{'account':'a','currency':'USD','balance':'1'}])
        self.assertEqual(result['observations'],[])

    def test_unknown_spend_does_not_become_zero(self):
        from billing_actual import account_observation
        self.assertIsNone(account_observation([],[])['task_actual_charge'])
        self.assertIsNone(cost_breakdown([])['invoice_actual_usd'])

    def test_same_account_is_read_once_per_snapshot(self):
        from billing_actual import BalanceMeter
        class Client:
            billing_account_identity='shared'
            calls=0
            def balance_snapshot(self,timeout=3):
                self.calls+=1;return [{'currency':'CNY','balance':'1'}]
        c=Client();meter=BalanceMeter({'low':c,'high':c,'reviewer':c});meter.start();meter.finish()
        self.assertEqual(c.calls,2)

    def test_ui_order_matches_research_workflow(self):
        from html.parser import HTMLParser
        from pathlib import Path
        class Elements(HTMLParser):
            def __init__(self):super().__init__();self.ids=[]
            def handle_starttag(self,tag,attrs):
                value=dict(attrs).get('id')
                if value:self.ids.append(value)
        p=Elements();p.feed(Path('static/ml-app.html').read_text(encoding='utf-8'))
        self.assertLess(p.ids.index('form'),p.ids.index('flow-state'))
        self.assertLess(p.ids.index('flow-state'),p.ids.index('research-answer'))
        self.assertEqual(len(p.ids),len(set(p.ids)))

    def test_every_literal_javascript_id_exists(self):
        from pathlib import Path
        import re
        markup=Path('static/ml-app.html').read_text(encoding='utf-8')
        js=Path('static/ml-app.js').read_text(encoding='utf-8')
        ids=set(re.findall(r'id="([^"]+)"',markup))
        needed=set(re.findall(r"\$\('([^']+)'\)",js))
        self.assertEqual(needed-ids,set())

    def test_completed_api_report_contains_observation_not_claimed_task_charge(self):
        import tempfile
        from fastapi.testclient import TestClient
        from sklearn.datasets import load_iris
        from tools.asgi_test_runtime import asgi_test_runtime
        from tests.test_ml_pipeline import clients
        from ml_api import create_app
        with tempfile.TemporaryDirectory() as tmp,asgi_test_runtime():
            c=clients();balances=iter(('10.0000','9.9980'))
            c['low'].billing_account_identity='fixture-account'
            c['low'].balance_snapshot=lambda timeout:[{'currency':'CNY','balance':next(balances)}]
            with TestClient(create_app(tmp,client_factory=lambda:c)) as api:
                run=api.post('/ml/runs',json={'goal':'Compare iris classifiers','csv_text':load_iris(as_frame=True).frame.to_csv(index=False),'target':'target'}).json()['run_id']
                result=api.get('/ml/runs/'+run).json()
                self.assertEqual(result['status'],'completed')
                bill=result['report']['account_billing']
                self.assertEqual(bill['observations'][0]['decrease'],'0.0020')
                self.assertIsNone(bill['task_actual_charge'])

if __name__=='__main__':unittest.main()
