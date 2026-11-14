"""Live acceptance: real classifier call, injected low confidence, real subsequent APIs."""
import json,sys,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from credential_store import CredentialStore
from ml_engine import MLResearch,configured_clients
from executor import ResearchSession
from research_deliverables import deliver
from local_catalog import LocalCatalog
from research_system import save

def main():
    root=Path(__file__).resolve().parents[1]
    clients=configured_clients(CredentialStore(root/'artifacts/ml/credentials.dpapi').load())
    class ForceUncertainty:
        def __init__(self,inner):
            self.inner=inner;self.model=inner.model;self.max_output_tokens=inner.max_output_tokens;self.last_usage={}
        def __call__(self,messages,timeout):
            self.last_usage={}
            try:
                value=json.loads(self.inner(messages,timeout=timeout));value['confidence']=.1
                return json.dumps(value)
            finally:self.last_usage=dict(self.inner.last_usage)
    clients['classifier']=ForceUncertainty(clients['classifier'])
    path=root/'artifacts/routing-recheck-acceptance'/uuid.uuid4().hex;path.mkdir(parents=True)
    csv,target,_=LocalCatalog().download('customer_churn');(path/'data.csv').write_bytes(csv.encode())
    session=ResearchSession(path/'data.csv',target)
    def emit(e):
        with (path/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(e,ensure_ascii=False)+'\n')
    engine=MLResearch(session,clients,max_calls=12,max_tokens=60000,max_seconds=180,max_fits=3,max_expert_calls=1,deliverer=deliver,emit=emit)
    print('acceptance_id='+path.name,flush=True)
    report=engine.run('请基于这份客户流失数据训练分类预测模型，给出验证样本的预测和研究局限。',False)
    report['acceptance_setup']='Real Jev request, confidence deliberately set to0.1; all subsequent model calls are real.'
    save(path/'report.json',report)
    print(json.dumps({'status':report['research_status'],'events':[e['event'] for e in report['events']],
        'calls':report['usage']['calls'],'known_tokens':report['usage']['known_tokens'],
        'predictions':(report.get('research_outputs') or {}).get('records_predicted'),'test_used':report['test_evaluated']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
