"""Loopback HTTP smoke: one configured LLM, real sklearn, injected model replies.

No paid API, no real provider credentials, no held-out test evaluation.
"""
import json, sys, tempfile, threading
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from urllib.request import Request,urlopen
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from ml_engine import configured_clients,MLResearch
from executor import ResearchSession
from sklearn.datasets import load_iris

class Model(BaseHTTPRequestHandler):
    count=0
    def log_message(self,*args):pass
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        Model.count+=1
        if Model.count==1:
            content={'model':'random_forest','parameters':{'n_estimators':20,'max_depth':3},'hypothesis':'Tree boundaries may improve validation','expected_direction':'up','falsifier':'No validation improvement'}
        else:
            content={'decision':'approve','answer':'仅依据真实验证结果，未评估测试集。','summary':'Fixture protocol audit, not a real model assessment','experiment_ids':['exp_001']}
        data=json.dumps({'choices':[{'message':{'content':json.dumps(content)}}],'usage':{'prompt_tokens':20,'completion_tokens':10}}).encode()
        self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)

def main():
    with ThreadingHTTPServer(('127.0.0.1',0),Model) as server:
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            env={'DEEPSEEK_API_KEY':'fixture-only-not-real','ML_SINGLE_MODEL':'1','ML_BASE_URL':'http://127.0.0.1:'+str(server.server_port),'ML_HIGH_MODEL':'fixture-single-model'}
            clients=configured_clients(env)
            with tempfile.TemporaryDirectory() as tmp:
                data=Path(tmp)/'iris.csv';load_iris(as_frame=True).frame.to_csv(data,index=False)
                report=MLResearch(ResearchSession(data,'target'),clients).run('Compare iris classifiers')
                assert report['research_status']=='approved'
                assert report['usage']['calls']==Model.count==2
                assert not report['test_evaluated']
                result={'status':report['research_status'],'source':'loopback_model_fixture_not_live_LLM','classifier_source':'rules','llm_models':list({clients[r].model for r in ('source','low','high','reviewer')}),'training_fits':report['training_fits'],'model_http_requests':Model.count,'known_fixture_tokens':report['usage']['known_tokens'],'validation_f1':report['best']['validation_f1'],'test_evaluated':False,'paid_api_calls':0}
                out=Path(__file__).resolve().parents[1]/'artifacts/v20-http-smoke.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
        finally:server.shutdown();thread.join()

if __name__=='__main__':main()
