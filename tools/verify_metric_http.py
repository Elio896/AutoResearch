"""Live loopback task-objective verification; no LLM requests."""
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler

opener=build_opener(ProxyHandler({}))
def call(method,path,payload=None):
    request=Request('http://127.0.0.1:8002'+path, method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type':'application/json'})
    try:
        with opener.open(request,timeout=60) as response: return response.status,json.load(response)
    except HTTPError as error:
        with error: return error.code,json.load(error)

def main():
    _,schema=call('GET','/openapi.json')
    assert 'metric' in schema['components']['schemas']['TaskInput']['properties']
    status,created=call('POST','/tasks',{'csv_text':Path('bench/tasks/t2_imbalanced/data.csv').read_text(encoding='utf-8'),'target':'target','metric':'f1_minority','pos_label':'1','max_experiments':2})
    assert status==201,created
    assert not created['risk_report']['blocked'],created['risk_report']
    task=created['task_id']; experiments=[]
    for model,parameters in [('logistic_regression',{}),('random_forest',{'n_estimators':100,'class_weight':'balanced'})]:
        status,result=call('POST',f'/tasks/{task}/experiments',{'model':model,'parameters':parameters})
        assert status==200,result
        assert result['selection_metric']=='f1_minority'
        assert result['validation_score']==result['validation_f1_minority']
        experiments.append(result)
    _,before=call('GET',f'/tasks/{task}')
    assert before['final_result'] is None and before['metric']=='f1_minority'
    winner=max(experiments,key=lambda item:item['validation_score'])
    status,final=call('POST',f'/tasks/{task}/finalize',{})
    assert status==200 and final['experiment_id']==winner['experiment_id'],final
    assert final['test_score']==final['test_f1_minority']
    duplicate,_=call('POST',f'/tasks/{task}/finalize',{})
    assert duplicate==409
    evidence={'task_id':task,'protocol':'stratified 60/20/20 seed=42; not five-fold CV', 'selection_metric':'f1_minority','pos_label':'1','experiments':experiments,'final':final,'duplicate_finalize_http':duplicate,'model_api_calls':0}
    Path('artifacts/metric_http_verification.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(evidence,indent=2))

if __name__=='__main__': main()
