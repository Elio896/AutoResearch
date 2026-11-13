"""Exploratory matched-cap comparison; never accesses held-out test scores."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from executor import ResearchSession
from orchestrator import ResearchOrchestrator,LocalQwenClient
from task_store import runtime_versions

BUDGET={'max_experiments':3,'max_calls':2,'max_seconds':60}
FIXED=[('logistic_regression',{}),('random_forest',{'n_estimators':100,'class_weight':'balanced'}),('logistic_regression',{'C':.1,'class_weight':'balanced'})]
ARMS=[('fixed_search',None,None),('qwen_0_5b','http://127.0.0.1:11435','Qwen2.5-0.5B-Instruct:latest'),('qwen_3b','http://127.0.0.1:11434','qwen2.5:3b-instruct-q4_K_M')]

def run_one(task,seed,arm):
    name,base,model=arm
    metric='f1_minority' if task=='t2_imbalanced' else 'f1_macro'
    session=ResearchSession(ROOT/'bench/tasks'/task/'data.csv','target',seed=seed,metric=metric,pos_label='1')
    started=time.monotonic()
    if base:
        report=ResearchOrchestrator(session,LocalQwenClient(base,model),**BUDGET).run()
    else:
        experiments=[]
        for estimator,parameters in FIXED:
            if time.monotonic()-started>=BUDGET['max_seconds']:break
            experiments.append(session.run(estimator,parameters))
        report={'experiments':experiments,'model_calls':0,'reported_local_tokens':{'prompt':0,'completion':0},'cloud_calls':0,'jev_calls':0,'stop_reason':'fixed_plan_complete','status':'completed','events':[]}
    assert not session.closed
    results=report['experiments']
    assert len(results)<=BUDGET['max_experiments'] and report['model_calls']<=BUDGET['max_calls']
    assert not report['cloud_calls'] and not report['jev_calls']
    best=max(results,key=lambda value:value['validation_score'])
    split_fingerprint=hashlib.sha256(json.dumps({'train':session.train_indices,'validation':session.validation_indices,'test':session.test_indices},default=int,sort_keys=True).encode()).hexdigest()
    return {'task':task,'seed':seed,'arm':name,'model':model,'endpoint':base,'metric':metric,'dataset_sha256':session.dataset_sha256,'split_sha256':split_fingerprint,'budget':BUDGET,'elapsed_seconds':time.monotonic()-started,'best_score':best['validation_score'],'best_experiment_id':best['experiment_id'],'training_attempts':len(results),'test_evaluated':False,'report':report}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    destination=Path(args.output)
    destination.mkdir(parents=True,exist_ok=False)
    protocol={'created_at':datetime.now().astimezone().isoformat(),'seeds':[42,43,44],'tasks':['t2_imbalanced','t3_clean'],'arms':ARMS,'budget':BUDGET,'versions':runtime_versions(),'test_evaluated':False,'limits':['exploratory three seeds only','equal upper budgets, not necessarily equal actual use','sequential shared-machine timings include inference and possible model loading; no power measurement','no cloud/Jev empirical comparison without privately configured credentials','not five-fold benchmark protocol; do not compare to benchmark thresholds','t1 intentionally omitted: sentinel and unit cleaning still requires separate design']}
    (destination/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[]
    for task in protocol['tasks']:
        for seed in protocol['seeds']:
            # Rotate order to reduce systematic warmup/order bias.
            offset=(seed-42)%len(ARMS)
            for arm in ARMS[offset:]+ARMS[:offset]:
                row=run_one(task,seed,arm)
                rows.append(row)
                (destination/f'{task}_{seed}_{arm[0]}.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
                print(json.dumps({k:row[k] for k in ('task','seed','arm','best_score','training_attempts','elapsed_seconds')}),flush=True)
    for task in protocol['tasks']:
        for seed in protocol['seeds']:
            group=[row for row in rows if row['task']==task and row['seed']==seed]
            assert len({row['split_sha256'] for row in group})==1
    summary=[]
    for task in protocol['tasks']:
        for arm,_,_ in ARMS:
            group=[row for row in rows if row['task']==task and row['arm']==arm]
            summary.append({'task':task,'arm':arm,'runs':len(group),'mean_validation_score':sum(row['best_score'] for row in group)/len(group),'mean_seconds':sum(row['elapsed_seconds'] for row in group)/len(group),'total_local_calls':sum(row['report']['model_calls'] for row in group),'total_training_attempts':sum(row['training_attempts'] for row in group),'reported_tokens':sum(sum(row['report']['reported_local_tokens'].values()) for row in group)})
    (destination/'summary.json').write_text(json.dumps({'protocol':protocol,'summary':summary},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
