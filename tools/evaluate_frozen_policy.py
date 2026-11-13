"""Frozen-policy transfer evaluation. No policy tuning, no cloud calls."""
import argparse,hashlib,json,subprocess,sys,time
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from executor import ResearchSession
from orchestrator import ResearchOrchestrator,LocalQwenClient
from task_store import runtime_versions
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine,make_classification
from urllib.request import build_opener,ProxyHandler

BUDGET={'max_experiments':3,'max_calls':2,'max_seconds':60}
FIXED=[('logistic_regression',{}),('random_forest',{'n_estimators':100,'class_weight':'balanced'}),('logistic_regression',{'C':.1,'class_weight':'balanced'})]
SEEDS=[101,202,303]

def save(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
 out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 sources={name:digest(ROOT/name) for name in ['executor.py','orchestrator.py','task_store.py']}
 with build_opener(ProxyHandler({})).open('http://127.0.0.1:11434/api/tags',timeout=5) as response: tags=json.load(response)
 protocol={'created_at':datetime.now().astimezone().isoformat(),'policy':'context_v2','commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':sources,'versions':runtime_versions(),'seeds':SEEDS,'budget':BUDGET,'model':'qwen2.5:3b-instruct-q4_K_M','model_tags':tags,'generation_seed':8675309,'tasks':['wine','synthetic_imbalance','synthetic_mixed'],'fixed_proposals':FIXED,'test_access':'after ALL validation winners selected, once per arm/task/split; claims written before evaluation','limits':['first transfer evaluation, small sample','wine is real built-in data; other2 are synthetic, not enterprise evidence','repeated splits overlap: not independent test samples','same caps not equal actual use; time is soft launch limit','no policy changes after protocol write; no causal or population significance claim']}
 save(out/'protocol.json',protocol)
 wine=load_wine(as_frame=True).frame.rename(columns={'target':'target'});wine.to_csv(out/'wine.csv',index=False)
 for name,weights,sep,noise in [('synthetic_imbalance',[.93,.07],.8,.02),('synthetic_mixed',[.6,.4],1.2,.03)]:
  x,y=make_classification(n_samples=1600,n_features=12,n_informative=7,n_redundant=2,weights=weights,class_sep=sep,flip_y=noise,random_state=8675309)
  frame=pd.DataFrame(x,columns=[f'x{i}' for i in range(12)])
  if name=='synthetic_mixed':
   frame['category']=pd.cut(frame.pop('x0'),bins=[-np.inf,-.5,.5,np.inf],labels=['low','mid','high']).astype(str)
   rng=np.random.default_rng(8675309)
   for column in frame.select_dtypes(include='number'):frame.loc[rng.random(len(frame))<.1,column]=np.nan
  frame['target']=y;frame.to_csv(out/(name+'.csv'),index=False)
 records=[];pending=[]
 for task in protocol['tasks']:
  for seed in SEEDS:
   arms=['fixed_search','qwen_3b'] if seed%2 else ['qwen_3b','fixed_search']
   for arm in arms:
    session=ResearchSession(out/(task+'.csv'),'target',seed=seed,metric='f1_minority' if task=='synthetic_imbalance' else 'f1_macro',pos_label='1')
    started=time.monotonic()
    if arm=='qwen_3b':report=ResearchOrchestrator(session,LocalQwenClient('http://127.0.0.1:11434',protocol['model']),**BUDGET).run()
    else:
     experiments=[]
     for model,parameters in FIXED:
      if time.monotonic()-started>=BUDGET['max_seconds']:break
      experiments.append(session.run(model,parameters))
     report={'experiments':experiments,'model_calls':0,'cloud_calls':0,'jev_calls':0,'reported_local_tokens':{'prompt':0,'completion':0}}
    assert not session.closed and report['model_calls']<=2 and len(report['experiments'])<=3
    assert not report['cloud_calls'] and not report['jev_calls']
    best=max(report['experiments'],key=lambda x:x['validation_score'])
    fingerprint=hashlib.sha256(json.dumps([session.train_indices,session.validation_indices,session.test_indices],default=int).encode()).hexdigest()
    row={'task':task,'seed':seed,'arm':arm,'dataset_sha256':session.dataset_sha256,'split_sha256':fingerprint,'metric':session.metric,'validation_score':best['validation_score'],'selected_experiment_id':best['experiment_id'],'elapsed_seconds':time.monotonic()-started,'report':report,'test_evaluated':False}
    records.append(row);pending.append((session,row))
    save(out/f'{task}_{seed}_{arm}.json',row)
    print(json.dumps({k:row[k] for k in ['task','seed','arm','validation_score']}),flush=True)
 for name,expected in sources.items():assert digest(ROOT/name)==expected,'Policy changed during evaluation'
 for task in protocol['tasks']:
  for seed in SEEDS:assert len({r['split_sha256'] for r in records if r['task']==task and r['seed']==seed})==1
 save(out/'validation_selection_locked.json',[{k:r[k] for k in ['task','seed','arm','selected_experiment_id']} for r in records])
 for session,row in pending:
  claim=out/f"{row['task']}_{row['seed']}_{row['arm']}.test_claim"
  with claim.open('x') as f:f.write('claimed before accessing test; do not repeat')
  row['final']=session.finalize(row['selected_experiment_id']);row['test_evaluated']=True
  save(out/f"{row['task']}_{row['seed']}_{row['arm']}.json",row)
 summary=[]
 for task in protocol['tasks']:
  for arm in ['fixed_search','qwen_3b']:
   group=[r for r in records if r['task']==task and r['arm']==arm]
   summary.append({'task':task,'arm':arm,'mean_validation_score':sum(r['validation_score'] for r in group)/3,'mean_test_score':sum(r['final']['test_score'] for r in group)/3,'mean_seconds':sum(r['elapsed_seconds'] for r in group)/3,'local_calls':sum(r['report']['model_calls'] for r in group),'reported_tokens':sum(sum(r['report']['reported_local_tokens'].values()) for r in group)})
 save(out/'summary.json',{'protocol':protocol,'summary':summary});print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
