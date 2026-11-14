"""Frozen live acceptance against the actual local HTTP service.

No key input, mocked model replies, silent paid retries or fabricated success.
Each result is saved before the next job. Price totals are not invoice charges.
"""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
import uuid
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import Request,build_opener,ProxyHandler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app_version import APP_VERSION
from local_catalog import LocalCatalog

TERMINAL={'completed','failed','blocked','needs_input','partial_completed','needs_revision'}


class Acceptance:
    def __init__(self,args):
        self.args=args;self.out=Path(args.output).resolve();self.out.mkdir(parents=True,exist_ok=True)
        self.opener=build_opener(ProxyHandler({}));self.known_cost=0.;self.calls=0;self.rows=[];self.matrix_costs=[]
        self.version=self.api('/ml/version')
        if self.version['version']!=APP_VERSION or self.version['restart_required']:raise RuntimeError('Deploy matching backend before acceptance')
        self.fingerprint=self.version['source_fingerprint']
        self.protocol={'frozen_at':datetime.now(timezone.utc).isoformat(),'version':self.version,'models':self.api('/ml/config').get('roles'),
                       'datasets':[{k:r.get(k) for k in ('id','sha256','rows','target')} for r in LocalCatalog().records()],
                       'limits':{'max_rounds':3,'patience':2,'max_fits':4,'max_calls':16,'max_tokens':60000,'max_seconds':180,'max_expert_calls':3},
                       'validation_only':True,'no_test_set_selection':True,'official_calculated_usd_limit':args.cost_limit,
                       'model_attempt_limit':400,'versions':{n:importlib.metadata.version(n) for n in ('scikit-learn','pandas','numpy','fastapi')},
                       'cost_cap_scope':'Between jobs, known official calculated costs only; unknown charges and invoice totals cannot be bounded by this guard.',
                       'evidence_checks_are_not_semantic_correctness_scores':True}
        if (self.out/'protocol.json').exists():raise RuntimeError('Choose a fresh output directory; do not overwrite a frozen evaluation')
        self.write('protocol.json',self.protocol)

    def write(self,name,value):
        path=self.out/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

    def api(self,path,body=None):
        request=Request(self.args.base+path,data=json.dumps(body).encode('utf-8') if body is not None else None,
                        headers={'Content-Type':'application/json'} if body is not None else {})
        with self.opener.open(request,timeout=30) as response:return json.load(response)

    def guard(self):
        if self.known_cost>=self.args.cost_limit or self.calls>=400:raise RuntimeError('Campaign cost/call cap reached; remaining cases not run')
        current=self.api('/ml/version')
        if current['source_fingerprint']!=self.fingerprint or current['restart_required']:raise RuntimeError('Source changed during frozen acceptance; stop rather than combine versions')

    def wait(self,path):
        began=time.monotonic()
        while time.monotonic()-began<360:
            state=self.api(path)
            if state['status'] in TERMINAL:return state
            time.sleep(1)
        raise TimeoutError('Job still active; not submitted again')

    def run(self,case,body):
        self.guard();body={**self.protocol['limits'],**body,'evaluate_test':False}
        began=time.monotonic();started=self.api('/ml/runs',body);rid=started['run_id']
        self.write('jobs/'+rid+'.json',{'case':case,'request':{k:v for k,v in body.items() if k!='csv_text'},'csv_sha256':hashlib.sha256(body['csv_text'].encode()).hexdigest() if 'csv_text' in body else None})
        state=self.wait('/ml/runs/'+rid+'?compact=true');report=state.get('report') or {};usage=report.get('usage') or (state.get('partial')or{}).get('usage') or {}
        outputs=report.get('research_outputs') or {};cost=(usage.get('costs')or{}).get('calculated_total_usd')
        self.known_cost+=cost or 0.;self.calls+=usage.get('calls',0)
        row={'case':case,'run_id':rid,'status':state['status'],'research_status':report.get('research_status'),
             'answer':(report.get('review')or{}).get('answer'),'research_outputs':{k:v for k,v in outputs.items() if k!='predictions'},
             'experience':report.get('experience'),'iteration':report.get('iteration'),'training_fits':report.get('training_fits'),
             'events':[e['event'] for e in state.get('events',[])],'known_tokens':usage.get('known_tokens'),
             'model_calls':usage.get('calls'),'unreconciled_calls':usage.get('unreconciled_calls'),
             'official_calculated_usd':cost,'request_wall_seconds':time.monotonic()-began,
             'validation_macro_f1':(report.get('best')or{}).get('validation_f1'),
             'test_evaluated':report.get('test_evaluated',False),'reason':state.get('reason')}
        link=outputs.get('prediction_download_url')
        if link:
            import csv,io
            with self.opener.open(self.args.base+link,timeout=30) as response:count=sum(1 for _ in csv.DictReader(io.StringIO(response.read().decode('utf-8-sig'))))
            row['download_rows']=count
        self.rows.append(row);self.write('results.json',self.rows)
        print(json.dumps({k:row.get(k) for k in ('case','run_id','status','training_fits','model_calls','official_calculated_usd')},ensure_ascii=True),flush=True)
        return row

    def dataset(self,key):
        csv,target,source=LocalCatalog().download(key)
        return {'csv_text':csv,'target':target}

    def smoke(self):
        return self.run('multiround_smoke',{'goal':'请比较客户流失分类预测方案，依据每轮验证反馈提出不同实验，最后给出预测结果、数据范围和局限。',
            **self.dataset('customer_churn'),'use_memory':False,'memory_namespace':uuid.uuid4().hex,'patience':3})

    def qa(self):
        questions=[]
        for name in ('qa-1-10.json','qa-11-18.json','qa-19-30.json'):
            questions+=json.loads((ROOT/'artifacts/acceptance-20261114'/name).read_text(encoding='utf-8'))
        namespace=uuid.uuid4().hex;last=None;prediction=None;churn_prediction=None;results=[]
        for item in questions:
            q=item['id'];body={'goal':item['question'],'memory_namespace':namespace}
            parent=churn_prediction if q in (9,10,27,29,30) else last
            if q in (1,11,12,19,20,21,22,23,24,25):parent=None
            if parent:body['previous_run_id']=parent
            row=self.run('qa_'+str(q),body);output=row['research_outputs'];answer=row.get('answer') or ''
            checks={'no_runtime_failure':row['status'] not in ('failed','blocked'),
                    'answer_present':bool(answer),'test_not_used_for_selection':not row['test_evaluated']}
            if q in (19,20,21,22,23,24,25):checks['data_gap_no_irrelevant_predictions']=row['status']=='needs_input' and not output.get('records_predicted')
            if q in (2,8,11,12,13,17,29):checks['real_prediction_file']=row.get('download_rows',0)>0
            if q in (9,10,14,18,27,28,30):checks['followup_no_new_training']=row.get('training_fits')==0
            if q==4:checks['outcome_group_comparison']=bool(output.get('target_group_comparison')) and output.get('records_predicted')==0
            if q==16:checks['association_without_predictions']=bool(output.get('association_evidence')) and output.get('records_predicted')==0
            if q==17:checks['changed_existing_target']=output.get('target')=='international_plan'
            if q in (10,30):checks['requested_sections']=all(h in answer for h in (['发现了什么','证据是什么','有哪些局限'] if q==10 else ['直接回答','证据','预测结果','局限']))
            if q==29:checks['operations_audience']=len(answer)<=500 and 'F1' not in answer and 'total_day_minutes' not in answer
            row['evidence_checks']=checks;row['evidence_checks_passed']=all(checks.values());results.append(row)
            if row['status'] in ('completed','partial_completed','needs_revision') and output:
                last=row['run_id']
                if output.get('records_predicted'):
                    prediction=last
                    if output.get('target')=='class':churn_prediction=last
            self.write('qa-results.json',results)
        self.write('qa-summary.json',{'submitted':len(results),'evidence_checks_passed':sum(r['evidence_checks_passed'] for r in results),
                                     'statuses':{s:sum(r['status']==s for r in results) for s in sorted(TERMINAL)},
                                     'not_a_semantic_correctness_score':True,'parent_references_are_explicit':True})

    def matrix(self):
        rows=[]
        for dataset,seeds in [('customer_churn',[42,7,2026]),('iris',[42]),('wine',[42]),('breast_cancer',[42])]:
            for seed in seeds:
                self.guard();body={k:v for k,v in self.protocol['limits'].items() if k not in ('max_expert_calls',)}
                body.update(goal='比较分类方案，并给出验证预测结果、适用范围与局限。',dataset_id=dataset,seeds=[seed])
                started=self.api('/ml/comparisons',body);cid=started['comparison_id'];state=self.wait('/ml/comparisons/'+cid)
                report=state.get('report') or {};costs=[]
                for arm in report.get('arms',[]):
                    m=arm['metrics'];self.calls+=m.get('calls',(m.get('usage')or{}).get('calls',0));cost=(m.get('costs')or{}).get('calculated_total_usd')
                    self.known_cost+=cost or 0.;costs.append(cost);self.matrix_costs.append(cost)
                row={'dataset':dataset,'seed':seed,'comparison_id':cid,'status':state['status'],'report':report}
                rows.append(row);self.write('matrix-results.json',rows)
                print(json.dumps({'case':'matrix','dataset':dataset,'seed':seed,'comparison_id':cid,'costs':costs}),flush=True)

    def memory(self):
        results=[]
        for seed in (42,7,2026):
            namespace=uuid.uuid4().hex
            body={'goal':'请研究客户流失分类预测，尝试改进验证表现，交付预测及局限。',**self.dataset('customer_churn'),'seed':seed}
            acquired=self.run('memory_acquisition_'+str(seed),{**body,'use_memory':False,'memory_namespace':namespace})
            source=ROOT/'artifacts/ml/memory-experiments'/namespace
            if not source.is_dir():raise RuntimeError('Run service from this project default artifacts/ml root for isolated memory evaluation')
            forks={mode:uuid.uuid4().hex for mode in ('no_memory','persistent_memory')}
            for mode,identity in forks.items():shutil.copytree(source,source.parent/identity)
            arms={}
            for mode in ('no_memory','persistent_memory'):
                arms[mode]=self.run('memory_'+mode+'_'+str(seed),{**body,'memory_namespace':forks[mode],'use_memory':mode=='persistent_memory'})
            row={'seed':seed,'common_acquisition':acquired,'arms':arms,
                 'acquisition_cost_is_shared_overhead':True,'not_model_weight_learning':True,
                 'limitation':'Validation history reuse is contextual adaptation, not out-of-sample generalization proof.'}
            results.append(row);self.write('memory-results.json',results)

    def summary(self):
        self.write('campaign-summary.json',{'jobs_submitted':len(self.rows),'known_model_attempts':self.calls,
            'known_official_calculated_usd':self.known_cost,'invoice_actual_usd':None,
            'all_costs_complete':all(r.get('official_calculated_usd') is not None for r in self.rows) and all(c is not None for c in self.matrix_costs),
            'source_fingerprint':self.fingerprint,'automatic_policy_promotion':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8018');p.add_argument('--output',required=True)
    p.add_argument('--phase',choices=['smoke','qa','matrix','memory','all'],default='all');p.add_argument('--cost-limit',type=float,default=.20);args=p.parse_args()
    if not args.base.startswith('http://127.0.0.1:'):raise ValueError('Acceptance supports the local loopback service only')
    run=Acceptance(args)
    try:
        for phase in (['smoke','qa','matrix','memory'] if args.phase=='all' else [args.phase]):getattr(run,phase)()
    finally:run.summary()
    print('SAVED '+str(run.out),flush=True)


if __name__=='__main__':main()
