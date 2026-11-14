"""Paid targeted regression, isolated reports, existing encrypted credentials.

Exercises actual providers through the ASGI API, not a browser/TCP web server.
Never prints credentials. No automatic retries. Default twelve previously failing
or misleading questions, with explicit parent research (not a global latest run).
"""
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from credential_store import CredentialStore
from ml_engine import configured_clients
from ml_api import create_app
from fastapi.testclient import TestClient
from tools.asgi_test_runtime import asgi_test_runtime


def main():
    original=ROOT/'artifacts/ml'
    out=ROOT/'artifacts'/('conversation-acceptance-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()
    keys=CredentialStore(original/'credentials.dpapi').load()
    env=dict(os.environ);env.update(keys)
    for role in ('SOURCE','LOW','HIGH','REVIEWER'):
        env['ML_'+role+'_API_KEY']=keys['DEEPSEEK_API_KEY']
    def factory():return configured_clients(env)
    if (original/'official-prices.json').exists():shutil.copy2(original/'official-prices.json',out/'official-prices.json')
    parents={}
    for kind,prefix in [('prediction','44a7fa9b'),('experience','b629d918')]:
        found=next(original.glob(prefix+'*'),None)
        if found is None and kind=='experience':found=next(original.glob('b629d*'),None)
        if found is None:raise RuntimeError('Required acceptance parent missing: '+kind)
        dest=out/found.name;dest.mkdir()
        for name in ('data.csv','report.json','protocol.json','request.json','state.json','predictions.csv'):
            if (found/name).exists():shutil.copy2(found/name,dest/name)
        parents[kind]=found.name
    questions=[]
    for name in ('qa-1-10.json','qa-11-18.json','qa-19-30.json'):
        questions+=json.loads((ROOT/'artifacts/acceptance-20261114'/name).read_text(encoding='utf-8'))
    chosen={9,10,14,15,17,18,23,24,27,28,29,30}
    rows=[]
    with asgi_test_runtime(),TestClient(create_app(out,client_factory=factory)) as client:
        for item in questions:
            q=item['id']
            if q not in chosen:continue
            body={'goal':item['question'],'max_seconds':180,'max_calls':12,'max_tokens':50000,'max_fits':3}
            if q not in (23,24):body['previous_run_id']=parents['experience' if q in (14,15,18) else 'prediction']
            response=client.post('/ml/runs',json=body)
            if response.status_code!=202:
                row={'q':q,'http_status':response.status_code,'detail':response.text[:300]}
            else:
                run=response.json()['run_id'];state=client.get('/ml/runs/'+run).json()
                report=state.get('report') or {};outputs=report.get('research_outputs') or {};review=report.get('review') or {}
                events=[e['event'] for e in state.get('events',[])]
                checks={'no_execution_failure':state['status'] not in ('failed','blocked'),
                        'state_consistent':not (state['status']=='completed' and report.get('research_status')=='needs_revision')}
                if q not in (15,17,23,24):
                    checks['explain_without_search_or_training']=report.get('training_fits')==0 and 'source_collection_started' not in events
                if q in (23,24):checks['no_unrelated_predictions']=state['status']=='needs_input' and not outputs
                if q==17:checks['changed_target']=report.get('protocol',{}).get('target')=='international_plan'
                if q in (10,30):checks['format']=all(h+'：' in review.get('answer','') for h in (['发现了什么','证据是什么','有哪些局限'] if q==10 else ['直接回答','证据','预测结果','局限']))
                if q==29:
                    checks['audience_and_length']=len(review.get('answer',''))<=500 and 'F1' not in review.get('answer','') and 'total_day_minutes' not in review.get('answer','')
                link=outputs.get('prediction_download_url')
                if link:checks['real_csv_download']=client.get(link).status_code==200
                usage=report.get('usage') or state.get('partial',{}).get('usage') or {}
                row={'q':q,'question':item['question'],'parent':body.get('previous_run_id'),'run_id':run,
                     'status':state['status'],'research_status':report.get('research_status'),'answer':review.get('answer'),
                     'checks':checks,'all_checks_passed':all(checks.values()),'calls':usage.get('calls'),
                     'tokens':usage.get('known_tokens'),'cost_usd':usage.get('costs',{}).get('calculated_total_usd'),
                     'training_fits':report.get('training_fits'),'reason':state.get('reason')}
            rows.append(row)
            (out/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({k:row.get(k) for k in ('q','status','run_id','all_checks_passed','cost_usd')},ensure_ascii=True),flush=True)
    summary={'questions':len(rows),'checks_passed':sum(r.get('all_checks_passed',False) for r in rows),
             'calculated_usd':sum(r.get('cost_usd') or 0 for r in rows),
             'limitations':['Targeted twelve, not a rerun of all thirty.','Real provider calls via in-process ASGI; browser/TCP acceptance is separate.','Official-price calculation, not invoice reconciliation.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('SAVED '+str(out),flush=True)


if __name__=='__main__':main()
