"""Small real paired check; no mocks, key inputs or automatic paid retries."""
import argparse,json,time
from pathlib import Path
from urllib.request import Request,build_opener,ProxyHandler


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='artifacts/token-economy-live');args=parser.parse_args()
    out=Path(args.output);out.mkdir(exist_ok=False)
    opener=build_opener(ProxyHandler({}));base='http://127.0.0.1:8018'
    def api(path,body=None):
        with opener.open(Request(base+path,data=json.dumps(body).encode() if body is not None else None,
                         headers={'Content-Type':'application/json'}),timeout=20) as r:return json.load(r)
    version=api('/ml/version')
    if version['restart_required']:raise RuntimeError('Restart matching source first')
    (out/'protocol.json').write_text(json.dumps({'version':version,'seeds':[42,7],'memory':'off for both','quality_gap_limit':.005,'scope':'Two descriptive validation pairs; not generalization proof or invoice reconciliation.'},indent=2),encoding='utf-8')
    rows=[]
    for seed in [42,7]:
        if api('/ml/version')['source_fingerprint']!=version['source_fingerprint']:raise RuntimeError('Source changed')
        body={'goal':'比较分类方案，给出验证预测结果、适用范围与局限。','dataset_id':'customer_churn','seeds':[seed],
              'economy':True,'max_rounds':3,'patience':2,'max_fits':4,'max_calls':16,'max_tokens':60000,'max_seconds':180}
        cid=api('/ml/comparisons',body)['comparison_id'];print('Started '+cid,flush=True)
        began=time.monotonic()
        while time.monotonic()-began<400:
            state=api('/ml/comparisons/'+cid)
            if state['status'] not in ('queued','running'):break
            time.sleep(1)
        else:raise TimeoutError('Still running; not resubmitted')
        report=state.get('report') or {};arms={a['mode']:a['metrics'] for a in report.get('arms',[])}
        a,b=arms.get('routed',{}),arms.get('high_only',{})
        comparable=all(m.get('status')=='completed' and not m.get('baseline_only') for m in [a,b])
        gap=b.get('validation_macro_f1',0)-a.get('validation_macro_f1',0) if comparable else None
        row={'seed':seed,'comparison_id':cid,'report':report,'quality_gate':comparable and gap<=.005,'macro_f1_gap':gap}
        rows.append(row);(out/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'seed':seed,'quality_gate':row['quality_gate'],'gap':gap,'arms':{k:{**{f:v.get(f) for f in ('known_tokens','wall_seconds','validation_macro_f1','minority_f1','calls')},'usd':(v.get('costs')or{}).get('calculated_total_usd')} for k,v in arms.items()}},ensure_ascii=True),flush=True)


if __name__=='__main__':main()
