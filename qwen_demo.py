"""Original Qwen placement demo: actual handoffs, isolated memory and evidence."""
import csv,json,os,sqlite3,time,uuid,urllib.request
from contextlib import closing
from pathlib import Path
from agents import Scientist
from capability_memory import CapabilityMemory
from executor import ResearchSession
from research_loop import DEFAULT_ALLOWED
from research_deliverables import deliver
from ml_engine import TierBudget,_ClientSwitch,safe_error_code
from model_budget import ModelLimits,ModelBudgetExceeded

MODEL='qwen2.5:3b-instruct-q4_K_M'
DATASETS=['g1_easy','g2_mild','g3_medium','g4_hard','g5_harder','g6_extreme']

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,path)

class Qwen:
    model=MODEL
    base_url='http://127.0.0.1:11434/v1'
    max_output_tokens=400
    def __init__(self):self.last_usage={};self.last_billing_usage={}
    def __call__(self,messages,timeout=120):
        self.last_usage={};self.last_billing_usage={}
        body={'model':self.model,'temperature':0,'messages':messages,'max_tokens':self.max_output_tokens}
        request=urllib.request.Request(self.base_url+'/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request,timeout=timeout) as response:data=json.load(response)
        self.last_usage=data.get('usage') or {}
        return data['choices'][0]['message']['content']

class DemoStore:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.memory=self.root/'capability.sqlite3';CapabilityMemory(self.memory)
    def report(self):
        p=self.root/'report.json'
        return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {'steps':[],'summary':{},'state':'idle'}
    def snapshot(self):
        folder=self.root/'snapshots'/uuid.uuid4().hex;folder.mkdir(parents=True)
        with closing(sqlite3.connect(self.memory)) as a,closing(sqlite3.connect(folder/'memory.sqlite3')) as b:a.backup(b)
        save(folder/'report.json',self.report());save(self.root/'latest.json',{'snapshot':folder.name});return folder.name
    def reset(self):
        with closing(sqlite3.connect(self.memory)) as c,c:c.execute('DELETE FROM attempts')
    def rollback(self):
        pointer=self.root/'latest.json'
        if not pointer.exists():raise ValueError('尚无可回退的演示快照。')
        identity=json.loads(pointer.read_text())['snapshot']
        if len(identity)!=32 or any(c not in '0123456789abcdef' for c in identity):raise ValueError('Invalid snapshot')
        folder=self.root/'snapshots'/identity
        with closing(sqlite3.connect(folder/'memory.sqlite3')) as a,closing(sqlite3.connect(self.memory)) as b:a.backup(b)
        save(self.root/'report.json',json.loads((folder/'report.json').read_text(encoding='utf-8')))
        return {'restored':'已恢复演示经验和对应图表，正式研究未改动。'}
    def interrupted(self,reason):
        report=self.report()
        if report.get('state')=='running':report.update(state='partial',reason=reason);save(self.root/'report.json',report)
        return report

def tasks_for(source,mode,folder):
    if mode=='mixed':
        rows=[]
        for name in DATASETS:
            path=source/'fixtures/graded'/(name+'.csv')
            if not path.exists():path=source/'data/gen/graded'/(name+'.csv')
            if not path.exists():raise ValueError('Missing original graded fixture: '+name)
            rows.append((name,path,'label'))
        return rows
    from local_catalog import LocalCatalog
    catalog=LocalCatalog();rows=[]
    for name in ['iris','wine','breast_cancer','ops_churn','customer_churn','sme_credit','sensor_fault']:
        try:raw,target,_=catalog.download(name)
        except Exception:continue
        path=folder/(name+'.csv');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(raw,encoding='utf-8');rows.append((name,path,target))
    if not rows:raise ValueError('No available original catalog data')
    return rows

def run(store,source,*,rounds=3,mode='mixed',reset=True,low=None,high=None,high_factory=None,limits=None,price_feed=None):
    if mode not in ('mixed','clean','faulty') or not 1<=rounds<=3:raise ValueError('Invalid demo parameters')
    snapshot=store.snapshot()
    if reset:store.reset()
    run_id=uuid.uuid4().hex;folder=store.root/'runs'/run_id;folder.mkdir(parents=True)
    tasks=tasks_for(Path(source),mode,folder/'data');memory=CapabilityMemory(store.memory)
    low=low or Qwen();clients={'low':low}
    if high is not None:clients['high']=high
    gateway=TierBudget(_ClientSwitch(clients),limits or ModelLimits(36,60000,300),price_feed=price_feed)
    report={'run_id':run_id,'model':getattr(low,'model',MODEL),'rounds':rounds,'mode_label':{'mixed':'原版混合难度 · 真实高站接手','clean':'原版内置数据','faulty':'原版截断故障注入'}[mode],
        'snapshot':snapshot,'quality_gate':.6,'steps':[],'faulty':mode=='faulty','notes':['保留原版千问3B及原始六组数据；混合难度为合成评测数据。','通过必须有真实训练且验证宏F1≥0.6；转交本身不算成功。','本地API费用为零，不含设备电力；高级模型可能产生费用，官方估算非账单实扣。','经验与图表回退仅作用于演示；最终测试集未打开。']}
    def persist(state='running'):
        report['state']=state;usage=gateway.snapshot();report['usage']=usage;s=report['steps']
        report['summary']={'tasks':len(tasks)*rounds,'executed':len(s),'passed':sum(x['ok'] for x in s),'completed':sum(x.get('delivered',False) for x in s),
            'failed':sum(not x['ok'] for x in s),'unexecuted':len(tasks)*rounds-len(s),'delegated':sum(x.get('high_attempted',False) for x in s),
            'wasted_local_calls':sum(x['attempted'] and not x['low_ok'] for x in s),'tokens':usage['known_tokens'],'unknown_usage_calls':usage['unreconciled_calls'],
            'official_usd':usage['costs']['calculated_total_usd'],'final_evidenced_ratio':s[-1]['evidenced_ratio'] if s else 0}
        save(store.root/'report.json',report);save(folder/'report.json',report)
    persist();evidenced=0;stopped=False
    for round_no in range(1,rounds+1):
        for name,path,target in tasks:
            if gateway.snapshot()['calls']>=gateway.limits.max_calls or gateway.snapshot()['elapsed_seconds']>=gateway.limits.max_seconds:stopped=True;break
            began=time.monotonic();before=len(gateway.records);session=ResearchSession(path,target);session.ensure_ready();profile=session.planning_context()
            verdict=memory.route(report['model'],'model_search',profile)
            step={'step':len(report['steps'])+1,'round':round_no,'dataset':name,'decision':verdict['reason'],'bucket':verdict.get('bucket'),'tier':verdict['tier'],
                'attempted':False,'high_attempted':False,'low_ok':False,'ok':False,'score':None,'error':None,'delivered':False}
            best=None;history=[]
            for tier in (['low','high'] if verdict['tier']=='low' else ['high']):
                if tier=='high' and step['low_ok']:break
                if tier=='high' and 'high' not in clients:
                    try:clients['high']=high_factory()
                    except Exception:step['error']='high_model_not_configured';break
                client=clients[tier]
                if tier=='high':report['high_model']=getattr(client,'model','configured')
                old_cap=getattr(client,'max_output_tokens',None)
                if tier=='low' and mode=='faulty' and round_no<=2:client.max_output_tokens=60
                call_before=len(gateway.records)
                try:
                    proposal=Scientist(gateway.for_role(tier),DEFAULT_ALLOWED).propose(history,profile,feedback='Require validation macro F1 >= 0.6. Prior failure: '+str(step['error']),timeout=60)
                    result=session.run(proposal['model'],proposal.get('parameters') or {})
                    score=result['validation_f1'];ok=score>=.6
                    step[tier+'_score']=score
                    if tier=='low':step['low_ok']=ok
                    memory.record(getattr(client,'model','configured'),'model_search',ok,None if ok else 'below_quality_gate',profile)
                    history.append({'model':result['model'],'parameters':result['parameters'],'score':score})
                    if best is None or score>best['validation_f1']:best=result
                    if not ok:step['error']='below_quality_gate'
                except ModelBudgetExceeded:step['error']='budget_exhausted';stopped=True;break
                except Exception as exc:
                    step['error']=safe_error_code(exc)
                    if isinstance(exc,ValueError):memory.record(getattr(client,'model','configured'),'model_search',False,'invalid_proposal',profile)
                finally:
                    step['attempted' if tier=='low' else 'high_attempted']=len(gateway.records)>call_before
                    if old_cap is not None:client.max_output_tokens=old_cap
            if best is not None:
                step['score']=best['validation_f1'];step['ok']=step['score']>=.6;step['delivered']=True
                output=deliver(session,best,'Original Qwen demo validation predictions')
                directory=folder/('step-'+str(step['step']));directory.mkdir()
                with (directory/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as stream:
                    writer=csv.DictWriter(stream,fieldnames=['source_row','actual_label','predicted_label']);writer.writeheader();writer.writerows({k:r[k] for k in writer.fieldnames} for r in output['predictions'])
                import hashlib
                save(directory/'evidence.json',{'dataset_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'best':best,'experiments':history,'test_evaluated':False})
            records=gateway.records[before:];step['tokens']=sum(sum(r['usage'].values()) for r in records if r['usage'])
            step['seconds']=round(time.monotonic()-began,2);step['cumulative_tokens']=gateway.snapshot()['known_tokens']
            evidenced+=verdict['reason']!='unproven_explore';step['evidenced_ratio']=round(100*evidenced/(len(report['steps'])+1),1)
            step['status']='通过' if step['ok'] else '未达标' if step['delivered'] else '未完成'
            report['steps'].append(step);persist()
            print(json.dumps({k:step[k] for k in ['step','round','dataset','decision','status','score','tokens']},ensure_ascii=True),flush=True)
            if stopped:break
        if stopped:break
    persist('done' if len(report['steps'])==len(tasks)*rounds and all(s['ok'] for s in report['steps']) else 'partial' if any(s['delivered'] for s in report['steps']) else 'failed')
    return report

def cli(default_mode='mixed'):
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--reset',action='store_true');p.add_argument('--rounds',type=int,default=3);p.add_argument('--mode',choices=['mixed','clean','faulty'],default=default_mode);p.add_argument('--faulty',action='store_true');a=p.parse_args()
    source=Path(__file__).parent
    def high_factory():
        from credential_store import CredentialStore
        from ml_engine import configured_clients
        env=dict(os.environ);env.update(CredentialStore(source/'artifacts/ml/credentials.dpapi').load());return configured_clients(env)['high']
    r=run(DemoStore(source/'artifacts/ml/demo'),source,mode='faulty' if a.faulty else a.mode,rounds=a.rounds,reset=a.reset,high_factory=high_factory)
    print(json.dumps(r['summary'],ensure_ascii=True))
