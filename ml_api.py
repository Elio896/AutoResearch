"""The reduced product surface: tabular ML, API routing and evidence."""
import sys
import time
import json
import threading
import uuid
from pathlib import Path
import hashlib
from app_version import APP_VERSION

def source_fingerprint():
    base=Path(__file__).parent
    files=['app_version.py','ml_api.py','ml_engine.py','light_model.py','model_budget.py','failure_recovery.py','billing_actual.py','answer_evidence.py','research_context.py','comparison_benchmark.py','data_source.py','local_catalog.py','association_analysis.py','research_deliverables.py','strategy_store.py','experiment_lessons.py']
    return hashlib.sha256(b''.join(f.encode()+b'\0'+(base/f).read_bytes()+b'\0' for f in files)).hexdigest()

LOADED_SOURCE_FINGERPRINT=source_fingerprint()


def asset_fingerprint():
    """Static files are served live, so their edits do not require a restart."""
    base=Path(__file__).parent
    files=['static/ml-app.js','static/ml-app.html','static/ml-app.css','static/pixel-theme.css']
    return hashlib.sha256(b''.join(f.encode()+b'\0'+(base/f).read_bytes()+b'\0' for f in files)).hexdigest()


from fastapi import Body,  FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from pydantic import BaseModel, Field, ConfigDict, StrictInt

from executor import ResearchSession
from ml_engine import MLResearch, configured_clients, CAPABILITIES, TierBudget, _ClientSwitch
from model_budget import ModelLimits
from billing_actual import BalanceMeter
from failure_recovery import failure_kind
from strategy_store import StrategyStore
from research_deliverables import ExperienceBank, deliver
from experiment_lessons import LessonStore
from credential_store import CredentialStore, CredentialStoreError
from price_feed import PriceFeed
from data_source import collect_dataset, DataSourceError
from local_catalog import LocalCatalog
from research_system import save, read_saved_state
from comparison_benchmark import run_comparison
from capability_memory import CapabilityMemory
from research_context import (ANSWER_RULES, answer_contract, evidence_summary,
    is_followup, requires_experiment, training_forbidden, needs_new_analysis, needs_prediction_execution, scope_gap, target_from_question, research_target, present_answer)


class MLInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    csv_text:str|None=Field(default=None,min_length=1,max_length=2_000_000)
    target:str|None=Field(default=None,min_length=1,max_length=200)
    goal:str=Field(default='Improve classification validation macro F1 within budget.',min_length=3,max_length=1000)
    max_calls:int=Field(default=16,ge=3,le=40,strict=True)
    max_tokens:int=Field(default=60000,ge=1000,le=200000,strict=True)
    max_seconds:int=Field(default=180,ge=10,le=600,strict=True)
    max_fits:int=Field(default=5,ge=1,le=10,strict=True)
    max_expert_calls:int=Field(default=1,ge=0,le=3,strict=True)
    evaluate_test:bool=False
    previous_run_id:str|None=Field(default=None,pattern=r'^[a-f0-9]{32}$')
    max_rounds:int=Field(default=1,ge=1,le=8,strict=True)
    patience:int=Field(default=2,ge=1,le=5,strict=True)
    seed:int=Field(default=42,ge=0,le=2147483647,strict=True)
    use_memory:bool=True
    economy:bool=False
    memory_namespace:str|None=Field(default=None,pattern=r'^[a-f0-9]{32}$')


class ComparisonInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    goal:str=Field(min_length=3,max_length=1000)
    dataset_id:str=Field(default='customer_churn',min_length=1,max_length=80)
    csv_text:str|None=Field(default=None,min_length=1,max_length=2_000_000)
    target:str|None=Field(default=None,min_length=1,max_length=200)
    seeds:list[StrictInt]=Field(default_factory=lambda:[42],min_length=1,max_length=3)
    max_calls:int=Field(default=12,ge=3,le=40,strict=True)
    max_tokens:int=Field(default=60000,ge=4000,le=200000,strict=True)
    max_seconds:int=Field(default=180,ge=10,le=600,strict=True)
    max_fits:int=Field(default=3,ge=2,le=10,strict=True)
    max_rounds:int=Field(default=1,ge=1,le=8,strict=True)
    patience:int=Field(default=2,ge=1,le=5,strict=True)
    economy:bool=True


def create_app(data_directory=None,client_factory=None,catalog_factory=None):
    root=Path(data_directory or Path(__file__).parent/'artifacts/ml').resolve()
    root.mkdir(parents=True,exist_ok=True)
    static=Path(__file__).parent/'static'
    price_feed=PriceFeed(root/'official-prices.json',auto_refresh=client_factory is None)
    credential_store=CredentialStore(root/'credentials.dpapi')
    credential_error=None
    try: runtime_keys=credential_store.load()
    except CredentialStoreError as exc:
        runtime_keys={}
        credential_error=str(exc)
    key_lock=threading.Lock()
    def runtime_clients():
        import os
        with key_lock:
            env=dict(os.environ)
            env.update(runtime_keys)
            if runtime_keys.get('DEEPSEEK_API_KEY'):
                for name in ('ML_API_KEY','ML_SOURCE_API_KEY','ML_LOW_API_KEY','ML_HIGH_API_KEY','ML_REVIEWER_API_KEY'):
                    env[name]=runtime_keys['DEEPSEEK_API_KEY']
        return configured_clients(env)
    factory=client_factory or runtime_clients
    strategies=StrategyStore(root/'ml-strategies.json')
    strategies.bootstrap({'planning_prompt':'Use validation evidence and bounded diverse proposals; do not repeat failed configurations.'})

    class Credentials(BaseModel):
        model_config=ConfigDict(extra='forbid')
        deepseek:str|None=Field(default=None,repr=False)
        jev:str|None=Field(default=None,repr=False)
    app=FastAPI(title='EvoResearch ML',version='2.0',description='API-only ML research; classifier routing and bounded escalation.')
    @app.exception_handler(RequestValidationError)
    async def safe_validation(request,exc):
        if request.url.path=='/ml/credentials':
            return JSONResponse(status_code=400,content={'detail':'密钥设置格式无效，原配置未修改。'})
        return await request_validation_exception_handler(request,exc)

    @app.middleware('http')
    async def no_cache(request,call_next):
        response=await call_next(request)
        response.headers['Cache-Control']='no-store'
        response.headers['X-EvoResearch-Version']=APP_VERSION
        return response


    # --- evolution demo (local-only, zero API cost) ---
    import subprocess as _sp
    _evo = {'proc': None, 'state': 'idle', 'started': None, 'log': ''}

    def _evo_script():
        return Path(__file__).parent / 'artifacts' / 'acceptance-20261114' / 'evolution_demo.py'

    def _evo_python():
        return sys.executable

    @app.get('/ml/evolution/status')
    def evolution_status():
        proc = _evo['proc']
        if proc is not None and proc.poll() is not None:
            _evo['state'] = 'done' if proc.returncode == 0 else 'failed'
            _evo['proc'] = None
        elapsed = ''
        if _evo['state'] == 'running' and _evo['started']:
            elapsed = '%ds' % int(time.time() - _evo['started'])
        return {'state': _evo['state'], 'elapsed_hint': elapsed,
                'log_tail': (_evo['log'] or '')[-400:]}

    @app.post('/ml/evolution/run')
    def evolution_run(body: dict = Body(default={})):
        if _evo['proc'] is not None and _evo['proc'].poll() is None:
            raise HTTPException(409, 'A demo is already running.')
        mode = (body or {}).get('mode', 'faulty')
        if mode not in ('faulty', 'clean'):
            raise HTTPException(422, 'mode must be faulty or clean.')
        args = [_evo_python(), str(_evo_script()), '--reset']
        if mode == 'faulty':
            args.append('--faulty')
        log_path = root / 'evolution-demo.log'
        with open(log_path, 'w', encoding='utf-8') as handle:
            _evo['proc'] = _sp.Popen(args, cwd=str(Path(__file__).parent), stdout=handle,
                                     stderr=_sp.STDOUT, text=True)
        _evo['state'] = 'running'
        _evo['started'] = time.time()
        _evo['log'] = str(log_path)
        return {'state': 'running', 'mode': mode}

    @app.post('/ml/evolution/rollback')
    def evolution_rollback():
        script = Path(__file__).parent / 'artifacts' / 'acceptance-20261114' / 'rollback_capability.py'
        out = _sp.run([_evo_python(), str(script), '--latest'], capture_output=True, text=True,
                      cwd=str(Path(__file__).parent))
        if out.returncode != 0:
            raise HTTPException(400, (out.stderr or out.stdout or 'rollback failed')[:200])
        return {'restored': (out.stdout or '').strip()[-160:]}

    @app.get('/ml/version')
    def version(): return {'version':APP_VERSION,'restart_required':source_fingerprint()!=LOADED_SOURCE_FINGERPRINT,'source_fingerprint':LOADED_SOURCE_FINGERPRINT,'asset_fingerprint':asset_fingerprint(),'auto_evolution_enabled':False,'data_source':'local_catalog','catalog_count':len(LocalCatalog().records())}

    @app.get('/static/{asset}',include_in_schema=False)
    def asset_file(asset:str):
        if asset not in {'ml-app.js','ml-app.css','pixel-theme.css','evolution.json'}:
            raise HTTPException(404,'Unknown asset.')
        return FileResponse(static/asset)
    busy=threading.Lock()

    def path_for(run_id):
        if len(run_id)!=32 or any(c not in '0123456789abcdef' for c in run_id): raise HTTPException(404,'Unknown run.')
        path=root/run_id
        if not (path/'state.json').exists(): raise HTTPException(404,'Unknown run.')
        return path

    @app.get('/',include_in_schema=False)
    def home(): return FileResponse(static/'ml-app.html')

    @app.get('/showcase',include_in_schema=False)
    @app.get('/research-agent',include_in_schema=False)
    def old_page(): return RedirectResponse('/')

    @app.post('/ml/credentials')
    def credentials(body:Credentials,request:Request):
        from urllib.parse import urlsplit
        origin=request.headers.get('origin')
        if origin and urlsplit(origin).netloc!=request.headers.get('host'):
            raise HTTPException(403,'Same-origin requests only.')
        updates={name:value for name,value in [('DEEPSEEK_API_KEY',body.deepseek),('JEV_API_KEY',body.jev)] if value}
        if not updates or any(not 8<=len(v)<=512 or not v.isascii() or any(c.isspace() for c in v) for v in updates.values()):
            raise HTTPException(400,'请填写有效密钥；原配置未修改，已保存的密钥仍然保留。')
        with key_lock:
            changed={**runtime_keys,**updates}
            try: credential_store.save(changed)
            except CredentialStoreError as exc: raise HTTPException(500,str(exc)) from None
            runtime_keys.clear()
            runtime_keys.update(changed)
        return {'configured':bool(changed.get('DEEPSEEK_API_KEY')),'persisted':True,'connection_verified':False}

    @app.get('/ml/credentials/status')
    def credential_status():
        with key_lock:
            return {'deepseek_configured':bool(runtime_keys.get('DEEPSEEK_API_KEY')),
                    'jev_configured':bool(runtime_keys.get('JEV_API_KEY')),
                    'persisted':credential_store.path.exists(),
                    'storage':'Windows 当前账户加密保存',
                    'load_error':credential_error if not runtime_keys else None}

    @app.post('/ml/connection-check')
    def connection_check(request:Request):
        from urllib.parse import urlsplit
        origin=request.headers.get('origin')
        if origin and urlsplit(origin).netloc!=request.headers.get('host'):
            raise HTTPException(403,'Same-origin requests only.')
        try: clients=factory()
        except (ValueError,RuntimeError):
            raise HTTPException(409,'请先保存两个 API 密钥。') from None
        results=[]
        for role in ('low','high','classifier'):
            client=clients[role]
            try:
                if role=='classifier':
                    client([{'role':'user','content':json.dumps({'task':'Choose a classifier for tabular data','capabilities':CAPABILITIES})}],timeout=15)
                else:
                    client([{'role':'user','content':'Reply with the JSON object {"ok":true} only.'}],timeout=15)
                results.append({'role':role,'model':client.model,'connected':True})
            except Exception as exc:
                import re
                match=re.search(r'HTTP (\d{3})',str(exc))
                code=match.group(1) if match else None
                reason={'401':'密钥无效或未获授权','402':'账户余额或计费问题','403':'权限不足','404':'模型或接口不可用','429':'请求限流或配额不足'}.get(code,'网络、TLS、超时或响应格式异常')
                results.append({'role':role,'model':client.model,'connected':False,'http_status':code,'reason':reason})
        return {'connected':all(r['connected'] for r in results),'results':results}

    @app.get('/ml/config')
    def config():
        try:
            clients=factory()
            return {'configured':True,'roles':{r:getattr(c,'model','configured') for r,c in clients.items()},
                    'classifier_source':getattr(clients['classifier'],'source','configured'),'auto_evolution_enabled':False,'capabilities':CAPABILITIES,'local_llm_required':False,
                    'note':'Tier labels are configured capabilities, not a verified claim of model price.'}
        except (ValueError,RuntimeError) as exc:
            return {'configured':False,'reason':str(exc),'capabilities':CAPABILITIES,'local_llm_required':False}

    @app.get('/ml/pricing')
    def pricing_status():
        price_feed.refresh_async()
        return price_feed.status()

    @app.post('/ml/pricing/refresh',status_code=202)
    def refresh_prices():
        price_feed.refresh_async(force=True)
        return price_feed.status()

    @app.get('/ml/catalog')
    def catalog_index():
        return {'provider':'local_catalog','datasets':[{k:r[k] for k in ('id','name','aliases','rows','columns','target')} for r in LocalCatalog().records()]}

    @app.get('/ml/sample')
    def sample():
        from sklearn.datasets import load_iris
        return {'csv_text':load_iris(as_frame=True).frame.to_csv(index=False),'target':'target'}

    def load_context(run_id):
        if not run_id: return None
        parent=path_for(run_id)
        state=read_saved_state(parent/'state.json')
        if state.get('status') in ('queued','running'):
            raise HTTPException(409,'上一轮尚未结束，请等待结果后继续提问。')
        if not (parent/'report.json').exists():
            raise HTTPException(409,'上一轮没有可引用的研究结果，请补充数据或打开已完成的任务。')
        report=json.loads((parent/'report.json').read_text(encoding='utf-8'))
        evidence_id=report.get('context',{}).get('evidence_run_id') or run_id
        evidence_path=path_for(evidence_id)
        return {'report':report,'parent_run_id':run_id,'evidence_run_id':evidence_id,'path':evidence_path}

    def needs_input_report(path,reason,gateway,source=None):
        datasets=[{'id':r['id'],'name':r['name']} for r in (catalog_factory or LocalCatalog)().records()] if hasattr((catalog_factory or LocalCatalog)(),'records') else []
        report={'research_status':'needs_clarification','review_performed':False,'research_outputs':None,
                'review':{'decision':'revise','answer':reason,'summary':'需要匹配的数据或明确的上下文，未生成无关预测。','experiment_ids':[]},
                'clarification':{'reason':reason,'available_datasets':datasets},'usage':gateway.snapshot(),
                'source':source or {},'events':[],'training_fits':0,'test_evaluated':False}
        save(path/'report.json',report)
        return {'run_id':path.name,'status':'needs_input','reason':reason,'available_datasets':datasets}

    def answer_followup(path,body,context,gateway):
        import copy
        report=copy.deepcopy(context['report'])
        evidence=evidence_summary(report,context['evidence_run_id'])
        append_event(path,{'event':'context_loaded','source_run':context['evidence_run_id'],'reason':'解释已有结果，不重新搜索或训练'})
        append_event(path,{'event':'review_started','tier':'reviewer','execution_kind':'answer_followup'})
        from ml_engine import object_response
        review=object_response(gateway.call('reviewer',[
            {'role':'system','content':ANSWER_RULES+' Return answer in Chinese. Do not claim any new execution. Approve an explanation that directly answers the current question, including explicitly acknowledging a quantity the user conditionally asks whether we computed. Revise if a requested supported deliverable is actually missing. Never invent evidence.'},
            {'role':'user','content':json.dumps({'current_question':body.goal,'answer_contract':answer_contract(body.goal),'evidence':evidence},ensure_ascii=False)}],timeout=35))
        if not isinstance(review.get('answer'),str) or not review['answer'].strip() or review.get('decision') not in ('approve','revise','reject'):
            raise ValueError('Follow-up reviewer returned invalid answer.')
        ids={r['experiment_id'] for r in report.get('experiments',[])}
        if any(x not in ids for x in review.get('experiment_ids',[])):
            raise ValueError('Follow-up cited an unknown experiment.')
        outputs=report.get('research_outputs')
        if outputs:
            outputs['question']=body.goal
            outputs['prediction_download_url']=evidence['prediction_download_url']
            outputs['predictions']=outputs.get('predictions',[])[:20]
        rendered=present_answer(review,body.goal,outputs,evidence['prediction_download_url'])
        prior_status=report.get('research_status')
        status='needs_revision'
        if rendered['decision']=='approve':
            status='partial_completed' if outputs and outputs.get('scope')=='training_association_only' else 'approved' if outputs else 'needs_clarification'
        report.update(review=rendered,
            execution_kind='answer_followup',context={k:context[k] for k in ('parent_run_id','evidence_run_id')},
            training_fits=0,test_evaluated=False,final_test=None,usage=gateway.snapshot(),review_performed=True,
            research_status=status,context_research_status=prior_status,events=[])
        append_event(path,{'event':'final_review','tier':'reviewer','review':report['review']})
        return report

    def execute(path,body,clients,context=None):
        engine=None
        terminal_state=None
        meter=BalanceMeter(clients)
        if client_factory is None:price_feed.refresh_async()
        gateway=TierBudget(_ClientSwitch(clients),ModelLimits(body.max_calls,body.max_tokens,body.max_seconds),price_feed=price_feed)
        try:
            save(path/'state.json',{'run_id':path.name,'status':'running'})
            meter.start()
            target=body.target
            goal=body.goal
            use_context=context is not None and body.csv_text is None and is_followup(goal)
            if use_context:
                catalog=(catalog_factory or LocalCatalog)()
                matches=catalog.search(goal) if hasattr(catalog,'records') else []
                old_id=context['report'].get('source',{}).get('dataset_id')
                import csv as csv_module
                previous_columns=next(csv_module.reader((context['path']/'data.csv').read_text(encoding='utf-8-sig').splitlines()),[]) if (context['path']/'data.csv').exists() else []
                compatible=any(len(set(previous_columns)&set(catalog.schema(r['id'])))>=max(2,len(previous_columns)//2) for r in matches) if matches and not old_id and hasattr(catalog,'schema') else False
                if matches and ((old_id and old_id not in {r['id'] for r in matches}) or (not old_id and not compatible)):
                    use_context=False
            if use_context:
                previous=context['report']
                columns=list(__import__('csv').reader((context['path']/'data.csv').read_text(encoding='utf-8').splitlines()))[0] if (context['path']/'data.csv').exists() else []
                gap=scope_gap(goal,columns,previous.get('source',{}))
                if gap:
                    terminal_state=needs_input_report(path,gap,gateway,previous.get('source'))
                    return
            if use_context and (training_forbidden(goal) or (not requires_experiment(goal) and not needs_new_analysis(goal,context['report'].get('research_outputs'),columns) and not needs_prediction_execution(goal,context['report'].get('research_outputs')))):
                report=answer_followup(path,body,context,gateway)
                save(path/'report.json',report)
                terminal_state={'run_id':path.name,'status':{'approved':'completed','partial_completed':'partial_completed','needs_clarification':'needs_input'}.get(report['research_status'],'needs_revision')}
                return
            if training_forbidden(goal):
                terminal_state=needs_input_report(path,'已遵守不训练的要求。本次没有关联可解释的已有研究结果，请打开原任务后追问；未执行模型训练。',gateway)
                return
            if use_context:
                if not (context['path']/'data.csv').exists():
                    terminal_state=needs_input_report(path,'上一轮没有可复用的数据，请上传 CSV 或明确数据集。',gateway)
                    return
                (path/'data.csv').write_bytes((context['path']/'data.csv').read_bytes())
                provenance=context['report'].get('source',{})
                target=target or context['report'].get('protocol',{}).get('target')
                target=research_target(goal,columns,target)
                save(path/'source.json',provenance)
                append_event(path,{'event':'context_loaded','source_run':context['evidence_run_id'],'reason':'复用上一轮数据，执行新的研究请求'})
            elif body.csv_text is None:
                append_event(path,{'event':'source_collection_started','provider':'local_catalog','input':'natural_language'})
                csv,target,provenance=collect_dataset(body.goal,gateway,(catalog_factory or LocalCatalog)(),emit=lambda event: append_event(path,event))
                (path/'data.csv').write_bytes(csv.encode('utf-8'))
                save(path/'source.json',provenance)
                append_event(path,{'event':'source_collected','provenance':provenance})
            else:
                save(path/'source.json',{'provider':'user_csv','target':target})
            import csv as csv_module
            columns=next(csv_module.reader((path/'data.csv').read_text(encoding='utf-8-sig').splitlines()),[])
            target=research_target(goal,columns,target)
            source_info=json.loads((path/'source.json').read_text(encoding='utf-8'))
            gap=scope_gap(goal,columns,source_info)
            if gap:
                terminal_state=needs_input_report(path,gap,gateway,source_info)
                return
            session=ResearchSession(path/'data.csv',target,seed=body.seed)
            session.ensure_ready()
            save(path/'protocol.json',{'dataset_sha256':session.dataset_sha256,'seed':session.seed,
                 'target':target,'model_roles':{r:getattr(c,'model','configured') for r,c in clients.items()},
                 'capabilities':CAPABILITIES,'request':body.model_dump(exclude={'csv_text'}),
                 'source':json.loads((path/'source.json').read_text(encoding='utf-8'))})
            memory_root=root/'memory-experiments'/body.memory_namespace if body.memory_namespace else root
            memory_root.mkdir(parents=True,exist_ok=True)
            engine=MLResearch(session,clients,strategy_store=strategies,lessons=LessonStore(memory_root/'experiment-lessons.sqlite3',session,source_run=path.name),experience=ExperienceBank(memory_root).find(session) if body.use_memory else None,consult_lessons=body.use_memory,deliverer=deliver,max_calls=body.max_calls,max_tokens=body.max_tokens,
                              max_seconds=body.max_seconds,max_fits=body.max_fits,max_expert_calls=body.max_expert_calls,
                              emit=lambda event: append_event(path,event),gateway=gateway,max_rounds=body.max_rounds,patience=body.patience,economy=body.economy,capability=CapabilityMemory(root/"capability.sqlite3"),
                              answer_context={'current_question':body.goal,'previous':evidence_summary(context['report'],context['evidence_run_id']) if use_context else None,
                                              'prediction_download_url':'/ml/runs/'+path.name+'/predictions'})
            report=engine.run(body.goal,body.evaluate_test)
            report['execution_kind']='research'
            if use_context:
                report['context']={'parent_run_id':context['parent_run_id'],'evidence_run_id':path.name}
            report['source']=json.loads((path/'source.json').read_text(encoding='utf-8'))
            report['protocol']=json.loads((path/'protocol.json').read_text(encoding='utf-8'))
            if report.get('research_outputs') and report['research_outputs'].get('records_predicted',0)>0:
                import csv as csv_module
                outputs=report['research_outputs']
                save(path/'predictions.json',outputs['predictions'])
                with (path/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as stream:
                    fields=['source_row','actual_label','predicted_label','correct','predicted_probability']+['probability_'+label for label in outputs['labels']]
                    writer=csv_module.DictWriter(stream,fieldnames=fields)
                    writer.writeheader()
                    for row in outputs['predictions']:
                        writer.writerow({**{k:row.get(k) for k in fields[:5]},**{'probability_'+k:v for k,v in row.get('probabilities',{}).items()}})
            save(path/'report.json',report)
            if body.memory_namespace:
                # Isolated evaluation history contains only reproducible champion
                # metadata, never account balances, secrets or raw prediction rows.
                save(memory_root/path.name/'report.json',{k:report.get(k) for k in ('research_status','best','protocol','source')})
            terminal_state={'run_id':path.name,'status':{'needs_clarification':'needs_input','blocked':'blocked','partial_completed':'partial_completed','approved':'completed'}.get(report['research_status'],'needs_revision')}
        except Exception as exc:
            if isinstance(exc,DataSourceError) and exc.needs_input:
                terminal_state=needs_input_report(path,exc.reason,gateway)
                return
            if engine and engine.lessons:
                try: engine.lessons.record(None,'failed',error_type=type(exc).__name__)
                except Exception: pass  # Preserve the original execution failure if lesson persistence also fails.
            if engine:
                save(path/'partial.json',{'events':engine.events,'usage':engine.gateway.snapshot(),
                                         'experiments':engine.results,'training_fits':engine.fits,'lessons':engine.lessons.snapshot() if engine.lessons else []})
            else:
                save(path/'partial.json',{'usage':gateway.snapshot(),'stage':'source_collection_or_data_validation'})
            # Keep the concrete failure reason: a bare class name cannot be acted on.
            _detail=' '.join(str(exc).split())
            _detail=''.join(ch for ch in _detail if ch.isprintable())[:200].rstrip('。.')
            terminal_state={'run_id':path.name,'status':'failed','failure_kind':failure_kind(exc),'error_type':type(exc).__name__,
                                   'stage':getattr(exc,'stage','research_execution' if engine else 'source_collection_or_data_validation'),
                                   'http_status':getattr(exc,'http_status',None),
                                   'detail':_detail,
                                   'reason':exc.reason if isinstance(exc,DataSourceError) else ({'network_failure':'网络请求失败（最多重试一次），已保留部分证据。','budget_exhausted':'调用、Token 或时间预算已用尽，已停止，不会超额重试。','model_protocol_error':'响应协议、认证或数据校验失败；不自动重试。'}[failure_kind(exc)]+' ')+ '研究已停止，部分记录已保留。失败类型：'+type(exc).__name__+'。'+(('具体原因：'+_detail+'。') if _detail else '')+'请检查模型响应、数据要求与预算。'}
        finally:
            try:
                try: observation=meter.finish()
                except Exception:
                    observation={'task_actual_charge':None,'task_actual_status':'provider_task_receipt_unavailable','attribution':'account_window_not_task_charge','observations':[],'errors':[{'status':'balance_unavailable'}]}
                save(path/'billing.json',observation)
                for filename in ('report.json','partial.json'):
                    file=path/filename
                    if file.exists():
                        payload=json.loads(file.read_text(encoding='utf-8'))
                        payload['account_billing']=observation
                        save(file,payload)
                if terminal_state is not None:save(path/'state.json',terminal_state)
            finally:busy.release()

    def append_event(path,event):
        with (path/'events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(event,ensure_ascii=False)+'\n')

    @app.post('/ml/runs',status_code=202)
    def start(body:MLInput,tasks:BackgroundTasks):
        context=load_context(body.previous_run_id) if body.csv_text is None else None
        try: clients=factory()
        except (ValueError,RuntimeError) as exc: raise HTTPException(409,str(exc)) from None
        if not busy.acquire(False): raise HTTPException(409,'A research run is already active.')
        path=root/uuid.uuid4().hex
        try:
            path.mkdir()
            if body.csv_text is not None:
                if not body.target: raise ValueError('CSV input requires a target column.')
                (path/'data.csv').write_bytes(body.csv_text.encode('utf-8'))
                check=ResearchSession(path/'data.csv',body.target,seed=body.seed)
                check.ensure_ready()
            save(path/'request.json',body.model_dump(exclude={'csv_text'}))
            save(path/'state.json',{'run_id':path.name,'status':'queued'})
        except Exception:
            busy.release()
            raise HTTPException(400,'Invalid dataset, unresolved feature risks or conflicting feedback.') from None
        tasks.add_task(execute,path,body,clients,context)
        return {'run_id':path.name,'status':'queued'}

    @app.get('/ml/runs/{run_id}')
    def show(run_id:str,compact:bool=False):
        path=path_for(run_id)
        state=read_saved_state(path/'state.json')
        for name in ('report','partial'):
            if (path/(name+'.json')).exists(): state[name]=json.loads((path/(name+'.json')).read_text(encoding='utf-8'))
        if compact and state.get('report',{}).get('research_outputs'):
            outputs=state['report']['research_outputs']
            rows=outputs.get('predictions',[])
            outputs['predictions_total']=len(rows)
            outputs['predictions_truncated']=len(rows)>20
            outputs['predictions']=rows[:20]
        if (path/'events.jsonl').exists():
            state['events']=[json.loads(line) for line in (path/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        return state

    comparisons=root/'comparisons'
    def comparison_path(comparison_id):
        if len(comparison_id)!=32 or any(c not in '0123456789abcdef' for c in comparison_id):
            raise HTTPException(404,'Unknown comparison.')
        path=comparisons/comparison_id
        if not (path/'state.json').exists():raise HTTPException(404,'Unknown comparison.')
        return path

    def execute_comparison(path,body,csv,target,provenance):
        try:
            if client_factory is None:price_feed.refresh_async()
            run_comparison(path,body,csv,target,factory,provenance,price_feed=price_feed)
        except Exception as exc:
            save(path/'state.json',{'comparison_id':path.name,'status':'failed','error_type':type(exc).__name__,'reason':'对比已停止，请查看已保存的分组记录。'})
        finally: busy.release()

    @app.post('/ml/comparisons',status_code=202)
    def start_comparison(body:ComparisonInput,tasks:BackgroundTasks):
        if len(set(body.seeds))!=len(body.seeds) or any(type(s) is not int or not 0<=s<=2147483647 for s in body.seeds):
            raise HTTPException(400,'随机种子必须为不重复的非负整数。')
        try: factory()
        except (ValueError,RuntimeError):raise HTTPException(409,'请先在设置中配置模型密钥。') from None
        if not busy.acquire(False):raise HTTPException(409,'研究或对比正在运行，请等待当前任务完成。')
        path=comparisons/uuid.uuid4().hex
        try:
            path.mkdir(parents=True)
            if body.csv_text is not None:
                if not body.target:raise ValueError('Target required')
                csv,target,provenance=body.csv_text,body.target,{'provider':'user_csv'}
            else: csv,target,provenance=LocalCatalog().download(body.dataset_id)
            (path/'data.csv').write_bytes(csv.encode('utf-8'))
            check=ResearchSession(path/'data.csv',target,seed=body.seeds[0]);check.ensure_ready()
            save(path/'request.json',body.model_dump(exclude={'csv_text'}))
            save(path/'state.json',{'comparison_id':path.name,'status':'queued','total_arms':2*len(body.seeds)})
        except Exception:
            busy.release()
            raise HTTPException(400,'数据或目标列不适用，尚未启动模型对比。') from None
        tasks.add_task(execute_comparison,path,body,csv,target,provenance)
        return {'comparison_id':path.name,'status':'queued'}

    @app.get('/ml/comparisons/{comparison_id}')
    def show_comparison(comparison_id:str):
        path=comparison_path(comparison_id)
        state=read_saved_state(path/'state.json')
        if (path/'report.json').exists():state['report']=json.loads((path/'report.json').read_text(encoding='utf-8'))
        return state

    @app.get('/ml/comparisons/{comparison_id}/export')
    def export_comparison(comparison_id:str):
        path=comparison_path(comparison_id)
        if not (path/'report.json').exists():raise HTTPException(409,'对比尚未完成。')
        return FileResponse(path/'report.json',media_type='application/json',filename='comparison-report.json')

    @app.get('/ml/runs/{run_id}/predictions')
    def prediction_export(run_id:str):
        path=path_for(run_id)
        if not (path/'predictions.csv').exists(): raise HTTPException(409,'当前任务尚无逐条预测结果。')
        return FileResponse(path/'predictions.csv',media_type='text/csv',filename='predictions.csv')

    @app.get('/ml/runs/{run_id}/export')
    def export(run_id:str):
        path=path_for(run_id)
        if not (path/'report.json').exists(): raise HTTPException(409,'No completed report; inspect partial evidence.')
        return FileResponse(path/'report.json',media_type='application/json',filename='ml-research-report.json')

    return app
