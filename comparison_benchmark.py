"""Paired, cold-start validation-only comparison. No assumed cost advantage."""
import copy,hashlib,json,random,statistics,time
from pathlib import Path
from executor import ResearchSession
from ml_engine import MLResearch
from research_deliverables import deliver
from research_system import save

def summarize(report,session,elapsed):
    from sklearn.metrics import average_precision_score,confusion_matrix,f1_score
    pipeline,_=session.experiments[report['best']['experiment_id']]
    actual=session.y.iloc[session.validation_indices]
    predictions=pipeline.predict(session.x.iloc[session.validation_indices])
    minority=int(session.y.iloc[session.train_indices].value_counts().idxmin())
    usage=report['usage']
    complete=usage['unreconciled_calls']==0
    tokens_in=sum(r['usage']['prompt_tokens'] for r in usage['records'] if r['usage'])
    tokens_out=sum(r['usage']['completion_tokens'] for r in usage['records'] if r['usage'])
    ap=None
    if len(session.label_encoder.classes_)==2 and hasattr(pipeline,'predict_proba'):
        probability=pipeline.predict_proba(session.x.iloc[session.validation_indices])[:,list(pipeline.classes_).index(minority)]
        ap=float(average_precision_score(actual==minority,probability))
    rows=report.get('research_outputs') or {}
    association_only=rows.get('scope')=='training_association_only'
    research_ready=report['research_status'] in ('approved','partial_completed')
    checks={'final_answer_present':report['research_status'] not in ('blocked','needs_clarification') and bool(report['review'].get('answer')),
            'data_results_present':bool(rows.get('association_evidence',{}).get('features')) if association_only else rows.get('records_predicted')==len(session.validation_indices),
            'limitations_present':bool(rows.get('limitations'))}
    return {'status':'completed','research_status':report['research_status'],
        # A blocked/needs-revision arm only carries its untouched baseline score.
        # Numbers stay visible for audit, but they are not a research result.
        'baseline_only':not research_ready,
        'analysis_scope':rows.get('scope'),'validation_macro_f1':None if association_only else report['best']['validation_f1'],
        'minority_label':str(session.label_encoder.classes_[minority]),
        'minority_f1':None if association_only else float(f1_score(actual==minority,predictions==minority,zero_division=0)),
        'minority_pr_auc_average_precision':None if association_only else ap,
        'confusion_matrix':None if association_only else confusion_matrix(actual,predictions,labels=list(range(len(session.label_encoder.classes_)))).tolist(),
        'labels':[str(v) for v in session.label_encoder.classes_],
        'usage_records':usage['records'],'model_roles':report.get('model_roles',{}),'calls':usage['calls'],'known_input_tokens':tokens_in,'known_output_tokens':tokens_out,
        'known_tokens':usage['known_tokens'],'usage_complete':complete,
        'unreconciled_calls':usage['unreconciled_calls'],'wall_seconds':elapsed,
        'model_request_seconds':sum(r['elapsed_seconds'] for r in usage['records']),
        'training_seconds':sum(r['duration_seconds'] for r in report['experiments']),
        'training_fits':report['training_fits'],'failed_calls':sum(r['status']!='returned' for r in usage['records']),
        'final_answer':report['review'].get('answer'),'delivery_checks':checks,'delivery_checks_passed':sum(checks.values()),
        'costs':usage.get('costs'),'cost':None,'cost_note':'费用未对账；Token 数量不等于费用。','test_evaluated':False}

def aggregate(rows):
    result={}
    for mode in ('routed','high_only'):
        group=[r for r in rows if r['mode']==mode]
        # Only arms that finished a research task may contribute metric means; a
        # blocked arm's number is the untouched baseline and must not be averaged in.
        ready=[r for r in group if r['metrics']['status']=='completed' and not r['metrics'].get('baseline_only')]
        result[mode]={'runs':len(group),'completed':len(ready),'not_research_ready':len(group)-len(ready),
            'approved':sum(r['metrics'].get('research_status')=='approved' for r in group)}
        for key in ('validation_macro_f1','minority_f1','known_tokens','wall_seconds','calls','training_fits'):
            values=[r['metrics'][key] for r in ready if isinstance(r['metrics'].get(key),(int,float))]
            result[mode][key]={'mean':statistics.mean(values) if values else None,
                'std':statistics.stdev(values) if len(values)>1 else None}
    paired=[]
    for seed in sorted({r['seed'] for r in rows}):
        pair={r['mode']:r['metrics'] for r in rows if r['seed']==seed}
        if all(pair.get(m,{}).get('status')=='completed' for m in ('routed','high_only')):
            a,b=pair['routed'],pair['high_only']
            matching_scope=a.get('analysis_scope')==b.get('analysis_scope')
            ready=not a.get('baseline_only') and not b.get('baseline_only')
            comparable=bool(matching_scope and ready)
            reason=None if comparable else ('unfinished_arm' if not ready else 'scope_mismatch')
            paired.append({'seed':seed,'comparable_scope':matching_scope,'comparable':comparable,'reason':reason,
                'macro_f1_delta':a['validation_macro_f1']-b['validation_macro_f1'] if comparable and isinstance(a.get('validation_macro_f1'),(int,float)) and isinstance(b.get('validation_macro_f1'),(int,float)) else None,
                'known_tokens_delta':a['known_tokens']-b['known_tokens'] if comparable and a['usage_complete'] and b['usage_complete'] else None,
                'calculated_usd_delta':a['costs']['calculated_total_usd']-b['costs']['calculated_total_usd'] if comparable and (a.get('costs') or {}).get('calculated_total_usd') is not None and (b.get('costs') or {}).get('calculated_total_usd') is not None else None,
                'wall_seconds_delta':a['wall_seconds']-b['wall_seconds'] if comparable else None})
    def savings(value,reference):
        if not isinstance(value,(int,float)) or not isinstance(reference,(int,float)) or reference<=0:return None
        return (reference-value)/reference*100
    for pair in paired:
        arms={r['mode']:r['metrics'] for r in rows if r['seed']==pair['seed']}
        a,b=arms['routed'],arms['high_only']
        complete=pair['comparable'] and a['usage_complete'] and b['usage_complete']
        pair['token_savings_percent']=savings(a['known_tokens'],b['known_tokens']) if complete else None
        pair['cost_savings_percent']=savings((a.get('costs') or {}).get('calculated_total_usd'),(b.get('costs') or {}).get('calculated_total_usd')) if complete else None
        pair['time_savings_percent']=savings(a['wall_seconds'],b['wall_seconds']) if pair['comparable'] else None
        pair['speedup']=b['wall_seconds']/a['wall_seconds'] if pair['comparable'] and a['wall_seconds']>0 and b['wall_seconds']>0 else None
    comparable=[p for p in paired if p['comparable']]
    relative={'paired_runs':len(comparable),'conclusive':False,'note':'Descriptive paired observations only; no statistical superiority or invoice savings established.'}
    for key in ('token_savings_percent','cost_savings_percent','time_savings_percent','speedup'):
        values=[p[key] for p in comparable if p[key] is not None]
        relative[key]={'mean':statistics.mean(values) if values else None,'n':len(values)}
    return {'relative_summary':relative,'by_mode':result,'paired_deltas_routed_minus_high_only':paired,
        'interpretation':'负 Token/时间差表示协作模式本次较少；正 F1 差表示协作模式本次较高。失败或未完成研究的运行单列，不计作零成本、零耗时，也不参与均值与差值计算。',
        'limitations':['相同数据、目标、划分、工具、允许算法、预算和初始基线；禁用历史经验与学习写入。',
            '比较范围为固定数据后的研究过程，不包含两边共用的数据获取步骤。',
            '高成本单模型直接执行实验和审核，不调用 Jev 或低成本模型；调用次数允许自然不同。',
            '只有审核通过或已交付关联分析的臂才进入指标均值；被阻塞的臂只保留基线分数供审计。',
            '结构完整度仅检查字段存在，不是回答正确性或因果有效性评分。',
            '单次或少量种子不足以证明普遍优越；顺序及缓存、网络波动会影响时间。',
            '仅比较验证集，不访问最终测试集；费用没有供应商账单对账。']}

def run_comparison(path,body,csv,target,client_factory,provenance,price_feed=None):
    path=Path(path);rows=[];began=time.monotonic()
    def progress(**detail):save(path/'state.json',{'comparison_id':path.name,'status':'running',**detail})
    for seed in body.seeds:
        order=['routed','high_only'];random.Random(seed).shuffle(order)
        for mode in order:
            progress(seed=seed,mode=mode,completed_arms=len(rows),total_arms=len(body.seeds)*2)
            arm=path/(str(seed)+'-'+mode);arm.mkdir()
            (arm/'data.csv').write_bytes(csv.encode('utf-8'))
            start=time.monotonic();engine=None
            try:
                session=ResearchSession(arm/'data.csv',target,seed=seed)
                session.ensure_ready()
                clients=client_factory()
                expensive=clients['high']
                audit=copy.deepcopy(expensive)
                audit.max_output_tokens=getattr(clients.get('reviewer'),'max_output_tokens',2048)
                clients['reviewer']=audit
                if mode=='high_only':
                    clients={role:expensive for role in ('source','classifier','low','high')}
                    clients['reviewer']=audit
                def emit(event):
                    with (arm/'events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(event,ensure_ascii=False)+'\n')
                    progress(seed=seed,mode=mode,event=event['event'],completed_arms=len(rows),total_arms=len(body.seeds)*2)
                engine=MLResearch(session,clients,max_calls=body.max_calls,max_tokens=body.max_tokens,
                    max_seconds=body.max_seconds,max_fits=body.max_fits,
                    max_expert_calls=body.max_fits,deliverer=deliver,execution_mode=mode,emit=emit,price_feed=price_feed)
                report=engine.run(body.goal,False)
                report['model_roles']={k:getattr(v,'model','configured') for k,v in clients.items()}
                metrics=summarize(report,session,time.monotonic()-start)
                save(arm/'report.json',report)
            except Exception as exc:
                usage=engine.gateway.snapshot() if engine else None
                metrics={'status':'failed','error_type':type(exc).__name__,'usage':usage,
                    'wall_seconds':time.monotonic()-start,'cost':None,'costs':usage.get('costs') if usage else None}
            row={'seed':seed,'mode':mode,'metrics':metrics}
            rows.append(row);save(arm/'metrics.json',row)
    result={'comparison_id':path.name,'goal':body.goal,'provenance':provenance,
        'protocol':{'seeds':body.seeds,'memory':'disabled_for_both','test_access':False,
                    'dataset_sha256':hashlib.sha256(csv.encode()).hexdigest(),'target':target,'per_arm_limits':{'max_calls':body.max_calls,'max_tokens':body.max_tokens,'max_seconds':body.max_seconds,'max_fits':body.max_fits}},
        'arms':rows,'aggregate':aggregate(rows),'total_wall_seconds':time.monotonic()-began}
    save(path/'report.json',result)
    save(path/'state.json',{'comparison_id':path.name,'status':'completed','completed_arms':len(rows),'total_arms':len(rows)})
    return result
