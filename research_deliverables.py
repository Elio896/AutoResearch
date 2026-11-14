"""Evidence-based research delivery and exact-dataset strategy reuse."""
import hashlib
import json
from pathlib import Path

class ExperienceBank:
    def __init__(self,root): self.root=Path(root)
    def find(self,session):
        best=None
        for path in self.root.glob('*/report.json'):
            try:
                report=json.loads(path.read_text(encoding='utf-8'))
                result=report['best']
                if report.get('research_status')!='approved': continue
                if result['dataset_sha256']!=session.dataset_sha256 or result['seed']!=session.seed: continue
                if str(result.get('pos_label','1'))!=session.pos_label: continue
                if result.get('selection_metric')!=session.metric or result.get('feature_review',{})!=(session.feature_decisions or {}): continue
                if result.get('excluded_columns',[])!=session.risk_report['excluded_columns']: continue
                target=report.get('protocol',{}).get('target') or report.get('protocol',{}).get('request',{}).get('target')
                if not target and report.get('source',{}).get('provider')=='local_catalog':
                    from local_catalog import LocalCatalog
                    target=next((r['target'] for r in LocalCatalog().records() if r['id']==report['source'].get('dataset_id')),None)
                if target!=session.target: continue
                if not best or result['validation_score']>best['score']:
                    best={'model':result['model'],'parameters':result['parameters'],
                          'preprocessing':result.get('preprocessing',{}),'score':result['validation_score'],
                          'source_run':path.parent.name}
            except (KeyError,ValueError,TypeError,OSError): continue
        return best

def deliver(session,best,goal):
    import numpy as np
    from sklearn.inspection import permutation_importance
    pipeline,_=session.experiments[best['experiment_id']]
    indices=session.validation_indices
    x=session.x.iloc[indices]
    actual=session.y.iloc[indices].to_numpy()
    predicted=pipeline.predict(x)
    probabilities=pipeline.predict_proba(x) if hasattr(pipeline,'predict_proba') else None
    labels=[str(v) for v in session.label_encoder.classes_]
    rows=[]
    for offset,index in enumerate(indices):
        pred=int(predicted[offset]);truth=int(actual[offset])
        row={'source_row':int(index)+1,'predicted_label':labels[pred],
             'actual_label':labels[truth],'correct':pred==truth}
        if probabilities is not None:
            row['probabilities']={labels[int(c)]:float(probabilities[offset,j]) for j,c in enumerate(pipeline.classes_)}
            row['predicted_probability']=row['probabilities'][labels[pred]]
        rows.append(row)
    counts={label:sum(r['predicted_label']==label for r in rows) for label in labels}
    sample=x.sample(n=min(300,len(x)),random_state=session.seed)
    importance=permutation_importance(pipeline,sample,session.y.loc[sample.index],scoring='f1_macro',
        n_repeats=2,random_state=session.seed,n_jobs=1)
    order=np.argsort(importance.importances_mean)[::-1][:6]
    factors=[{'feature':str(x.columns[i]),'macro_f1_drop':float(importance.importances_mean[i]),
              'std':float(importance.importances_std[i])} for i in order]
    factors=[f for f in factors if f['macro_f1_drop']>0]
    return {'question':goal,'scope':'validation_rows_only','target':session.target,
        'records_predicted':len(rows),'prediction_counts':counts,'labels':labels,
        'validation_macro_f1':best['validation_f1'],'validation_accuracy':best['validation_accuracy'],
        'feature_evidence':factors,'feature_method':'Validation permutation importance, 2 repeats, at most 300 sampled rows; exploratory.',
        'answer':'针对你的研究需求，本轮已对 '+str(len(rows))+' 条留出验证记录给出逐条分类预测。预测分布为：'+
            '；'.join(k+'：'+str(v)+' 条' for k,v in counts.items())+'。验证宏 F1 为 '+format(best['validation_f1'],'.4f')+'。'+
            ('对模型预测较重要的字段包括：'+ '、'.join(f['feature'] for f in factors[:4])+'。' if factors else '当前未发现稳定的正向字段重要性证据。'),
        'limitations':['这是已有标签的验证样本预测，不是未来客户或新上传客户的预测。',
            '字段重要性表示模型预测依赖，不能证明因果或影响方向；验证集参与了方案选择。',
            '概率未做校准；字段重要性为小样本探索结果，需独立数据复核。'],
        'predictions':rows}
