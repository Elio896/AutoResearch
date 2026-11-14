"""Training-only descriptive group association, never causal identification."""
import math


def target_group_comparison(session):
    """Describe real training features by outcome group; no causal inference."""
    import pandas as pd
    x=session.x.iloc[session.train_indices]
    y=session.y.iloc[session.train_indices]
    labels=sorted(int(v) for v in y.unique())
    numeric=[];categorical=[]
    for name in x.columns:
        series=x[name];groups=[]
        if pd.api.types.is_numeric_dtype(series):
            values=[]
            for label in labels:
                data=series.loc[y[y==label].index].dropna()
                mean=float(data.mean()) if len(data) else None
                groups.append({'label':str(session.label_encoder.classes_[label]),'n':int((y==label).sum()),
                               'nonmissing_n':len(data),'mean':mean,'median':float(data.median()) if len(data) else None})
                if mean is not None:values.append(mean)
            scale=float(series.std())
            contrast=(max(values)-min(values))/scale if len(values)>1 and math.isfinite(scale) and scale>0 else 0.
            numeric.append({'feature':str(name),'type':'numeric','groups':groups,'exploratory_standardized_range':contrast})
        else:
            values=series.astype('string').fillna('缺失')
            categories=list(values.value_counts().head(6).index)
            for label in labels:
                data=values.loc[y[y==label].index];counts=data.value_counts()
                groups.append({'label':str(session.label_encoder.classes_[label]),'n':len(data),
                               'categories':[{'value':str(v),'count':int(counts.get(v,0)),'proportion':float(counts.get(v,0)/len(data))} for v in categories]})
            contrast=max((max(g['categories'][i]['proportion'] for g in groups)-min(g['categories'][i]['proportion'] for g in groups) for i in range(len(categories))),default=0.)
            categorical.append({'feature':str(name),'type':'categorical','groups':groups,'exploratory_max_proportion_gap':contrast})
    numeric.sort(key=lambda r:-r['exploratory_standardized_range'])
    categorical.sort(key=lambda r:-r['exploratory_max_proportion_gap'])
    return {'observations':len(x),'features':[],'comparison':{'numeric':numeric[:3],'categorical':categorical[:3],
            'method':'Training-only outcome-group means/medians and categorical proportions; exploratory ranking within feature type.'},
            'causal_identification_performed':False,
            'limitations':['仅比较训练样本中不同目标类别的特征分布，未进行因果识别。',
                           '差异排序是探索性描述，没有进行显著性检验或多重比较校正，也没有估计因果效应。']}

def association_report(session,columns):
    import pandas as pd
    if not isinstance(columns,list) or not 1<=len(columns)<=8 or any(not isinstance(c,str) or c not in session.x.columns for c in columns):
        raise ValueError('Association columns must be existing reviewed features')
    x=session.x.iloc[session.train_indices]
    y=session.y.iloc[session.train_indices]
    positive=session.positive_id
    if positive is None:positive=int(y.value_counts().idxmin())
    label=str(session.label_encoder.classes_[positive])
    results=[]
    for name in columns:
        series=x[name]
        if pd.api.types.is_numeric_dtype(series) and series.nunique()>10:
            groups=pd.qcut(series,q=4,duplicates='drop').astype('string').fillna('缺失').astype(str)
        else:groups=series.astype('string').fillna('缺失').astype(str)
        counts=groups.value_counts()
        retained=set(counts[counts>=5].nlargest(8).index)
        groups=groups.map(lambda g:g if g in retained else '其他或小样本组')
        rows=[]
        for group in sorted(groups.unique()):
            values=(y.loc[groups[groups==group].index]==positive)
            n=len(values);k=int(values.sum());rate=k/n
            z=1.959963984540054;den=1+z*z/n
            center=(rate+z*z/(2*n))/den
            radius=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
            rows.append({'group':str(group),'low_sample':n<30,'n':n,'positive_count':k,'rate':rate,
                         'ci95_lower':max(0.,center-radius),'ci95_upper':min(1.,center+radius)})
        results.append({'feature':name,'groups':rows,'observations':len(x)})
    return {'scope':'training_rows_only','positive_label':label,'observations':len(x),
            'method':'Observed group rates with Wilson95% intervals; numeric values use quartiles when >10 unique values; small/rare groups merged.',
            'features':results,'causal_identification_performed':False,
            'limitations':['仅描述现有样本中的分组关联，未控制所有混杂或排除反向因果。',
                '置信区间是各组观察比例的不确定性，不是因果效应区间。',
                '少于30条记录的分组样本很小，不能据此作稳定结论；其他或小样本组合并项不能推断具体取值。',
                '未考虑复杂抽样权重，未做独立重复验证；不能直接推广到其他人群。']}

def association_delivery(session,best,goal,evidence,reason):
    result={'question':goal,'scope':'training_association_only','target':session.target,
            'records_predicted':0,'prediction_counts':{},'labels':[],'predictions':[],
            'feature_evidence':[],'association_evidence':evidence,
            'answer':'仅凭当前数据和已执行分析，不能确立因果关系；已继续完成可支持的分组关联分析。',
            'causal_conclusion':'当前分析未建立因果识别，不能据此确立因果关系。',
            'scope_adjustment_reason':reason,'limitations':evidence['limitations']}
    if evidence.get('comparison'):
        result['target_group_comparison']=evidence['comparison']
        result['answer']='已比较不同目标类别在训练样本中的特征均值、中位数与类别比例。请根据实际比较表解释差异。'
    return result
