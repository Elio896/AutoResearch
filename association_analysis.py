"""Training-only descriptive group association, never causal identification."""
import math

def association_report(session,columns):
    import pandas as pd
    if not isinstance(columns,list) or not 1<=len(columns)<=3 or any(not isinstance(c,str) or c not in session.x.columns for c in columns):
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
    return {'question':goal,'scope':'training_association_only','target':session.target,
            'records_predicted':0,'prediction_counts':{},'labels':[],'predictions':[],
            'feature_evidence':[],'association_evidence':evidence,
            'answer':'仅凭当前数据和已执行分析，不能确立因果关系；已继续完成可支持的分组关联分析。',
            'causal_conclusion':'当前分析未建立因果识别，不能据此确立因果关系。',
            'scope_adjustment_reason':reason,'limitations':evidence['limitations']}
