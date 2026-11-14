"""Deterministic answer evidence, not an answer correctness score."""
import math,re

def checked_facts(outputs,download_url=None,plain=False):
    outputs=outputs or {};assoc=outputs.get('association_evidence') or {}
    count=outputs.get('records_predicted');scope=outputs.get('scope')
    prediction=scope=='validation_rows_only' and type(count) is int and count>0
    association=scope=='training_association_only' and type(assoc.get('observations')) is int and assoc['observations']>0
    link=download_url if type(count) is int and count>0 and scope in (None,'validation_rows_only') and isinstance(download_url,str) and re.fullmatch(r'/ml/runs/[0-9a-f]{32}/predictions',download_url) else None
    text=[];refs=[]
    if prediction:
        text.append(f'已对 {count} 条验证记录生成预测。');refs.append('research_outputs.records_predicted')
        score=outputs.get('validation_macro_f1')
        if not plain and type(score) in (int,float) and math.isfinite(score) and 0<=score<=1:
            text.append(f'验证宏 F1={score:.4f}。');refs.append('research_outputs.validation_macro_f1')
        text.append('适用范围：验证记录不是未来客户；验证集参与方案选择，不是独立最终测试。')
        text.append('特征重要性不能证明因果。')
        if outputs.get('calibration_performed') is False:
            text.append('分数未校准，不能当作可靠发生概率。')
            refs.append('research_outputs.calibration_performed')
        else:text.append('校准状态未记录，概率解释需另行核实。')
        if link:text.append('明细下载：'+link);refs.append('download_url')
        else:text.append('本回答未提供已核验的下载链接。')
    elif association:
        text.append(f'统计来自 {assoc["observations"]} 条训练记录。');refs.append('research_outputs.association_evidence.observations')
        for feature in assoc.get('features',[])[:1]:
            for group in feature.get('groups',[])[:2]:
                n,rate=group.get('n'),group.get('rate')
                if type(n) is int and n>0 and type(rate) in (int,float) and math.isfinite(rate) and 0<=rate<=1:
                    text.append(f'{feature["feature"]} / {group["group"]}：n={n}，观察比例 {rate:.1%}。')
                    refs.append('research_outputs.association_evidence.features.groups')
        text.append('适用范围：描述性关联不能证明因果；小样本和分组方法会影响结果，不能直接推广到其他人群。未生成客户预测文件。')
    return {'text':''.join(text),'download_url':link,'references':refs,'scope':scope,
            'scope_supported':prediction or association,'direct_answer_verified':False}

def related_association_columns(goal,requested,available):
    """Require BOTH a relationship question and an explicit existing field reference."""
    from research_context import FIELD_NAMES
    lower=goal.lower()
    if not any(w in lower for w in ('causal','causality','causation','association','因果','关联','关系','导致','影响')):return []
    selected=[]
    for column in requested:
        if column not in available:continue
        labels=[column,column.replace('_',' '),FIELD_NAMES.get(column,'')]
        if any(label and label.lower() in lower for label in labels):selected.append(column)
    return selected
