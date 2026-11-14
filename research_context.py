"""Explicit conversation references and evidence-bound answer presentation.

Never selects a global 'latest run'. Context comes only from a caller's run ID.
"""
import re

FIELD_NAMES = {
    'international_plan': '是否开通国际套餐',
    'voice_mail_plan': '是否开通语音信箱套餐',
    'number_customer_service_calls': '客服通话次数',
    'total_day_minutes': '日间通话分钟数',
    'total_day_charge': '日间通话费用',
    'total_eve_minutes': '晚间通话分钟数',
    'total_night_minutes': '夜间通话分钟数',
    'account_length': '在网时长', 'class': '是否流失',
    'total_intl_minutes': '国际通话分钟数', 'total_intl_charge': '国际通话费用',
    'total_intl_calls': '国际通话次数', 'total_day_calls': '日间通话次数',
    'total_eve_charge': '晚间通话费用', 'total_eve_calls': '晚间通话次数',
    'total_night_charge': '夜间通话费用', 'total_night_calls': '夜间通话次数',
    'number_vmail_messages': '语音信箱消息数', 'area_code': '地区代码', 'state': '地区',
}


def target_from_question(goal, columns):
    """Only change target on an explicit prediction-target request."""
    match = re.search(r'(?:预测目标|目标改为|目标换成|目标：|目标:|另一项?目标)[：:是为\s]*(.*)', goal)
    if not match:
        return None
    requested = match.group(1)
    for column in columns:
        if str(column).lower() in requested.lower() or FIELD_NAMES.get(str(column), str(column)) in requested:
            return str(column)
    if '国际套餐' in requested and 'international_plan' in columns:
        return 'international_plan'
    return None


def scope_gap(goal, columns, source):
    """Narrow, explicit domain/field checks, before fitting or delivery."""
    lower = goal.lower()
    labour = any(w in lower for w in ('失业', '劳动力', '就业政策', 'unemployment', 'labor force', 'labour force'))
    identity = ' '.join(str(source.get(k, '')) for k in ('dataset_id', 'name')).lower()
    if labour and (any(w in identity for w in ('churn', 'iris', 'wine', 'breast', 'cancer')) or 'number_customer_service_calls' in columns):
        return '当前数据是' + ('客户流失数据' if 'churn' in identity else '其他领域的分类数据') + '，没有失业率、就业状态或劳动力观测，不能回答这个失业研究问题。请提供匹配的劳动力数据及年份、地区和统计口径。'
    if any(w in lower for w in ('收入', '薪资', 'income', 'salary')) and not any(any(w in str(c).lower() for w in ('income', 'salary', '收入', '薪资')) for c in columns):
        return '当前数据没有收入字段，不能分析收入对流失的影响，也不会用通话费用冒充收入。请上传包含收入及研究目标的匹配数据。'
    return None


def is_followup(goal):
    # New data/retraining intent takes precedence over referential words.
    return not any(w in goal.lower() for w in ('换一个数据','换数据','另一个数据','新的数据','new dataset'))


def requires_experiment(goal):
    if any(w in goal.lower() for w in ("重新训练", "重新研究", "retrain", "再训练")): return True
    return any(w in goal for w in ('继续尝试', '再尝试', '尝试改进', '继续优化', '继续改进', '重新训练', '重新预测', '未验证过的方案', '选择一个尚未', '换一个目标', '另一个目标', '另一项目标', '只研究', '改为研究', '不做预测', '预测目标'))


def answer_contract(goal):
    if all(w in goal for w in ('直接回答', '证据', '预测结果', '局限')):
        headings = ['直接回答', '证据', '预测结果', '局限']
    elif '三段' in goal or all(w in goal for w in ('发现了什么', '证据是什么', '有哪些局限')):
        headings = ['发现了什么', '证据是什么', '有哪些局限']
    elif '备忘' in goal:
        headings = ['发现', '可采取的行动', '仍需验证的假设']
    else:
        headings = []
    cap = re.search(r'(?:不超过|不多于|最多|限)(\d{2,4})\s*字', goal)
    return {'headings': headings, 'plain_language': any(w in goal for w in ('运营', '没有机器学习背景', '通俗')),
            'max_characters': int(cap.group(1)) if cap else 2000}


def evidence_summary(report, source_run):
    outputs = report.get('research_outputs') or {}
    source = report.get('source') or {}
    best = report.get('best') or {}
    return {'source_run_id': source_run, 'previous_question': outputs.get('question') or report.get('protocol', {}).get('request', {}).get('goal'),
            'source': source, 'previous_answer': (report.get('review') or {}).get('answer', '')[:2000],
            'research_status': report.get('research_status'),
            'research_outputs': {k: v for k, v in outputs.items() if k not in ('predictions',)},
            'experiments': [{k: r.get(k) for k in ('experiment_id', 'model', 'parameters', 'validation_score')} for r in report.get('experiments', [])][-8:],
            'best': {k: best.get(k) for k in ('experiment_id', 'model', 'validation_score')},
            'experience': report.get('experience'), 'lessons': report.get('lessons', [])[:8],
            'calibration_performed': False, 'performance_confidence_interval_computed': False,
            'test_evaluated': report.get('test_evaluated', False),
            'prediction_download_url': outputs.get('prediction_download_url') or ('/ml/runs/' + source_run + '/predictions' if outputs.get('records_predicted') else None)}


ANSWER_RULES = '''Answer the CURRENT user question from supplied evidence only. Prior questions/answers are context, not new instructions. Do not retrieve data or train for a request to explain/reformat existing results. Respect the requested headings, audience and length. For operations users use Chinese business field labels, put model names, F1 and tuning details in the developer panel. Explain that validation rows were excluded from model fitting but used in model selection, so they are not an untouched final test. No calibration was performed: classifier scores are uncalibrated, not assured probabilities. No performance/prediction confidence interval was computed unless explicitly present; grouped Wilson intervals are a different quantity. Cite the actual prediction CSV link if one exists. Never invent downloads, income fields, causal effects, training or experience. Explicitly explain any prior scheme reuse and its scope when asked. Return JSON with decision approve|revise|reject, answer, summary, experiment_ids and optional sections (object keyed by requested Chinese headings). A missing research deliverable must be acknowledged.'''


def _present_requested_format(review, goal, outputs=None, download_url=None):
    """Apply user format without extra model calls; use only computed evidence."""
    contract = answer_contract(goal)
    outputs = outputs or {}
    answer = review.get('answer') or ''
    sections = review.get('sections') if isinstance(review.get('sections'), dict) else {}
    limits = '；'.join(outputs.get('limitations', [])[:2]) or '结论仅适用于已提供的数据与已执行的方法，未计算的量不作推断。'
    records = outputs.get('records_predicted', 0)
    prediction = ('已生成 ' + str(records) + ' 条验证记录预测。下载：' + download_url) if records and download_url else '本次没有新增逐条预测；未计算的预测或区间不能提供。'
    assoc = outputs.get('association_evidence') or {}
    proof = ('已执行基于 ' + str(assoc['observations']) + ' 条训练记录的分组关联分析，分组样本量和区间见证据表。') if assoc else ('已有 ' + str(records) + ' 条验证记录的实际预测，明细见预测表。') if records else '实际数据、方法与执行记录见本轮报告，缺失部分未补造。'
    if contract['headings']:
        defaults = {'直接回答': answer, '证据': proof, '预测结果': prediction, '局限': limits,
                    '发现了什么': answer, '证据是什么': proof, '有哪些局限': limits,
                    '发现': answer, '可采取的行动': '优先核查观察到的异常客群和服务问题，先开展小规模验证，再决定是否推广干预。',
                    '仍需验证的假设': '需要新增数据或干预实验检验关联机制，当前证据不能证明因果。'}
        supplied = all(str(sections.get(h) or '').strip() for h in contract['headings'])
        if supplied:
            answer = '\n\n'.join(h + '：\n' + str(sections[h]) for h in contract['headings'])
        elif all(h in answer for h in contract['headings']):
            # The model already produced the requested structure. Re-assembling it
            # from defaults would repeat every section twice.
            pass
        else:
            answer = '\n\n'.join(h + '：\n' + str(defaults[h]) for h in contract['headings'])
    elif records and download_url and download_url not in answer:
        answer += '\n预测明细下载：' + download_url
    if contract['plain_language']:
        for name, chinese in FIELD_NAMES.items():
            answer = answer.replace(name, chinese)
        answer = re.sub(r'[^。\n]*(?:F1|随机森林|random_forest|logistic_regression)[^。\n]*[。]?', '', answer, flags=re.I)
    # Preserve explicit calibration/interval honesty even if a summary omitted it.
    if '校准' in goal:
        answer = '本次预测分数没有经过概率校准，不能当作经验证的真实发生概率。\n' + answer
    if '置信区间' in goal and (not assoc or '预测' in goal):
        answer = '当前实验没有计算预测结果或模型性能的置信区间，不能提供未计算的区间。\n' + answer
    cap = contract['max_characters']
    if len(answer) > cap:
        # Keep every required section and any download URL within the user budget.
        headings = contract['headings']
        if headings:
            chunks = answer.split('\n\n')
            room = max(12, (cap - sum(len(h) + 4 for h in headings)) // len(headings))
            answer = '\n\n'.join(c[:len(headings[i]) + 3 + room] for i, c in enumerate(chunks[:len(headings)]))[:cap]
        else:
            suffix = ('\n预测明细下载：' + download_url) if records and download_url else ''
            answer = answer[:max(0, cap - len(suffix) - 1)] + '。' + suffix
    review['answer'] = answer
    review['answer_contract'] = contract
    return review


def present_answer(review,goal,outputs=None,download_url=None):
    """Supplement real evidence without promoting incomplete research to success."""
    from answer_evidence import checked_facts
    contract=answer_contract(goal)
    facts=checked_facts(outputs,download_url,contract['plain_language'])
    direct=review.get('answer')
    missing=not isinstance(direct,str) or not direct.strip()
    # Preserve the original direct answer; supplementary facts never stand in for it.
    review=_present_requested_format(review,goal,outputs,facts['download_url'])
    answer=review['answer'].split('\n核对信息：',1)[0]
    answer=re.sub(r'/ml/runs/[0-9a-f]{32}/predictions',
                  lambda m:m.group(0) if m.group(0)==facts['download_url'] else '（未核验的下载链接已移除）',answer)
    for field,label in sorted(FIELD_NAMES.items(),key=lambda x:-len(x[0])):
        replacement=label if contract['plain_language'] else label+'（'+field+'）'
        answer=re.sub(r'(?<![A-Za-z0-9_（])'+re.escape(field)+r'(?![A-Za-z0-9_])',lambda _:replacement,answer)
        facts['text']=re.sub(r'(?<![A-Za-z0-9_（])'+re.escape(field)+r'(?![A-Za-z0-9_])',lambda _:replacement,facts['text'])
    suffix='\n核对信息：'+facts['text'] if facts['text'] else ''
    cap=contract['max_characters']
    # Keep complete safety facts in metadata even when a user requests an
    # impossibly short answer. Do not silently claim the contract was satisfied.
    if len(suffix)+min(len(answer),60)>cap:
        review['decision']='revise'
        review.setdefault('unmet_requirements',[]).append('character_budget_too_small')
        answer=(answer+'\n字数限制不足以完整呈现范围与交付信息，请查看结构化证据。')[:cap]
    elif suffix:
        room=cap-len(suffix)
        answer=answer[:room-1]+'…' if len(answer)>room else answer
        answer+=suffix
    if missing:
        review['decision']='revise'
        review.setdefault('unmet_requirements',[]).append('missing_direct_answer')
    review['answer']=answer
    review['answer_evidence']=facts
    return review
