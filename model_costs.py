"""Official USD list-price reconciliation, explicitly separate from invoice charges."""
from datetime import datetime,timezone
PRICES_VERIFIED='2026-11-14'
DEEPSEEK_URL='https://api-docs.deepseek.com/quick_start/pricing/'
JEV_URL='https://help.typesafe.ai/articles/5187202169-how-much-does-jev-cost'
# USD / million tokens: cache hit, cache miss, output; off-peak values.
RATES={'deepseek-flash':(.003,.15,.6),'deepseek-v4-pro':(.022,.66,1.98)}

def reference_prices():
    return {'deepseek':{'rates':{k:list(v) for k,v in RATES.items()},'peak_multiplier':2,
        'verified_date':PRICES_VERIFIED,'fetched_at':None,'source_url':DEEPSEEK_URL,'status':'reference'},
        'typesafe':{'input_per_million':.042,'output_per_million':0,'verified_date':PRICES_VERIFIED,
        'fetched_at':None,'source_url':JEV_URL,'status':'reference'}}

def request_cost(record):
    model,provider=record.get('model','unknown'),record.get('provider','unknown')
    usage=record.get('usage')
    prices=record.get('price_snapshot') or reference_prices()
    schedule=prices.get(provider,{})
    row={'provider':provider,'model':model,'currency':'USD','verified_date':schedule.get('verified_date',PRICES_VERIFIED),'price_fetched_at':schedule.get('fetched_at'),'price_status':schedule.get('status','unknown'),
         'input_tokens':usage.get('prompt_tokens') if usage else None,
         'output_tokens':usage.get('completion_tokens') if usage else None,
         'cache_hit_tokens':None,'cache_miss_tokens':None,'calculated_usd':None,
         'calculated_min_usd':None,'calculated_max_usd':None,'invoice_actual_usd':None,
         'invoice_status':'未获取供应商账单实扣','warnings':[],'rates':None}
    if schedule.get('status')!='live':row['warnings'].append('未在线核验最新价格或刷新失败，使用记录中的历史/参考价快照')
    if not usage:
        row['warnings'].append('用量未返回，不能按零费用处理');return row
    inp,out=usage['prompt_tokens'],usage['completion_tokens']
    if provider=='typesafe' and (model=='jev-latest' or model.startswith('jev-')):
        cost=(inp*schedule.get('input_per_million',.042)+out*schedule.get('output_per_million',0))/1_000_000
        row.update(calculated_usd=cost,calculated_min_usd=cost,calculated_max_usd=cost,
                   rates={'input_per_million':schedule.get('input_per_million',.042),'output_per_million':schedule.get('output_per_million',0)},source_url=JEV_URL)
        return row
    if str(model).lower().startswith(('qwen','ollama','local:')):
        row.update(calculated_usd=0.0,calculated_min_usd=0.0,calculated_max_usd=0.0,
                   rates={'input_per_million':0.0,'output_per_million':0.0})
        row['warnings'].append('本地推理，无 API 费用');return row
    if provider!='deepseek' or model not in schedule.get('rates',{}):
        row['warnings'].append('供应商或型号没有已核验价格');return row
    hit_rate,miss_rate,output_rate=schedule['rates'][model]
    peak_multiplier=schedule.get('peak_multiplier',2)
    multipliers=[1,peak_multiplier]
    try:
        stamp=datetime.fromisoformat(record['request_utc'].replace('Z','+00:00'))
        if stamp.tzinfo is None:raise ValueError()
        stamp=stamp.astimezone(timezone.utc)
        # Exact off-peak outside busy weekday hours; Chinese holiday calendar
        # not assumed inside those hours, so expose a range instead.
        if stamp.weekday()>=5 or not (1<=stamp.hour<4 or 6<=stamp.hour<10):multipliers=[1]
        else:row['warnings'].append('高峰时段的中国节假日日历未核定，显示高低峰范围')
    except (KeyError,ValueError):row['warnings'].append('未记录请求时间，显示高低峰范围')
    detail=record.get('billing_usage') or {}
    hit,miss=detail.get('prompt_cache_hit_tokens'),detail.get('prompt_cache_miss_tokens')
    if type(hit) is int and 0<=hit<=inp and miss is None:miss=inp-hit
    if type(miss) is int and 0<=miss<=inp and hit is None:hit=inp-miss
    cache_known=type(hit) is int and type(miss) is int and hit>=0 and miss>=0 and hit+miss==inp
    if cache_known:
        row.update(cache_hit_tokens=hit,cache_miss_tokens=miss)
        low=high=(hit*hit_rate+miss*miss_rate+out*output_rate)/1_000_000
    else:
        row['warnings'].append('缓存用量缺失，显示缓存命中到未命中的费用范围')
        low=(inp*hit_rate+out*output_rate)/1_000_000
        high=(inp*miss_rate+out*output_rate)/1_000_000
    row.update(calculated_min_usd=low*min(multipliers),calculated_max_usd=high*max(multipliers),
               rates={'offpeak_cache_hit_per_million':hit_rate,'offpeak_cache_miss_per_million':miss_rate,
                      'offpeak_output_per_million':output_rate,'possible_multipliers':multipliers},source_url=DEEPSEEK_URL)
    if cache_known and len(multipliers)==1:row['calculated_usd']=row['calculated_min_usd']
    return row

def cost_breakdown(records):
    groups={}
    for record in records:
        cost=request_cost(record);key=(cost['provider'],cost['model'])
        if key not in groups:
            groups[key]={'provider':key[0],'model':key[1],'currency':'USD','calls':0,'known_input_tokens':0,
                'known_output_tokens':0,'known_cache_hit_tokens':0,'known_cache_miss_tokens':0,
                'unknown_usage_calls':0,'calculated_usd':0.,'calculated_min_usd':0.,'calculated_max_usd':0.,
                'invoice_actual_usd':None,'invoice_status':'未获取供应商账单实扣','warnings':[],
                'price_sources':[],'requests':[]}
        group=groups[key];group['calls']+=1;group['requests'].append(cost)
        group['known_input_tokens']+=cost['input_tokens'] or 0;group['known_output_tokens']+=cost['output_tokens'] or 0
        group['known_cache_hit_tokens']+=cost['cache_hit_tokens'] or 0;group['known_cache_miss_tokens']+=cost['cache_miss_tokens'] or 0
        group['unknown_usage_calls']+=record.get('usage') is None
        for field in ('calculated_usd','calculated_min_usd','calculated_max_usd'):
            if group[field] is None or cost[field] is None:group[field]=None
            else:group[field]+=cost[field]
        group['warnings']=list(dict.fromkeys(group['warnings']+cost['warnings']))
        if cost.get('source_url') and cost['source_url'] not in group['price_sources']:group['price_sources'].append(cost['source_url'])
    models=list(groups.values())
    def total(field):return sum(r[field] for r in models) if all(r[field] is not None for r in models) else None
    return {'currency':'USD','models':models,'calculated_total_usd':total('calculated_usd'),
            'calculated_min_usd':total('calculated_min_usd'),'calculated_max_usd':total('calculated_max_usd'),
            'invoice_actual_usd':None,'price_verified_date':list(dict.fromkeys(r['verified_date'] for m in models for r in m['requests'])),
            'basis':'按接口实际用量与核验日官方公开美元单价核算；非账户账单实扣。缺失用量不按零计。'}
