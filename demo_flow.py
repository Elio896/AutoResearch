"""交互式演示：让每一步的「思考」可见，人可以在中间改。

流程（对应她描述的四步）：

    1. 点「开始」  → 本地千问盘点数据，返回**自然语言的思考**（不是 JSON）
    2. 页面暂停    → 人看过之后可以改（改度量、改预算、改模型偏好）
    3. 点「同意并继续」 → 带着人的修改执行真实实验
    4. 分叉显示    → JEV 那一支（决策/该不该花钱）+ 云端 DeepSeek 那一支（生成方案）
                     两条线各自返回它想了什么

设计上的克制：
    - 人没点「同意」之前，不跑任何实验、不花任何 token
    - JEV 和云端两条线是**并行**的，各自独立，互不阻塞
    - 所有数字都是真跑出来的，不是编的
"""
from __future__ import annotations

import json
import tempfile
import time
import uuid

# 会话存在内存里（演示用，进程重启就没了，正好）
_SESSIONS: dict[str, dict] = {}
_MAX_SESSIONS = 20


def _prune() -> None:
    if len(_SESSIONS) <= _MAX_SESSIONS:
        return
    oldest = sorted(_SESSIONS.items(), key=lambda kv: kv[1]['created'])[:len(_SESSIONS) - _MAX_SESSIONS]
    for key, _ in oldest:
        _SESSIONS.pop(key, None)


def _local_proposer_if_available(model: str = 'qwen2.5:7b-instruct-q4_K_M'):
    """Ollama 在就返回一个真客户端，不在就返回 None。

    妈妈要的是"真的千问在思考"，不是规则话术。
    所以默认就去探一下，探到就用真模型——探不到才退回规则，并在界面上标明。
    """
    try:
        import urllib.request
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open('http://127.0.0.1:11434/api/ps', timeout=2) as resp:
            json.loads(resp.read().decode('utf-8', 'replace'))
        from orchestrator import LocalQwenClient
        return LocalQwenClient(base_url='http://127.0.0.1:11434', model=model)
    except Exception:                       # noqa: BLE001 - 没起就退回规则
        return None


def start(csv_text: str, target: str, metric: str, pos_label: str,
          proposer=None, model: str = 'qwen2.5:7b-instruct-q4_K_M',
          use_local_model: bool | None = None) -> dict:
    """第一步：盘点数据。返回千问的思考 + 它建议的方案，等人审批。

    use_local_model=None 表示自动探测：Ollama 可用就用真模型。
    """
    import tempfile
    from pathlib import Path
    from agents import Analyst
    from executor import ResearchSession

    if proposer is None:
        want_local = True if use_local_model is None else bool(use_local_model)
        if want_local:
            proposer = _local_proposer_if_available(model)

    tmp = tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False, encoding='utf-8')
    try:
        tmp.write(csv_text)
        tmp.close()
        session = ResearchSession(tmp.name, target, metric=metric, pos_label=pos_label)
        context = session.planning_context()
    finally:
        try:
            Path(tmp.name).unlink()
        except OSError:
            pass

    profile = Analyst().profile(context)
    thought = _think(profile, context, proposer, model)

    session_id = uuid.uuid4().hex[:12]
    _SESSIONS[session_id] = {
        'session_id': session_id,
        'created': time.time(),
        'csv_text': csv_text,
        'target': target,
        'metric': metric,
        'pos_label': pos_label,
        'profile': profile,
        'context': context,
        'model': model,
        'approved': False,
        'runs': None,
    }
    _prune()

    return {
        'session_id': session_id,
        'profile': profile,
        'raw_context': context,
        'thought': thought,
        'suggested': _suggest(profile, context),
        'message': '看过上面的判断，可以改参数，然后点「同意并继续」。',
    }


def _think(profile: dict, context: dict, proposer, model: str) -> dict:
    """千问的「思考」——用自然语言说它看到了什么。

    有本地模型时让它自己说；没有就退回规则生成的话术（但要标明来源）。
    """
    rows = profile.get('rows') or 0
    features = profile.get('features') or 0
    classes = profile.get('class_count') or 0
    minority = profile.get('minority_fraction')
    missing = context.get('missing_fraction') or 0
    risks = profile.get('risks') or []
    imbalanced = bool(profile.get('imbalanced'))

    # 先把这份数据要做的事拆成两堆：我做得动的、我做不了的。
    # 这是整个思考的主线——不是先盯模型，而是先看数据里有哪些活。
    can_do = []
    cannot_do = []

    can_do.append(f'把 {rows} 行 × {features} 列读进来，逐列看类型和取值范围')
    if missing > 0.01:
        can_do.append(f'{missing * 100:.1f}% 的格子空着——中位数/众数填补我做得了')
    else:
        can_do.append('没有空值，不需要填补')
    can_do.append('把类别列编码成数字（one-hot），数值列做标准化')
    if imbalanced:
        can_do.append(f'少数类只占 {minority * 100:.1f}%——加 class_weight 让它被看见，这我也会')
    else:
        can_do.append(f'类别大致均衡（最少一类 {minority * 100:.1f}%），不用重采样' if minority is not None
                      else '类别分布正常')
    can_do.append('跑一遍基线模型，把分数算出来对比')

    cannot_do.append('**该选哪个模型、为什么**——这要我判断业务含义，我判断不了')
    cannot_do.append('**分数掉了到底是因为模型还是数据**——要诊断，我做不了')
    cannot_do.append('**该不该换指标**（比如从整体 F1 换成少数类 F1）——这是研究决策')
    if rows and rows < 1000:
        cannot_do.append(f'只有 {rows} 行，这么小的样本该不该相信这个分数——要人来定')
    if imbalanced:
        cannot_do.append('少数类样本这么少，是该重采样还是该换目标——要人来定')

    observations = [
        f'先看数据：{rows} 行、{features} 个特征、{classes} 个类别' +
        (f'，有 {missing * 100:.1f}% 缺失' if missing > 0.01 else '，没有缺失') +
        (f'，少数类占 {minority * 100:.1f}%' if minority is not None else ''),
    ]

    if proposer is not None:
        try:
            prompt = (
                '你是数据研究员。先判断这份数据要做的事里，哪些是基础的数据处理（清洗/填补/编码/重采样），'
                '哪些需要人的判断（选模型/定指标/诊断失败）。用三句话说完，中文，口语，不要给方案。\n'
                + json.dumps({'profile': profile, 'risks': risks}, ensure_ascii=False)
            )
            response = proposer([{'role': 'user', 'content': prompt}], timeout=30)
            content = response if isinstance(response, str) else \
                ((response.get('message') or {}).get('content') or '')
            if content.strip():
                return {'source': 'local_model', 'model': model,
                        'text': content.strip()[:800],
                        'can_do': can_do, 'cannot_do': cannot_do,
                        'observations': observations}
        except Exception:          # noqa: BLE001 - 模型挂了就用规则话术，不影响演示
            pass

    return {'source': 'rules', 'model': None,
            'text': ' | '.join(observations),
            'observations': observations,
            'can_do': can_do,
            'cannot_do': cannot_do}


def _suggest(profile: dict, context: dict) -> dict:
    """它建议的初始配置。人可以改这三个。"""
    rows = profile.get('rows') or 0
    imbalanced = bool(profile.get('imbalanced'))
    metric = 'f1_minority' if imbalanced else 'f1_macro'
    model = 'logistic_regression' if (rows and rows < 1000) else 'hist_gradient_boosting'
    budget = 3 if rows and rows < 1000 else 5
    return {
        'metric': metric,
        'baseline_model': model,
        'max_experiments': budget,
        'class_weight': 'balanced' if imbalanced else None,
        'why': (f'{"不平衡" if imbalanced else "较均衡"} → 用 {metric}；'
                f'{rows} 行 → 基线用 {model}；给 {budget} 次机会'),
    }


def approve(session_id: str, overrides: dict | None = None) -> dict:
    """第二步：人改完（或直接同意），锁定配置。不跑实验。"""
    session = _SESSIONS.get(session_id)
    if session is None:
        raise KeyError('session 不存在或已过期')
    session['approved'] = True
    session['overrides'] = overrides or {}
    return {'ok': True, 'approved': True, 'config': {**_suggest(session['profile'], session['context']),
                                                     **(overrides or {})}}


def _runner_for(session: dict) -> 'ResearchLoop':
    """给一个 session 造（或复用）一个可以逐代推进的研究循环。"""
    from pathlib import Path
    from agents import Budget
    from executor import ResearchSession
    from research_loop import ResearchLoop

    loop = session.get('loop')
    if loop is not None:
        return loop
    # session 里存的是 csv_text（不是 csv），参数在 metric/pos_label，人工改动在 overrides
    overrides = session.get('overrides') or {}
    metric = overrides.get('metric') or session['metric']
    pos_label = str(overrides.get('pos_label') or session['pos_label'])
    max_experiments = int(overrides.get('max_experiments') or 3)
    # ResearchSession 要文件路径，先把内存里的 csv 落到临时文件
    tmp_dir = Path(tempfile.gettempdir()) / 'evoresearch-demo'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_dir / f'{session["session_id"] if "session_id" in session else uuid.uuid4().hex}.csv'
    if not csv_path.is_file():
        csv_path.write_text(session['csv_text'], encoding='utf-8')
    research = ResearchSession(
        csv_path, session['target'],
        metric=metric, pos_label=pos_label,
    )
    budget = Budget(max_experiments=max_experiments,
                    max_seconds=float(overrides.get('max_seconds') or 300.0),
                    max_local_calls=max_experiments + 2)
    proposer = session.get('proposer')
    if proposer is None:
        # 离线演示：用固定目录轮着提，保证真跑但不依赖模型
        baseline = overrides.get('baseline_model') or 'logistic_regression'
        catalogue = [
            {'model': baseline, 'parameters': {}},
            {'model': 'hist_gradient_boosting', 'parameters': {'max_depth': 4}},
            {'model': 'random_forest', 'parameters': {'n_estimators': 50}},
            {'model': 'hist_gradient_boosting', 'parameters': {'max_depth': 8}},
        ]
        state = {'i': 0}

        def proposer(messages, timeout=None):
            item = dict(catalogue[state['i'] % len(catalogue)])
            state['i'] += 1
            item.update({'hypothesis': f'第 {state["i"]} 个方案试试看',
                         'expected_direction': 'up',
                         'falsifier': '验证分数没提升'})
            return {'message': {'content': json.dumps(item)},
                    'prompt_eval_count': 0, 'eval_count': 0}
    session['proposer'] = proposer
    loop = ResearchLoop(research, proposer=proposer, budget=budget, patience=2)
    session['loop'] = loop
    return loop


def step(session_id: str) -> dict:
    """手动模式：推进一代，返回当前全部状态。

    每一代都是真的跑一次实验，不是摆样子。
    """
    session = _SESSIONS.get(session_id)
    if session is None:
        raise KeyError('session 不存在或已过期')
    if not session['approved']:
        raise PermissionError('还没有人确认，不能开跑')

    loop = _runner_for(session)
    loop.max_rounds_override = 1
    result = loop.run()
    loop.max_rounds_override = None
    session['runs'] = result
    experiments = result.get('experiments') or []
    return {
        'rounds': len(experiments),
        'experiments': experiments,
        'summary': result['summary'],
        'best_series': [e.get('validation_score', e.get('validation_f1'))
                        for e in experiments],
        'done': bool(result['summary'].get('stop_reason')) and
                result['summary'].get('stop_reason') not in ('rounds_cap', 'experiment_budget'),
        'stop_reason': result['summary'].get('stop_reason'),
    }


def auto(session_id: str, rounds: int = 10) -> dict:
    """自动模式：连续跑 N 代。

    上限由 session 的预算决定，不许超过——不然就成了无限烧。
    """
    session = _SESSIONS.get(session_id)
    if session is None:
        raise KeyError('session 不存在或已过期')
    if not session['approved']:
        raise PermissionError('还没有人确认，不能开跑')

    overrides = session.get('overrides') or {}
    cap = max(1, int(overrides.get('max_experiments') or 8))
    wanted = max(1, min(int(rounds), cap))

    loop = _runner_for(session)
    loop.max_rounds_override = wanted
    result = loop.run()
    loop.max_rounds_override = None
    session['runs'] = result
    experiments = result.get('experiments') or []
    return {
        'rounds': len(experiments),
        'requested': int(rounds),
        'capped_to': cap,
        'experiments': experiments,
        'summary': result['summary'],
        'best_series': [e.get('validation_score', e.get('validation_f1'))
                        for e in experiments],
        'done': bool(result['summary'].get('stop_reason')),
        'stop_reason': result['summary'].get('stop_reason'),
    }


def execute(session_id: str, offline: bool = True,
            use_jev: bool = False, use_cloud: bool = False) -> dict:
    """第三步：带人改过的配置真跑，然后返回 JEV 和云端两支的思考。

    use_jev / use_cloud 默认关：演示默认不花钱。
    打开后就真的调 JEV 判、真的调云端（要 key）。
    """
    session = _SESSIONS.get(session_id)
    if session is None:
        raise KeyError('session 不存在或已过期')
    if not session['approved']:
        raise PermissionError('还没有人确认，不能开跑')

    import tempfile
    from pathlib import Path
    from agents import Budget
    from executor import ResearchSession
    from research_loop import ResearchLoop

    config = {**_suggest(session['profile'], session['context']), **(session.get('overrides') or {})}
    tmp = tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False, encoding='utf-8')
    try:
        tmp.write(session['csv_text'])
        tmp.close()
        run = ResearchSession(tmp.name, session['target'], metric=config['metric'],
                              pos_label=session['pos_label'])
        if offline:
            catalogue = [
                {'model': config['baseline_model'], 'parameters': {}},
                {'model': 'hist_gradient_boosting', 'parameters': {'max_depth': 4}},
                {'model': 'random_forest', 'parameters': {'n_estimators': 50}},
            ]
            state = {'i': 0}

            def proposer(messages, timeout=None):
                item = dict(catalogue[state['i'] % len(catalogue)])
                state['i'] += 1
                item.update({'hypothesis': f'第 {state["i"]} 个方案试试看',
                             'expected_direction': 'up',
                             'falsifier': '验证分数没提升'})
                return {'message': {'content': json.dumps(item)},
                        'prompt_eval_count': 0, 'eval_count': 0}
        else:
            from orchestrator import LocalQwenClient
            proposer = LocalQwenClient(model=session['model'])

        budget = Budget(max_experiments=int(config['max_experiments']), max_seconds=120.0,
                        max_local_calls=int(config['max_experiments']) + 1)
        loop = ResearchLoop(run, proposer=proposer, budget=budget, patience=2)
        result = loop.run()
    finally:
        try:
            Path(tmp.name).unlink()
        except OSError:
            pass

    # 两支：有 key 且用户要求时才真调
    jev_gate = None
    cloud = None
    if use_jev:
        try:
            from jev_router import JevGate
            jev_gate = JevGate.auto()
        except Exception:                       # noqa: BLE001
            jev_gate = None
    if use_cloud:
        try:
            import os
            from cloud_provider import DeepSeekClient
            key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
            if key:
                cloud = DeepSeekClient(key)
        except Exception:                       # noqa: BLE001
            cloud = None

    config['rows'] = session['profile'].get('rows') or 0
    branches = _branches(result, config, jev_gate=jev_gate, cloud=cloud)
    evidence = _evidence_pack(result, config, branches)
    session['runs'] = result
    return {'summary': result['summary'], 'events': result['events'],
            'experiments': result['experiments'], 'config': config,
            'branches': branches, 'evidence': evidence, 'offline': offline,
            'used_jev': jev_gate is not None, 'used_cloud': cloud is not None}


def _branches(result: dict, config: dict, jev_gate=None, cloud=None) -> dict:
    """两条支线各自的「思考」。

    jev       —— 决策支：有真 JEV 就用真 JEV 判；没有就用规则判（并标明）
    cloud     —— 生成支：JEV 说"值得"时才真调云端；否则只说明"它会怎么做"
    """
    summary = result.get('summary') or {}
    events = result.get('events') or []
    score = summary.get('best_score') or 0
    stuck = [e for e in events if e.get('event') == 'reviewer_verdict'
             and (e.get('verdict') or {}).get('verdict') in ('flat', 'regressed')]

    # ---- 决策支：先问 JEV ----
    jev_state = {
        'task': 'demo',
        'trigger': 'validation_plateau' if stuck else 'still_progressing',
        'experiments_used': len(events_by(events, 'experiment')),
        'best_validation_score': round(float(score), 4),
        'recent_scores': [round(float(e.get('score') or 0), 4)
                          for e in events_by(events, 'experiment')][-4:],
        'remaining_cloud_calls': 1,
        'rows': config.get('rows') or 0,
    }

    verdict, reason, cost, source = '不值得', '本地还在进步，现在花钱是浪费', '0 token（纯规则判断）', 'rules'
    if jev_gate is not None:
        try:
            decision = jev_gate(jev_state, timeout=30)
            source = decision.get('source', 'jev')
            allow = bool(decision.get('allow_cloud'))
            verdict = '值得' if allow else '不值得'
            probability = decision.get('probability')
            model_name = decision.get('model')
            if probability is not None:
                reason = (f'JEV 判「有价值」的概率 {probability:.2f}'
                          f'（阈值 {decision.get("threshold", 0.75)}），模型 {model_name}')
                cost = '按 token 计费（这次真调了）'
            else:
                reason = f'JEV 不可用，退回规则判定：{decision.get("reason", "")}'
        except Exception as exc:                    # noqa: BLE001
            reason = f'JEV 调用失败，退回规则：{str(exc)[:80]}'

    jev_thought = {
        'name': 'JEV · 决策支',
        'question': '这轮卡住了，值不值得花钱让云端接手？',
        'signals': jev_state,
        'verdict': verdict,
        'reason': reason,
        'source': source,
        'cost': cost,
    }

    # ---- 生成支：值得才真调 ----
    cloud_thought = {
        'name': 'DeepSeek · 生成支',
        'question': '如果交给我，我会怎么改？',
        'status': '未调用',
        'reason': '本地还在进步，暂时不需要我',
        'cost': '按 token 计费，只在上面那支说「值得」时才花',
    }
    if cloud is not None and verdict == '值得':
        try:
            history = [{'model': e.get('model'), 'parameters': e.get('parameters'),
                        'score': e.get('score')} for e in events_by(events, 'experiment')][-5:]
            prompt = (
                '你是云端专家。本地规划器卡住了，这是它的实验历史：\n'
                + json.dumps(history, ensure_ascii=False)
                + '\n用两句话说明你会怎么改，不要给完整 JSON。'
            )
            response = cloud([{'role': 'user', 'content': prompt}], timeout=45)
            text = response if isinstance(response, str) else \
                ((response.get('message') or {}).get('content') or '')
            cloud_thought.update({
                'status': '已调用',
                'reason': text.strip()[:500] or '云端没有给出内容',
                'cost': '这次真花了 token',
            })
        except Exception as exc:                    # noqa: BLE001
            cloud_thought.update({'status': '调用失败', 'reason': str(exc)[:160]})

    return {'jev': jev_thought, 'cloud': cloud_thought}


def _evidence_pack(result: dict, config: dict, branches: dict) -> dict:
    """把评委要看的证据打包。每一项都标数据来源，不许含糊。"""
    summary = result.get('summary') or {}
    events = result.get('events') or []
    experiments = result.get('experiments') or []

    scores = [float(e.get('validation_score') or e.get('validation_f1') or 0) for e in experiments]
    baseline = scores[0] if scores else None
    best = max(scores) if scores else None
    abs_gain = round(best - baseline, 4) if (best is not None and baseline is not None) else None
    rel_gain = round((best - baseline) / baseline * 100, 2) if (baseline and best is not None) else None

    # token 台账：本地不花钱，云端按调用次数估
    local_tokens = int(summary.get('local_tokens') or 0)
    cloud_calls = int(summary.get('cloud_calls') or 0)
    cloud_tokens = int(summary.get('cloud_tokens') or 0)
    # DeepSeek 定价按公开价粗估（输入 ¥1/百万、输出 ¥2/百万），只作为量级参考
    cloud_cost_cny = round(cloud_tokens / 1_000_000 * 1.5, 4) if cloud_tokens else 0.0

    # 成本按来源拆开：JEV 和 DeepSeek 要各自独立可查
    jev_events = annotate(events, 'jev_decision') + annotate(events, 'jev_harness_decision')
    jev_input = sum(int((e.get('usage') or {}).get('input_tokens') or 0) for e in jev_events)
    jev_output = sum(int((e.get('usage') or {}).get('output_tokens') or 0) for e in jev_events)
    # TypeSafe Jev 按 System One 计价（粗估 ¥2/千次请求，仅量级参考）
    jev_cost_cny = round(len(jev_events) * 0.002, 4) if jev_events else 0.0

    cloud_events = annotate(events, 'cloud_proposal')
    ds_input = sum(int((e.get('tokens') or {}).get('prompt') or 0) for e in cloud_events)
    ds_output = sum(int((e.get('tokens') or {}).get('completion') or 0) for e in cloud_events)
    ds_cost_cny = round((ds_input / 1_000_000 * 1.0 + ds_output / 1_000_000 * 2.0), 4) if cloud_events else 0.0

    cost_breakdown = {
        'jev': {'calls': len(jev_events), 'input_tokens': jev_input,
                'output_tokens': jev_output, 'cost_cny': jev_cost_cny,
                'note': 'TypeSafe Jev System One，按请求计价'},
        'deepseek': {'calls': len(cloud_events), 'input_tokens': ds_input,
                     'output_tokens': ds_output, 'cost_cny': ds_cost_cny,
                     'note': 'DeepSeek，按 token 计价'},
        'total': {'calls': len(jev_events) + len(cloud_events),
                  'cost_cny': round(jev_cost_cny + ds_cost_cny, 4)},
    }

    # 迭代能力：多轮下来稳不稳、有没有持续变好
    iteration = {'rounds': len(scores), 'trend': None, 'stability': None,
                 'verdict': '数据不足（至少 2 轮才能判断）'}
    if len(scores) >= 2:
        half = max(1, len(scores) // 2)
        first_half = scores[:half]
        second_half = scores[half:] or scores[-1:]
        mean_first = sum(first_half) / len(first_half)
        mean_second = sum(second_half) / len(second_half)
        variance = sum((s - (sum(scores) / len(scores))) ** 2 for s in scores) / len(scores)
        std = variance ** 0.5
        delta = mean_second - mean_first
        if delta > 0.01:
            verdict = f'越跑越好（后半段均值 +{delta:.4f}）'
        elif delta < -0.01:
            verdict = f'有退步（后半段均值 {delta:.4f}），但回滚机制兜住了'
        else:
            verdict = '基本持平（没有明显变好也没有崩）'
        iteration = {
            'rounds': len(scores),
            'mean_first_half': round(mean_first, 4),
            'mean_second_half': round(mean_second, 4),
            'delta': round(delta, 4),
            'std': round(std, 4),
            'stability': round(1 - min(std, 1.0), 4),
            'verdict': verdict,
        }

    # 闭环六步：每步都要有事件支撑
    def has(name: str) -> bool:
        return bool(annotate(events, name))

    closed_loop = [
        {'step': '规划', 'done': has('manager_plan'),
         'evidence': 'Manager 按预算决定实验次数与停止条件'},
        {'step': '收集证据', 'done': has('analyst_profile'),
         'evidence': 'Analyst 输出数据画像（行列/缺失/不平衡/风险）'},
        {'step': '提出假设', 'done': bool(annotate(events, 'scientist_proposal')),
         'evidence': 'Scientist 每条提案都带 hypothesis + falsifier'},
        {'step': '实验', 'done': has('experiment'),
         'evidence': f'Experimenter 真跑 {len(experiments)} 次，分数来自 sklearn'},
        {'step': '验证', 'done': has('reviewer_verdict'),
         'evidence': 'Reviewer 按分数差判定 improved/regressed/flat'},
        {'step': '修正', 'done': bool(annotate(events, 'rollback')) or (summary.get('evoundo_rollbacks') or 0) > 0,
         'evidence': '退步自动回滚到历史最优，被抛弃的版本保留'},
    ]

    # 自主调度：JEV 判了几次、云端调了几次
    jev_decisions = annotate(events, 'jev_decision')
    qwen_decisions = annotate(events, 'qwen_router_decision')
    autonomy = {
        'jev_evaluated': len(jev_decisions),
        'qwen_evaluated': len(qwen_decisions),
        'cloud_called': cloud_calls,
        'note': ('JEV/千问先判，只有判「值得」才调用云端'
                 if (jev_decisions or qwen_decisions)
                 else '本次走规则判定，未触发真 JEV（勾选「调 Jev」可验证）'),
    }

    return {
        'performance': {
            'baseline_score': baseline,
            'best_score': best,
            'abs_gain': abs_gain,
            'rel_gain_pct': rel_gain,
            'rounds': len(scores),
            'note': '基线 = 第一轮；最佳 = 所有轮次最高',
        },
        'token_ledger': {
            'local_tokens': local_tokens,
            'local_cost_cny': 0.0,
            'cloud_calls': cloud_calls,
            'cloud_tokens': cloud_tokens,
            'cloud_cost_cny': cloud_cost_cny,
            'total_cost_cny': round(cloud_cost_cny, 4),
            'note': '本地不花钱；云端按公开价粗估，实际以账单为准',
        },
        'cost_breakdown': cost_breakdown,
        'iteration': iteration,
        'closed_loop': closed_loop,
        'autonomy': autonomy,
        'reproducibility': {
            'dataset_sha256': (experiments[0] or {}).get('dataset_sha256') if experiments else None,
            'split_sizes': (experiments[0] or {}).get('split_sizes') if experiments else None,
            'note': '同 seed 重跑 diff=0（见 tools/verify_determinism.py）',
        },
        'rollback': {
            'versions': summary.get('evoundo_versions'),
            'rollbacks': summary.get('evoundo_rollbacks'),
            'best_version': summary.get('evoundo_best'),
            'note': '被回滚的版本保留，作为「假设被否证」的证据',
        },
        'branches': {k: (v.get('verdict') or v.get('status')) for k, v in branches.items()},
    }


def annotate(events: list, name: str) -> list:
    return [e for e in events if e.get('event') == name]


# 旧名字保留，避免其它调用点报错
events_by = annotate
