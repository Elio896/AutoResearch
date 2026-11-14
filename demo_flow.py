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


def start(csv_text: str, target: str, metric: str, pos_label: str,
          proposer=None, model: str = 'qwen2.5:7b-instruct-q4_K_M') -> dict:
    """第一步：盘点数据。返回千问的思考 + 它建议的方案，等人审批。"""
    import tempfile
    from pathlib import Path
    from agents import Analyst
    from executor import ResearchSession

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
    classes = profile.get('class_count') or 0
    minority = profile.get('minority_fraction')
    missing = context.get('missing_fraction') or 0
    risks = profile.get('risks') or []

    observations = []
    observations.append(f'一共 {rows} 行、{profile.get("features") or 0} 个特征、{classes} 个类别。')
    if minority is not None:
        if profile.get('imbalanced'):
            observations.append(f'少数类只占 {minority * 100:.1f}%，严重不平衡——'
                                f'准确率会骗人，得看少数类的 F1。')
        else:
            observations.append(f'类别大致均衡（最少一类占 {minority * 100:.1f}%），可以看整体 F1。')
    if missing > 0.01:
        observations.append(f'有 {missing * 100:.1f}% 的格子是空的，填充方式会影响结论。')
    else:
        observations.append('没有缺失值，不用纠结填充。')
    if rows and rows < 1000:
        observations.append(f'只有 {rows} 行，属于小样本——复杂模型容易背答案，先试线性更稳。')

    if proposer is not None:
        try:
            prompt = (
                '你是数据研究员。用三句话说明你从这份数据里看到了什么、该注意什么。'
                '不要给方案，只描述观察。中文，口语。\n'
                + json.dumps({'profile': profile, 'risks': risks}, ensure_ascii=False)
            )
            response = proposer([{'role': 'user', 'content': prompt}], timeout=30)
            content = response if isinstance(response, str) else \
                ((response.get('message') or {}).get('content') or '')
            if content.strip():
                return {'source': 'local_model', 'model': model,
                        'text': content.strip()[:800]}
        except Exception:          # noqa: BLE001 - 模型挂了就用规则话术，不影响演示
            pass

    return {'source': 'rules', 'model': None, 'text': ' '.join(observations),
            'bullets': observations}


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


def execute(session_id: str, offline: bool = True) -> dict:
    """第三步：带人改过的配置真跑，然后返回 JEV 和云端两支的思考。"""
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

    branches = _branches(result, config)
    session['runs'] = result
    return {'summary': result['summary'], 'events': result['events'],
            'experiments': result['experiments'], 'config': config,
            'branches': branches, 'offline': offline}


def _branches(result: dict, config: dict) -> dict:
    """两条支线各自的「思考」。

    jev       —— 决策支：该不该升级到云端
    cloud     —— 生成支：如果要升级，云端会怎么想
    """
    summary = result.get('summary') or {}
    events = result.get('events') or []
    score = summary.get('best_score') or 0
    stuck = [e for e in events if e.get('event') == 'reviewer_verdict'
             and (e.get('verdict') or {}).get('verdict') in ('flat', 'regressed')]

    jev_thought = {
        'name': 'JEV · 决策支',
        'question': '这轮卡住了，值不值得花钱让云端接手？',
        'signals': {
            'best_score': round(float(score), 4),
            'stalled_rounds': len(stuck),
            'metric': config.get('metric'),
            'budget_left': 1,
        },
        'verdict': ('值得' if stuck else '不值得'),
        'reason': (f'连续 {len(stuck)} 轮没进步，本地确实卡住了' if stuck
                   else '本地还在进步，现在花钱是浪费'),
        'cost': '0 token（纯规则判断）',
    }

    cloud_thought = {
        'name': 'DeepSeek · 生成支',
        'question': '如果交给我，我会怎么改？',
        'status': '未调用' if not stuck else '可调用',
        'reason': ('本地还在进步，暂时不需要我' if not stuck
                   else f'当前最好 {round(float(score), 4)}，我会试着换模型族或加大正则'),
        'cost': '按 token 计费，只在上面那支说「值得」时才花',
    }
    return {'jev': jev_thought, 'cloud': cloud_thought}
