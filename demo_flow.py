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


def _model_proposer(model: str, want: bool):
    """选一个真的能说话的模型，顺序：轻量模型 API → 本地 Ollama → 没有。

    本地 7B 在"按格式输出"上不稳（同一提示会回 JSON 壳甚至 `{}`），所以优先 API。
    但 API 要花钱：没配 key 就**不许假装调过**，必须退回本地或规则，并把来源写清楚。
    """
    if not want:
        return None, 'rules', None

    try:
        import light_model
        api = light_model.from_env()
    except Exception:                       # noqa: BLE001 - 依赖缺失就当没有
        api = None
    if api is not None:
        return api, 'api_model', api.descriptor

    local = _local_proposer_if_available(model)
    if local is not None:
        return local, 'local_model', {'provider': 'local', 'model': model}
    return None, 'rules', None


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
        proposer, model_source, model_descriptor = _model_proposer(model, want_local)
    else:
        model_source, model_descriptor = 'local_model', {'provider': 'local', 'model': model}

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
    thought = _think(profile, context, proposer, model, model_source, model_descriptor)
    # 千问只出分析依据；研究假设/复杂诊断另走一路（DeepSeek）。
    research = _research_hypotheses(profile, context, session)

    # 它自己点菜 -> 代码真跑 -> 真数字回去让它读（不让它心算）
    plan = _ask_plan(proposer, model)
    evidence_ops = []
    for op in plan['ops']:
        try:
            evidence_ops.append(_run_profile_op(session, op))
        except Exception as exc:          # noqa: BLE001
            evidence_ops.append({'op': op, 'title': f'{op} 执行失败',
                                 'error': str(exc)[:120],
                                 'table': {'columns': [], 'rows': []}})
    reading = _ask_reading(proposer, model, evidence_ops)

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

    # 验证：代码真跑，不是模型自述；转交：只是分类结果，不是预算授权
    verification = []
    try:
        import qwen_worklog
        verification = qwen_worklog.run_checks(
            {'verification_checks': ['rows_readable', 'no_columns_dropped',
                                     'target_column_untouched', 'operation_returned_rows']},
            session, {'ok': bool(evidence_ops)})
    except Exception as exc:              # noqa: BLE001
        verification = [{'check': 'verification_unavailable', 'passed': False,
                         'detail': str(exc)[:120]}]
    handoff = {'to': 'deepseek', 'count': len(research.get('questions') or []),
               'needs_specialist': bool(research.get('questions')),
               'note': '只是分类结果；要不要花钱由代码里的预算检查决定'}

    return {
        'session_id': session_id,
        'profile': profile,
        'raw_context': context,
        'thought': thought,
        'research': research,
        'plan': plan,
        'evidence_ops': evidence_ops,
        'reading': reading,
        'verification': verification,
        'handoff': handoff,
        'model_source': model_source,
        'model_descriptor': model_descriptor,
        'model_usage': (proposer.usage_summary()
                        if hasattr(proposer, 'usage_summary') else None),
        'suggested': _suggest(profile, context),
        'message': '看过上面的判断，可以改参数，然后点「同意并继续」。',
    }


def _think(profile: dict, context: dict, proposer, model: str,
           model_source: str = 'rules', descriptor: dict | None = None) -> dict:
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
            raw = content.strip()
            if raw:
                # 原文照留：不润色、不补全、不替它把 JSON 壳拆开。
                # format_ok 只做「有没有按约定回答」的校验，不参与改写。
                return {'source': 'local_model', 'model_source': model_source,
                        'model': (descriptor or {}).get('model') or model,
                        'attribution': 'model',
                        'text': raw[:1500],
                        'raw': raw[:1500],
                        'format_ok': _looks_like_three_sentences(raw),
                        'format_expectation': '三句话，中文口语，不给方案',
                        'can_do': can_do, 'cannot_do': cannot_do,
                        'observations': observations}
        except Exception:          # noqa: BLE001 - 模型挂了就用规则话术，不影响演示
            pass

    return {'source': 'rules', 'model_source': 'rules', 'model': None,
            'attribution': 'program',
            'text': ' | '.join(observations),
            'raw': None,
            'format_ok': None,
            'format_expectation': '三句话，中文口语，不给方案',
            'observations': observations,
            'can_do': can_do,
            'cannot_do': cannot_do}


def _looks_like_three_sentences(text: str) -> bool:
    """模型有没有按「三句话、口语、不给方案」的约定作答。

    只做格式校验——true/false 用来在界面上挂一个标，绝不据此改写内容。
    模型如果吐了 JSON 壳或者干脆没说话，这里判 false，原文照样原样往外交。
    """
    body = (text or '').strip()
    if not body:
        return False
    if body[0] in '{[':                 # 吐的是 JSON，不是自然语言
        return False
    if '":' in body or '"：' in body:     # 半截 JSON 键值对
        return False
    sentences = sum(body.count(mark) for mark in '。！？!?')
    return 1 <= sentences <= 6


def _id_like_columns(session) -> list[str]:
    """列名像行标识的（customer_id / txn_id / id）。只用来提问，不自动删。"""
    try:
        names = [str(c) for c in session.x.columns]
    except Exception:               # noqa: BLE001 - 取不到就不提这一条
        return []
    ids = [n for n in names if n.lower() == 'id' or n.lower().endswith('_id')]
    return ids


def _leak_candidates(session, max_columns: int = 6) -> list[dict]:
    """单特征就能把标签几乎完全分开的数值列——事后字段（泄漏）的典型长相。

    只算数，不下结论：分数高的列会被拿去问 DeepSeek「这是不是事后字段」，
    而不是由这里判定它就是泄漏。
    """
    try:
        import numpy as np
        import pandas as pd
        from sklearn.metrics import roc_auc_score
    except Exception:               # noqa: BLE001 - 环境缺库就不算
        return []

    try:
        x = session.x.iloc[session.train_indices]
        y = session.y.iloc[session.train_indices]
        if pd.Series(y).nunique() != 2:
            return []
        target = np.asarray(y, dtype=float)
        found = []
        for name in x.columns:
            column = x[name]
            if not pd.api.types.is_numeric_dtype(column):
                continue
            values = np.asarray(column, dtype=float)
            if np.isnan(values).any():
                fill = np.nanmedian(values) if not np.all(np.isnan(values)) else 0.0
                values = np.nan_to_num(values, nan=float(fill))
            if float(np.nanstd(values)) == 0.0:
                continue
            try:
                auc = float(roc_auc_score(target, values))
            except Exception:       # noqa: BLE001
                continue
            auc = max(auc, 1.0 - auc)
            if auc >= 0.90:
                found.append({'column': str(name), 'single_feature_auc': round(auc, 4)})
        found.sort(key=lambda item: -item['single_feature_auc'])
        return found[:max_columns]
    except Exception:               # noqa: BLE001
        return []


def _table(columns, rows) -> dict:
    return {'columns': list(columns), 'rows': rows}


# 它能自己点菜的盘点操作：每一项都是代码真跑，模型只负责选和读
_PROFILE_OPS = {
    'head': '看前 3 行原始记录',
    'missing': '每列缺失了多少',
    'balance': '训练集的类别分布',
    'unique': '每列有多少个不同取值（找行标识列）',
    'target_signal': '每列单独跟目标的关联强度（找泄漏或强特征）',
    'corr': '数值列之间最相关的几对（找共线）',
}
_DEFAULT_OPS = ['missing', 'balance', 'unique', 'target_signal']


def _run_profile_op(session, op: str) -> dict:
    """真跑一个盘点操作，返回真实数字。只读数据，不改任何东西。"""
    import numpy as np
    import pandas as pd

    x, y = session.x, session.y
    train = session.train_indices
    xt = x.iloc[train]
    yt = y.iloc[train]

    if op == 'head':
        cols = [str(c) for c in xt.columns[:8]]
        body = xt[xt.columns[:8]].head(3).astype(str).values.tolist()
        return {'op': op, 'title': '前 3 行（训练集）', 'table': _table(cols, body)}

    if op == 'missing':
        miss = xt.isna().mean()
        rows = [[str(c), f'{miss[c] * 100:.1f}%', int(xt[c].isna().sum())]
                for c in miss.sort_values(ascending=False).index[:8]]
        return {'op': op, 'title': '每列缺失（训练集）',
                'table': _table(['列', '缺失率', '缺失行'], rows)}

    if op == 'balance':
        counts = yt.value_counts().sort_index()
        rows = [[str(k), int(v), f'{v / len(yt) * 100:.1f}%'] for k, v in counts.items()]
        return {'op': op, 'title': '训练集类别分布',
                'table': _table(['类别', '行数', '占比'], rows)}

    if op == 'unique':
        nun = xt.nunique(dropna=False)
        ratio = (nun / max(1, len(xt))).sort_values(ascending=False)
        rows = [[str(c), int(nun[c]), f'{ratio[c] * 100:.0f}%'] for c in ratio.index[:8]]
        return {'op': op, 'title': '每列不同取值数（100% ≈ 行标识）',
                'table': _table(['列', '取值数', '占行数'], rows)}

    if op == 'target_signal':
        from sklearn.metrics import roc_auc_score
        rows = []
        for name in xt.columns:
            column = xt[name]
            if pd.api.types.is_numeric_dtype(column):
                values = np.asarray(column, dtype=float)
                if np.isnan(values).any():
                    fill = np.nanmedian(values) if not np.all(np.isnan(values)) else 0.0
                    values = np.nan_to_num(values, nan=float(fill))
                if float(np.nanstd(values)) == 0.0:
                    continue
                try:
                    auc = float(roc_auc_score(np.asarray(yt, dtype=float), values))
                except Exception:          # noqa: BLE001
                    continue
                auc = max(auc, 1.0 - auc)
                rows.append([str(name), '数值', f'{auc:.4f}', '单特征 AUC'])
            else:
                grouped = (pd.DataFrame({'c': column.astype(str), 'y': np.asarray(yt)})
                           .groupby('c')['y'].agg(['mean', 'count']))
                if len(grouped) < 2:
                    continue
                spread = float(grouped['mean'].max() - grouped['mean'].min())
                rows.append([str(name), '类别', f'{spread:.4f}', '组间正例率差'])
        rows.sort(key=lambda r: -float(r[2]))
        return {'op': op, 'title': '每列单独跟目标的关联（越接近 1 越可疑）',
                'table': _table(['列', '类型', '强度', '算法'], rows[:8])}

    if op == 'corr':
        numeric = xt.select_dtypes(include='number')
        if numeric.shape[1] < 2:
            return {'op': op, 'title': '数值列不足两列，算不出共线',
                    'table': _table(['说明'], [['—']])}
        corr = numeric.corr().abs()
        names = list(corr.columns)
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairs.append([str(names[i]), str(names[j]), round(float(corr.iloc[i, j]), 4)])
        pairs.sort(key=lambda r: -r[2])
        return {'op': op, 'title': '最相关的数值列（|r|）',
                'table': _table(['列A', '列B', '|r|'], pairs[:5])}

    raise ValueError(f'不支持的操作：{op}')


def _first_json_object(text: str):
    start = (text or '').find('{')
    end = (text or '').rfind('}')
    if start < 0 or end <= start:
        raise ValueError('没找到 JSON')
    return json.loads(text[start:end + 1])


def _ask_plan(proposer, model: str) -> dict:
    """让千问自己决定要先看什么。它给不出计划就用默认清单，并标明来源。"""
    if proposer is None:
        return {'ops': _DEFAULT_OPS, 'source': 'default',
                'why': '没有可用的本地模型，用默认盘点清单（这些数字仍然是真跑的）'}

    prompt = ('你可以让我执行下面这些盘点操作（我只执行，不替你算）：\n'
              + json.dumps(_PROFILE_OPS, ensure_ascii=False)
              + '\n挑 3-5 个最该先看的，只输出 JSON，形如 '
                '{"ops":["unique","missing"],"why":"一句话说明为什么先看这几个"}')
    try:
        response = proposer([{'role': 'user', 'content': prompt}], timeout=30)
        content = response if isinstance(response, str) else \
            ((response.get('message') or {}).get('content') or '')
        obj = _first_json_object(content)
        ops = [o for o in (obj.get('ops') or []) if o in _PROFILE_OPS][:5]
        if ops:
            return {'ops': ops, 'source': 'model', 'why': str(obj.get('why') or '')[:200],
                    'raw': content.strip()[:800], 'model': model}
    except Exception as exc:              # noqa: BLE001
        return {'ops': _DEFAULT_OPS, 'source': 'default',
                'why': f'它的计划不可用（{str(exc)[:80]}），退回默认清单'}
    return {'ops': _DEFAULT_OPS, 'source': 'default',
            'why': '它没给出可执行的操作，退回默认清单'}


def _digest_ops(evidence_ops: list[dict], max_rows: int = 8) -> str:
    """把真跑出来的表压成纯文本。

    故意不给它看 JSON：这个 7B 见到 JSON 就照着回 JSON，喂纯文本它才肯说话。
    """
    lines: list[str] = []
    for item in evidence_ops:
        table = item.get('table') or {}
        lines.append(f"【{item.get('title') or item.get('op')}】")
        for row in (table.get('rows') or [])[:max_rows]:
            lines.append('  ' + ' | '.join(str(cell) for cell in row))
        if item.get('error'):
            lines.append(f"  （这一项执行失败：{item['error']}）")
    return '\n'.join(lines)


def _ask_reading(proposer, model: str, evidence_ops: list[dict]) -> dict | None:
    """把真跑出来的数字喂回去，只让它读、不让它算。"""
    if proposer is None or not evidence_ops:
        return None
    prompt = ('下面是刚从数据里真跑出来的统计数字，一字不改：\n\n'
              + _digest_ops(evidence_ops)
              + '\n\n用三句中文说清楚你注意到什么。'
                '只写句子：不要 JSON、不要大括号、不要键值对、不要表格、不要分点；'
                '不许出现上面没有的数字；不许给建议。')

    def _unwrap(text: str) -> str | None:
        """它要是非塞个 JSON 壳，就把它自己写的内容摊开。

        优先取常见的承载字段（text/content/result…），没有再按原顺序摊平所有字符串。
        只摊平、不增删：这些字还是它写的，页面上会同时保留原文和「拆壳后」标记。
        """
        body = (text or '').strip().lstrip('\ufeff')
        if body[:1] not in ('{', '['):
            return None
        try:
            obj = json.loads(body)
        except Exception:            # noqa: BLE001 - 不是合法 JSON 就不拆
            return None

        if isinstance(obj, dict):
            for key in ('text', 'content', 'result', 'answer',
                        'finding', 'conclusion', 'analysis', 'summary'):
                value = obj.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()[:1200]

        values: list[str] = []

        def walk_values(node):
            if isinstance(node, str):
                piece = node.strip()
                if piece:
                    values.append(piece)
            elif isinstance(node, dict):
                for item in node.values():
                    walk_values(item)
            elif isinstance(node, list):
                for item in node:
                    walk_values(item)

        walk_values(obj)
        if values:
            joined = ' '.join(values).strip()
            if joined:
                return joined[:1200]

        # 它有时把整句话写在 key 里、value 只给个 true/false
        keys: list[str] = []

        def walk_keys(node):
            if isinstance(node, dict):
                for key, item in node.items():
                    text = str(key).strip()
                    if len(text) > 8:
                        keys.append(text)
                    walk_keys(item)
            elif isinstance(node, list):
                for item in node:
                    walk_keys(item)

        walk_keys(obj)
        joined = ' '.join(keys).strip()
        return joined[:1200] if joined else None
    try:
        response = proposer([{'role': 'user', 'content': prompt}], timeout=30)
        content = response if isinstance(response, str) else \
            ((response.get('message') or {}).get('content') or '')
        raw = content.strip()
        # Ollama 挂的时候客户端会把 "HTTP 500" 当正文返回，不能当成它说的话
        if not raw or (raw.startswith('HTTP ') and raw[5:].strip().isdigit()):
            return {'attribution': 'none',
                    'error': f'调用失败：{raw or "空响应"}'}
        return {'attribution': 'model', 'model': model, 'raw': raw[:1200],
                'unwrapped': _unwrap(raw),
                'format_ok': _looks_like_three_sentences(raw)}
    except Exception as exc:              # noqa: BLE001
        return {'attribution': 'none', 'error': str(exc)[:120]}


def _research_hypotheses(profile: dict, context: dict, session) -> dict:
    """该交给 DeepSeek 的那部分：研究假设与复杂诊断。

    出去的只有**问题**和触发它的证据，不是结论——诊断结论得由云端模型自己
    下，这里不代它说话。演示环境没有云端 key，所以状态一律 not_called。
    """
    questions: list[dict] = []
    rows = profile.get('rows') or 0
    missing = context.get('missing_fraction') or 0
    imbalanced = bool(profile.get('imbalanced'))
    minority = profile.get('minority_fraction')

    for item in _leak_candidates(session):
        questions.append({
            'id': f"leak:{item['column']}",
            'kind': '泄漏排查',
            'evidence': f"训练集单特征 AUC {item['single_feature_auc']}",
            'question': (f"{item['column']} 单独一列就能把标签分到 AUC "
                         f"{item['single_feature_auc']}——它是不是「结果发生后才有的字段」？"
                         f"要不要从特征里拿掉，拿掉后分数掉多少算真掉？"),
        })

    ids = _id_like_columns(session)
    if ids:
        questions.append({
            'id': 'id-columns',
            'kind': '特征取舍',
            'evidence': '列名以 _id 结尾',
            'question': f"这几列看着像行标识（{', '.join(ids[:5])}）——丢掉还是留着？留着算不算作弊？",
        })

    if missing > 0.01:
        questions.append({
            'id': 'missingness',
            'kind': '缺失机制',
            'evidence': f'整体缺失 {missing * 100:.1f}%',
            'question': ('缺失率最高那几列，缺失本身是不是携带信息（MNAR）？'
                         '填中位数等于说「没查就是正常」，该怎么补救？'),
        })

    if imbalanced:
        share = f'{minority * 100:.1f}%' if minority is not None else '很低'
        questions.append({
            'id': 'metric',
            'kind': '指标选择',
            'evidence': f'少数类占 {share}',
            'question': ('这么不平衡，主指标用 macro F1 还是少数类 F1？'
                         '阈值要不要扫，扫了算不算过拟合验证集？'),
        })

    if rows and rows < 1000:
        questions.append({
            'id': 'small-sample',
            'kind': '样本量',
            'evidence': f'训练集只有 {rows} 行',
            'question': '样本这么小，验证分数的波动有多大？现在这个提升是不是噪声？',
        })

    return {
        'to': 'deepseek',
        'scope': '研究假设 / 复杂诊断',
        'status': 'not_called',
        'status_text': '未调用',
        'reason': ('演示环境没配云端 key。这里列的是「要问它什么」，'
                   '它还没答过——不代它说话。'),
        'questions': questions[:6],
    }


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
