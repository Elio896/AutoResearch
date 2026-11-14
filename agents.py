"""五角色：Manager / Analyst / Scientist / Experimenter / Reviewer。

设计原则（跟设计文档对齐）：
1. **能算的用代码算**：Analyst 的画像是 pandas 统计，Manager 的预算判据是算术，
   Reviewer 的优劣判断看分数差——都不消耗模型能力。
2. **只有"提方案"和"解释原因"用 LLM**：Scientist 提假设，Reviewer 解释失败。
3. **每个角色只做一件事**，输入输出都是普通 dict，方便单独测。

与 orchestrator.py 的关系：orchestrator 是单 Agent 循环（已跑通，52 测试护着）；
本模块是五角色拆分版，先独立，验证稳定后再替换 orchestrator 的主循环。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field


# ---------------- 预算 ----------------

@dataclass
class Budget:
    """硬预算。任何一项用尽都必须停，不允许"再试一次"。"""
    max_experiments: int = 5
    max_seconds: float = 300.0
    max_local_calls: int = 6
    max_cloud_calls: int = 1
    max_local_tokens: int = 20000

    def snapshot(self, used: dict | None = None) -> dict:
        used = used or {}
        return {
            'experiments_left': self.max_experiments - int(used.get('experiments', 0)),
            'seconds_left': round(max(0.0, self.max_seconds - float(used.get('seconds', 0.0))), 1),
            'local_calls_left': self.max_local_calls - int(used.get('local_calls', 0)),
            'cloud_calls_left': self.max_cloud_calls - int(used.get('cloud_calls', 0)),
            'local_tokens_left': self.max_local_tokens - int(used.get('local_tokens', 0)),
        }


# ---------------- Analyst（确定性） ----------------

class Analyst:
    """读数据画像。全部是统计，不调模型。

    输出里最重要的两件事：
      - `imbalanced`：决定 baseline 要不要 class_weight
      - `metric_hint`：小样本/不平衡时提醒该看哪个指标
    """

    IMBALANCE_THRESHOLD = 0.2
    SMALL_SAMPLE_ROWS = 1000

    def profile(self, planning_context: dict) -> dict:
        counts = planning_context.get('class_counts') or []
        total = sum(int(c.get('count', 0)) for c in counts)
        minority = min((int(c.get('count', 0)) for c in counts), default=total)
        frac = (minority / total) if total else None
        rows = int(planning_context.get('rows', 0) or 0)

        risks = []
        if frac is not None and frac < self.IMBALANCE_THRESHOLD:
            risks.append('class_imbalance')
        if rows and rows < self.SMALL_SAMPLE_ROWS:
            risks.append('small_sample')
        if float(planning_context.get('missing_fraction', 0) or 0) > 0.1:
            risks.append('missing_values')
        if int(planning_context.get('constant_features', 0) or 0) > 0:
            risks.append('constant_features')

        return {
            'rows': rows,
            'features': int(planning_context.get('features', 0) or 0),
            'class_count': int(planning_context.get('class_count', 0) or 0),
            'minority_fraction': round(frac, 4) if frac is not None else None,
            'imbalanced': bool(frac is not None and frac < self.IMBALANCE_THRESHOLD),
            'risks': risks,
            'selection_metric': planning_context.get('selection_metric'),
            # 确定性建议，不给模型猜
            'baseline_class_weight': 'balanced' if (frac is not None and frac < self.IMBALANCE_THRESHOLD) else None,
        }


# ---------------- Manager（规则） ----------------

class Manager:
    """决定预算怎么分、什么时候停。全部是算术，不调模型。"""

    def __init__(self, budget: Budget, patience: int = 2):
        self.budget = budget
        self.patience = patience

    def initial_plan(self, profile: dict) -> dict:
        """开局决定 baseline 用什么。"""
        params = {}
        if profile.get('baseline_class_weight'):
            params['class_weight'] = profile['baseline_class_weight']
        return {'model': 'logistic_regression', 'parameters': params}

    def should_stop(self, state: dict) -> tuple[bool, str]:
        """返回 (该不该停, 原因)。原因要能写进 ledger。"""
        snap = self.budget.snapshot(state)
        if snap['experiments_left'] <= 0:
            return True, 'experiment_budget'
        if snap['seconds_left'] <= 0:
            return True, 'time_budget'
        if snap['local_calls_left'] <= 0:
            return True, 'local_call_budget'
        if state.get('no_improvement', 0) >= self.patience:
            return True, 'validation_plateau'
        return False, ''

    def should_escalate(self, state: dict, profile: dict) -> tuple[bool, str]:
        """要不要花钱调云端。这是赛题专门要看的那条。"""
        snap = self.budget.snapshot(state)
        if snap['cloud_calls_left'] <= 0:
            return False, 'no_cloud_budget'
        # 只有在本地确实卡住时才考虑花钱
        stalled, why = self.should_stop(state)
        if not stalled:
            return False, 'local_still_progressing'
        if why in ('experiment_budget', 'time_budget'):
            return False, 'budget_exhausted_no_point'
        # 平台期 + 还有钱 + 数据不太小（小数据调模型意义不大）
        if why == 'validation_plateau' and (profile.get('rows') or 0) >= 500:
            return True, 'plateau_on_nontrivial_data'
        return False, 'not_worth_it'

    def split_budget(self) -> dict:
        """把总预算按用途切块，防止一类用途吃光。"""
        return {
            'baseline': 0.3,
            'explore': 0.4,
            'diagnose': 0.2,
            'reserve': 0.1,
        }


# ---------------- Scientist（LLM） ----------------

class Scientist:
    """提假设 + 方案。唯一真正依赖模型判断的角色。

    约束：输出必须是 {model, parameters}，参数范围由 allowed 卡死。
    """

    def __init__(self, proposer, allowed: dict, max_output_tokens: int = 200):
        self.proposer = proposer
        self.allowed = allowed
        self.max_output_tokens = max_output_tokens
        # 最近一次调用的 token 消耗（来源：响应里的 prompt_eval_count / eval_count）
        self.last_tokens = {'prompt': 0, 'completion': 0}

    def _allowed_text(self) -> str:
        parts = []
        for model, spec in self.allowed.items():
            params = ', '.join(f'{k} ({v[0]}..{v[1]})' for k, v in spec.items())
            parts.append(f'{model}: {params}')
        return '; '.join(parts)

    def propose(self, history: list, profile: dict, feedback: str = '',
                advice: list | None = None, timeout: float = 30) -> dict:
        advice_text = ('\nPrior research experience (advisory only, still needs verification):\n'
                       + '\n'.join(advice)) if advice else ''
        messages = [
            {'role': 'system', 'content':
             'Propose ONE classifier experiment. Return only a JSON object. '
             'Required fields: model, parameters, hypothesis (one sentence: what you believe and why), '
             'expected_direction (up/down/flat), falsifier (what result would prove you wrong), '
             'confidence_before (a NUMBER from 0 to 1 - how sure you are THIS experiment beats the '
             'current best; 0.9 = very sure, 0.5 = coin flip, 0.2 = wild guess. Never null, never a word). '
             'No explanation outside JSON. No test evaluation. Never repeat an experiment.'},
            {'role': 'user', 'content':
             f'Allowed: {self._allowed_text()}\n'
             f'Data profile: {json.dumps(profile)}\n'
             f'Validation history: {json.dumps(history[-5:])}\n'
             'Guidance: try an untried model family before more tuning of the same one. '
             'For class imbalance, class_weight=balanced usually helps. '
             'Example of a complete answer: '
             '{"model":"hist_gradient_boosting","parameters":{"max_depth":4},'
             '"hypothesis":"shallow trees control variance on this small sample",'
             '"expected_direction":"up","falsifier":"validation F1 does not improve",'
             '"confidence_before":0.55}\n'
             f'Last rejection: {feedback or "(none)"}'
             f'{advice_text}'},
        ]
        response = self.proposer(messages, timeout=timeout)
        # 本地 Ollama 返回 dict（带 prompt_eval_count），云端适配器可能直接回字符串
        if isinstance(response, str):
            response = {'message': {'content': response}}
        # 先记 token：即使解析失败，这次调用的成本也已经发生了。
        self.last_tokens = {
            'prompt': int(response.get('prompt_eval_count', 0) or 0),
            'completion': int(response.get('eval_count', 0) or 0),
        }
        return self.parse(response)

    def parse(self, response: dict) -> dict:
        """把模型输出变成合法 proposal，越界就抛 ValueError（由调用方记一笔）。

        轨迹规范第 03 项要求每轮带「假设 + 可证伪预测」，所以
        hypothesis / expected_direction / falsifier 是必填——没有它们的提案不算研究。
        """
        content = (response.get('message') or {}).get('content') or ''
        if len(content) > 4000:
            raise ValueError('Proposal too long.')
        try:
            proposal = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError('Not valid JSON.') from None
        required = {'model', 'parameters', 'hypothesis', 'expected_direction', 'falsifier'}
        optional = {'confidence_before'}
        allowed_keys = required | optional
        if not isinstance(proposal, dict):
            raise ValueError('Proposal must be a JSON object.')
        missing = required - set(proposal)
        if missing:
            raise ValueError(f'Missing required fields: {sorted(missing)}.')
        extra = set(proposal) - allowed_keys
        if extra:
            raise ValueError(f'Unexpected fields: {sorted(extra)}.')

        hypothesis = proposal['hypothesis']
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError('hypothesis must be a non-empty string.')
        if len(hypothesis) > 300:
            raise ValueError('hypothesis too long (max 300 chars).')
        direction = proposal['expected_direction']
        if direction not in ('up', 'down', 'flat'):
            raise ValueError('expected_direction must be up / down / flat.')
        falsifier = proposal['falsifier']
        if not isinstance(falsifier, str) or not falsifier.strip():
            raise ValueError('falsifier must be a non-empty string.')
        if len(falsifier) > 300:
            raise ValueError('falsifier too long (max 300 chars).')
        confidence = proposal.get('confidence_before')
        if confidence is not None:
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
                raise ValueError('confidence_before must be numeric if given.')
            confidence = float(confidence)
            # 模型常把置信度当百分比给（75 而不是 0.75），这里归一化。
            # 改的是单位，不是数值——75% 就是 0.75。
            if 1.0 < confidence <= 100.0:
                confidence = confidence / 100.0
            if not (0.0 <= confidence <= 1.0):
                raise ValueError('confidence_before must be within [0, 1] (or 0..100 percent).')

        model = proposal['model']
        if model not in self.allowed:
            raise ValueError(f'Unsupported model: {model}.')
        params = proposal['parameters']
        if not isinstance(params, dict):
            raise ValueError('parameters must be an object.')
        unknown = set(params) - set(self.allowed[model])
        if unknown:
            raise ValueError(f'Parameters not allowed for {model}: {sorted(unknown)}.')
        for key, value in params.items():
            low, high = self.allowed[model][key]
            if key == 'class_weight':
                if value not in (None, 'balanced'):
                    raise ValueError('class_weight must be null or "balanced".')
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f'{key} must be numeric.')
            if not math.isfinite(value) or not (low <= value <= high):
                raise ValueError(f'{key} outside [{low}, {high}].')
        return {
            'model': model,
            'parameters': params,
            'hypothesis': hypothesis.strip(),
            'expected_direction': direction,
            'falsifier': falsifier.strip(),
            'confidence_before': float(confidence) if confidence is not None else None,
        }


# ---------------- Experimenter（执行） ----------------

class Experimenter:
    """跑实验。薄封装，真正的训练在 executor 里，这里不碰模型。"""

    def __init__(self, session):
        self.session = session
        self.count = 0

    def run(self, proposal: dict) -> dict:
        # 只把 model/parameters 交给执行器；假设类字段是轨迹用的，执行器不认识。
        result = self.session.run(model=proposal['model'], parameters=proposal['parameters'])
        self.count += 1
        result['sequence'] = self.count
        # 把假设一起挂到结果上，方便落轨迹
        for key in ('hypothesis', 'expected_direction', 'falsifier', 'confidence_before'):
            if key in proposal:
                result[key] = proposal[key]
        return result


# ---------------- Reviewer（规则 + 可选 LLM） ----------------

class Reviewer:
    """判好坏。**规则优先**：分数说话的就不问模型。

    只有在"分数没动但可能是别的原因"时才考虑让模型解释——而且解释不影响判定。
    """

    def __init__(self, min_delta: float = 0.0001, explainer=None):
        self.min_delta = min_delta
        self.explainer = explainer

    def judge(self, candidate: dict, best: dict, profile: dict) -> dict:
        cand = float(candidate.get('validation_score', candidate.get('validation_f1', 0.0)))
        base = float(best.get('validation_score', best.get('validation_f1', 0.0)))
        delta = cand - base

        if delta > self.min_delta:
            verdict = 'improved'
        elif delta < -self.min_delta:
            verdict = 'regressed'
        else:
            verdict = 'flat'

        # 数据泄漏红旗：分数异常高 + 数据量小，先质疑再采信
        suspicious = bool(
            verdict == 'improved'
            and cand > 0.99
            and (profile.get('rows') or 0) < 5000
        )
        if suspicious:
            verdict = 'suspicious_improvement'

        # 轨迹规范第 07 项：闸门要逐条记 pass/fail，红线要单独列，回滚要留痕
        gates = [
            {'name': 'no_regression', 'pass': delta >= -self.min_delta,
             'detail': f'delta={delta:+.6f}'},
            {'name': 'measurable_gain', 'pass': delta > self.min_delta,
             'detail': f'min_delta={self.min_delta}'},
            {'name': 'not_redline', 'pass': not suspicious,
             'detail': 'score>0.99 on small data' if suspicious else 'ok'},
        ]
        redlines = ['score_jump_too_good'] if suspicious else []
        # 回滚不等于失败：被回滚的轮次仍是证据（"这条假设被否证了"）
        reverted = verdict in ('regressed', 'suspicious_improvement', 'flat')

        return {
            'verdict': verdict,
            'delta': round(delta, 6),
            'candidate_score': cand,
            'best_score': base,
            'suspicious': suspicious,
            'reason_code': self._reason(verdict, delta),
            'gates': gates,
            'redlines': redlines,
            'reverted': reverted,
            'restored_from': None,      # 回滚目标版本，由轨迹层填
        }

    def _reason(self, verdict: str, delta: float) -> str:
        if verdict == 'improved':
            return 'score_up'
        if verdict == 'regressed':
            return 'score_down'
        if verdict == 'suspicious_improvement':
            return 'score_jump_too_good'
        return 'no_change'

    def explain(self, verdict: dict, history: list, timeout: float = 20) -> str:
        """可选：让模型用一句话解释，但不参与判定。"""
        if self.explainer is None:
            return ''
        try:
            response = self.explainer([
                {'role': 'system', 'content': 'You explain experiment result changes in one short sentence. No advice, no numbers guessing.'},
                {'role': 'user', 'content': f'Verdict: {json.dumps(verdict)}\nRecent: {json.dumps(history[-3:])}'},
            ], timeout=timeout)
            return ((response.get('message') or {}).get('content') or '').strip()[:300]
        except Exception:
            return ''
