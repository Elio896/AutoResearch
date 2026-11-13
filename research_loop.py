"""五角色研究闭环。

    Analyst  → 画数据像（确定性）
    Manager  → 定 baseline、分预算、决定停/升级（算术）
    Scientist→ 提方案（唯一依赖模型的判断）
    Experimenter → 跑实验（executor）
    Reviewer → 判好坏（规则优先，模型只解释）

设计上刻意让"决定"和"执行"分离：任何一步都能单独重放、单独测。
"""
from __future__ import annotations

import json
import hashlib
import importlib.metadata
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agents import Analyst, Budget, Experimenter, Manager, Reviewer, Scientist
from evoundo import VersionChain
from redteam import BlueTeam, RedTeam
from experience import ExperienceStore, advice_lines
from trajectory import TrajectoryWriter, canonical_json, diff_proposals


# 参数范围：既是给 Scientist 的约束，也是 Reviewer 的事后检查依据
DEFAULT_ALLOWED = {
    'logistic_regression': {
        'C': (0.001, 100.0),
        'max_iter': (100.0, 2000.0),
        'class_weight': (None, None),   # 特殊：只允许 null / "balanced"
    },
    'random_forest': {
        'n_estimators': (10.0, 200.0),
        'max_depth': (1.0, 20.0),
        'min_samples_leaf': (1.0, 20.0),
        'class_weight': (None, None),
    },
    'hist_gradient_boosting': {
        'max_iter': (10.0, 500.0),
        'max_depth': (2.0, 20.0),
        'min_samples_leaf': (1.0, 20.0),
        'learning_rate': (0.01, 0.5),
        'class_weight': (None, None),
    },
}


class _TrajectoryUnavailable(RuntimeError):
    """The current executor cannot provide evidence required by the round schema."""


class ResearchLoop:
    def __init__(self, session, proposer=None, budget: Budget | None = None,
                 allowed: dict | None = None, patience: int = 2, emit=None,
                 experience: ExperienceStore | None = None, task_name: str = '',
                 trajectory_dir: str | Path | None = None, run_id: str | None = None,
                 small_proposer=None, cloud_proposer=None, jev_gate=None,
                 max_cloud_calls: int = 0, qwen_router=None, jev_harness=None):
        self.session = session
        self.proposer = proposer
        self.budget = budget or Budget()
        self.allowed = allowed or DEFAULT_ALLOWED
        self.analyst = Analyst()
        self.manager = Manager(self.budget, patience=patience)
        self.reviewer = Reviewer(explainer=proposer)
        self.experimenter = Experimenter(session)
        self.scientist = Scientist(proposer, self.allowed) if proposer else None
        self.emit = emit              # 回调，用于把事件接到 UI
        self.events = []
        self.experience = experience
        self.task_name = task_name or 'task'
        # EvoUndo：每次实验都是一个版本，退步就回滚到历史最优
        self.chain = VersionChain(run_id='')
        # 红蓝对抗：结束前质疑一次最优结果（可关）
        # 红队可以用更小的模型——找茬比提方案简单，这是成本分层的第一刀
        self.redteam = RedTeam(small_proposer or proposer) if (small_proposer or proposer) else None
        # 云端兜底：本地卡住时，由 JEV 判值不值得花钱，值得才升级
        self.cloud_scientist = Scientist(cloud_proposer, self.allowed) if cloud_proposer else None
        self.jev_gate = jev_gate
        self.max_cloud_calls = max_cloud_calls
        # 千问路由：本地小模型自己决定「要不要调云端、调哪个档」
        self.qwen_router = qwen_router
        # Jev Harness：五道零 token 的语义门，先筛掉不值得上云的情况
        self.jev_harness = jev_harness
        self._recent_rejections: list[str] = []
        self._last_gain = None
        self.blueteam = BlueTeam()
        self.run_id = run_id or uuid.uuid4().hex
        self.trajectory = TrajectoryWriter(
            trajectory_dir or Path('artifacts') / 'trajectory' / self.run_id,
            self.run_id,
        )
        self._best_round_id = None
        self.used = {'experiments': 0, 'seconds': 0.0, 'local_calls': 0,
                     'cloud_calls': 0, 'local_tokens': 0}

    # ---------- 事件 ----------
    def _record(self, kind, **values):
        event = {'event': kind, 't': round(time.time(), 3), **values}
        self.events.append(event)
        if self.emit:
            try:
                self.emit(event)
            except Exception:
                pass                        # UI 挂了不影响研究

    def _spend(self, **deltas):
        for key, value in deltas.items():
            self.used[key] = self.used.get(key, 0) + value

    # ---------- 主流程 ----------
    def run(self) -> dict:
        started = time.monotonic()

        # 1) Analyst
        context = self.session.planning_context() if hasattr(self.session, 'planning_context') else {}
        profile = self.analyst.profile(context)
        self.profile = profile
        self._record('analyst_profile', profile=profile)

        # 2) Manager 定 baseline（不平衡时直接 balanced，不让模型猜）
        plan = self.manager.initial_plan(profile)
        self._record('manager_plan', plan=plan, split=self.manager.split_budget(),
                     budget=self.budget.snapshot(self.used), tokens={'prompt': 0, 'completion': 0})

        # 2.5) 跨任务经验：只作为"先试哪个"的建议，仍需实验验证
        advice = []
        if self.experience is not None:
            matched = self.experience.match(profile)
            advice = advice_lines(matched)
            self._advice_ids = [r['id'] for r in matched]
            if matched:
                self._record('experience_matched', count=len(matched),
                             ids=self._advice_ids)

        # 3) Experimenter 跑 baseline
        best = self._run_experiment(plan)
        # baseline 也是一个版本（v1），漏了它版本链就少一代
        if best is not None:
            self.chain.add(
                proposal={'model': best.get('model'), 'parameters': best.get('parameters') or {}},
                score=float(best.get('validation_score', best.get('validation_f1', 0.0))),
                note='baseline')
        if best is None:
            return self._finish('failed', 'baseline_failed', started)
        self._persist_round(best, plan, started, baseline_round=True)

        no_improvement = 0
        stop_reason = ''

        # 4) 循环：Manager 判停 → Scientist 提 → Experimenter 跑 → Reviewer 判
        while True:
            self.used['seconds'] = time.monotonic() - started
            stop, reason = self.manager.should_stop({**self.used, 'no_improvement': no_improvement})
            if stop:
                # 本地卡在平台期时，先问 JEV「这次花钱值不值」，值得才升级
                if reason == 'validation_plateau' and self.cloud_scientist is not None:
                    if self._try_escalate(best, profile, started, no_improvement):
                        no_improvement = 0
                        continue
                stop_reason = reason
                break

            if self.scientist is None:
                stop_reason = 'no_proposer'
                break

            # Scientist
            history = [{'model': e['model'], 'parameters': e['parameters'],
                        'score': e.get('validation_score', e.get('validation_f1'))}
                       for e in self._experiments]
            remaining = self.budget.max_seconds - (time.monotonic() - started)
            self._spend(local_calls=1)
            try:
                proposal = self.scientist.propose(history, profile, advice=advice,
                                                  timeout=min(45.0, max(1.0, remaining)))
            except ValueError as exc:
                # 方案不合法也要记成本：这次调用真的发生了
                self._spend(local_tokens=sum(self.scientist.last_tokens.values()))
                self._record('scientist_rejected', reason=str(exc),
                             tokens=self.scientist.last_tokens.copy())
                # 注意：**不**增加 no_improvement。
                # 模型输出格式不对 ≠ 研究方向停滞，混在一起会让 patience 提前触发。
                # 防死循环靠 local_calls 预算，不靠 patience。
                continue
            except Exception as exc:                    # 模型端点挂了
                self._record('scientist_error', reason=str(exc)[:200])
                stop_reason = 'local_model_unavailable'
                break

            self._spend(local_tokens=sum(self.scientist.last_tokens.values()))
            self._record('scientist_proposal', proposal=proposal,
                         tokens=self.scientist.last_tokens.copy(),
                         budget=self.budget.snapshot(self.used))

            # Experimenter
            result = self._run_experiment(proposal)
            if result is None:
                # 数据不兼容导致跑不起来，同样不算研究停滞（有 local_calls 兜底）
                continue

            # Reviewer
            verdict = self.reviewer.judge(result, best, profile)
            self._record('reviewer_verdict', verdict=verdict,
                         budget=self.budget.snapshot(self.used),
                         tokens=self.scientist.last_tokens.copy())
            self._persist_round(result, proposal, started, verdict=verdict,
                                baseline=best)

            accepted = verdict['verdict'] in ('improved', 'kept') and not verdict.get('reverted', False)
            # EvoUndo：把这次实验登记成一个版本
            version = self.chain.add(
                proposal={'model': result.get('model'), 'parameters': result.get('parameters') or {}},
                score=float(result.get('validation_score', result.get('validation_f1', 0.0))),
                note=verdict['verdict'])
            if accepted:
                best, no_improvement = result, 0
                self._best_round_id = self.trajectory.last_round_id or self._best_round_id
            else:
                no_improvement += 1
                # 退步就真的回到最优那一代（不重训、不删版本）
                target = self.chain.best()
                if target is not None and target.version_id != self.chain.current_id:
                    self.chain.rollback(target.version_id, reason=f"{verdict['verdict']}: delta={verdict['delta']}")
                    self._record('rollback', from_version=version.version_id,
                                 to_version=target.version_id,
                                 reason=verdict['verdict'], delta=verdict['delta'])

            # 可疑的"大幅提升"：不采信，但要留下证据
            if verdict['suspicious']:
                self._record('reviewer_flag', reason='suspicious_improvement',
                             score=verdict['candidate_score'])

        # 收工前让红队试一次：这个"最优"经不经得起换 seed 的检验
        self._adversarial_check(best, profile)
        return self._finish('completed', stop_reason, started)

    def _try_escalate(self, best: dict | None, profile: dict, started: float,
                      no_improvement: int = 0) -> bool:
        """本地卡在平台期。先问 JEV 值不值得花钱，值得才让云端出一个方案。

        返回 True 表示"真的升级并跑了一次实验"（调用方继续循环），
        False 表示没升级（调用方按原计划收工）。
        """
        if best is None:
            return False
        if self.used['cloud_calls'] >= self.max_cloud_calls:
            self._record('escalation_blocked', reason='cloud_budget',
                         used=self.used['cloud_calls'], cap=self.max_cloud_calls)
            return False

        # 0) Jev Harness：五道零 token 的门，先筛掉根本不值得上云的情况。
        #    这才是省 token 的地方——它自己一次模型都不调。
        if self.jev_harness is not None:
            harness_ctx = {
                'task': self.task_name,
                'rows': int(profile.get('rows', 0) or 0),
                'no_improvement': int(no_improvement),
                'last_gain': self._last_gain,
                'recent_rejections': list(self._recent_rejections[-4:]),
                'recent_proposals': [{'model': r.get('model'), 'parameters': r.get('parameters')}
                                     for r in self._experiments[-3:]],
            }
            harness_decision = self.jev_harness.decide(harness_ctx)
            self._record('jev_harness_decision', **harness_decision.to_dict())
            if harness_decision.action != 'escalate':
                return False

        # 1a) 千问路由：本地小模型先判「要不要调、调哪个档」
        if self.qwen_router is not None:
            router_state = {
                'task': self.task_name,
                'trigger': 'validation_plateau',
                'experiments_used': self.used['experiments'],
                'best_validation_score': float(best.get('validation_score', best.get('validation_f1', 0.0))),
                'recent_scores': [round(float(r.get('validation_score', r.get('validation_f1', 0.0))), 4)
                                  for r in self._experiments[-4:]],
                'remaining_cloud_calls': self.max_cloud_calls - self.used['cloud_calls'],
                'rows': int(profile.get('rows', 0) or 0),
            }
            try:
                decision = self.qwen_router(router_state)
            except Exception as exc:                    # noqa: BLE001
                self._record('qwen_router_error', reason=str(exc)[:200])
                decision = {'need_cloud': False, 'reason': 'router_exception'}
            self._record('qwen_router_decision',
                         need_cloud=decision.get('need_cloud'),
                         model=decision.get('model'),
                         reason=decision.get('reason'),
                         source=decision.get('source'))
            if not decision.get('need_cloud'):
                return False

        # 1b) JEV 判据：这一步本身就是"该不该花云端钱"的证据
        if self.jev_gate is not None:
            state = {
                'task': self.task_name,
                'trigger': 'validation_plateau',
                'experiments_used': self.used['experiments'],
                'local_calls_used': self.used['local_calls'],
                'best_validation_score': float(best.get('validation_score', best.get('validation_f1', 0.0))),
                'recent_scores': [round(float(r.get('validation_score', r.get('validation_f1', 0.0))), 4)
                                  for r in self._experiments[-4:]],
                'remaining_cloud_calls': self.max_cloud_calls - self.used['cloud_calls'],
            }
            try:
                decision = self.jev_gate(state, timeout=max(5.0, min(45.0, self.budget.max_seconds - (time.monotonic() - started))))
            except Exception as exc:                    # noqa: BLE001
                self._record('jev_error', reason=str(exc)[:200])
                return False
            allow = bool(decision.get('allow_cloud'))
            self._record('jev_decision', allow_cloud=allow,
                         source=decision.get('source'),
                         probability=decision.get('probability'),
                         threshold=decision.get('threshold'),
                         model=decision.get('model'),
                         usage=decision.get('usage') or {})
            if not allow:
                return False

        # 2) 真的调云端提一个方案
        history = [{'model': r.get('model'), 'parameters': r.get('parameters'),
                    'score': r.get('validation_score', r.get('validation_f1'))}
                   for r in self._experiments[-5:]]
        remaining = self.budget.max_seconds - (time.monotonic() - started)
        self._spend(cloud_calls=1)
        try:
            proposal = self.cloud_scientist.propose(history, profile,
                                                    timeout=max(5.0, min(60.0, remaining)))
        except Exception as exc:                        # noqa: BLE001
            self._record('cloud_error', reason=str(exc)[:200])
            return False

        self._record('cloud_proposal', proposal=proposal,
                     tokens=self.cloud_scientist.last_tokens.copy())
        result = self._run_experiment(proposal)
        if result is None:
            return False

        verdict = self.reviewer.judge(result, best, profile)
        self._record('reviewer_verdict', verdict=verdict,
                     budget=self.budget.snapshot(self.used), source='cloud')
        version = self.chain.add(
            proposal={'model': result.get('model'), 'parameters': result.get('parameters') or {}},
            score=float(result.get('validation_score', result.get('validation_f1', 0.0))),
            note='cloud:' + verdict['verdict'])
        if verdict['verdict'] in ('improved', 'kept') and not verdict.get('reverted', False):
            self._record('escalation_success', from_score=verdict['best_score'],
                         to_score=verdict['candidate_score'], delta=verdict['delta'])
            return True
        target = self.chain.best()
        if target is not None and target.version_id != self.chain.current_id:
            self.chain.rollback(target.version_id, reason='cloud_attempt_regressed')
            self._record('rollback', from_version=version.version_id,
                         to_version=target.version_id, reason='cloud_attempt_regressed')
        return False

    def _adversarial_check(self, best: dict | None, profile: dict) -> None:
        """红队质疑 → 蓝队用代码检验 → 成立就回滚。全程不让模型判好坏。"""
        if self.redteam is None or best is None:
            return
        history = [{'model': r.get('model'), 'score': r.get('validation_score', r.get('validation_f1'))}
                   for r in getattr(self, '_results', [])[-4:]]
        remaining = max(5.0, self.budget.max_seconds - self.used['seconds'])
        try:
            claims, rejected = self.redteam.propose(profile, best, history, timeout=min(45.0, remaining))
        except Exception as exc:                        # noqa: BLE001
            self._record('redteam_error', reason=str(exc)[:200])
            return
        for item in rejected:
            self._record('redteam_rejected', **item)
        if not claims:
            return

        proposal = {'model': best.get('model'), 'parameters': best.get('parameters') or {}}
        session = self.session
        # 检验器需要重新跑实验，把最优方案挂到 session 上供它们取
        try:
            setattr(session, 'best_proposal', proposal)
        except Exception:                               # noqa: BLE001
            pass

        verdicts = []
        for claim in claims:
            self._record('redteam_claim', **claim)
            verdict = self.blueteam.verify(claim, session)
            verdicts.append(verdict)
            self._record('blueteam_verdict', claim=claim['claim'],
                         check_kind=claim['check_kind'],
                         supported=verdict.get('supported'),
                         measured=verdict.get('measured'),
                         threshold=verdict.get('threshold'),
                         detail=verdict.get('detail') or verdict.get('reason'))

        summary = self.blueteam.summarize(verdicts)
        self._record('adversarial_summary', **summary)
        if summary.get('should_rollback'):
            target = self.chain.best()
            if target is not None:
                if target.version_id == self.chain.current_id:
                    # 当前就是最优、前面没有更可信的一代可回。
                    # 如实记「跳过」，并把它标成可疑——不许假装回滚成功。
                    self._record('rollback_skipped', reason='no_prior_version',
                                 current=target.version_id,
                                 claim=summary.get('reason', '')[:160])
                    target.note = (target.note + ' | ' if target.note else '') + 'redteam_suspect'
                else:
                    self.chain.rollback(target.version_id, reason='redteam: ' + str(summary.get('reason', ''))[:80])
                    self._record('rollback', from_version=self.chain.current_id,
                                 to_version=target.version_id, reason='redteam')

    # ---------- 子步骤 ----------
    @property
    def _experiments(self):
        return [e for e in getattr(self, '_results', [])]

    def _run_experiment(self, proposal: dict) -> dict | None:
        started = time.monotonic()
        self._round_started_at = datetime.now(timezone.utc)
        try:
            result = self.experimenter.run(proposal)
        except ValueError as exc:
            self._record('experiment_rejected', proposal=proposal, reason=str(exc)[:200])
            return None
        except Exception as exc:
            self._record('experiment_failed', proposal=proposal, reason=str(exc)[:200])
            return None
        if not hasattr(self, '_results'):
            self._results = []
        self._results.append(result)
        self._spend(experiments=1, seconds=time.monotonic() - started)
        self._record('experiment', model=result.get('model'),
                     parameters=result.get('parameters'),
                     score=result.get('validation_score', result.get('validation_f1')),
                     duration=round(time.monotonic() - started, 3))
        return result

    def _persist_round(self, result: dict, proposal: dict, run_started: float,
                       verdict: dict | None = None, baseline: dict | None = None,
                       baseline_round: bool = False) -> None:
        try:
            record = self._build_round_record(
                result, proposal, run_started, verdict, baseline, baseline_round)
        except _TrajectoryUnavailable as exc:
            self._record('trajectory_unavailable', reason=str(exc),
                         experiment_id=result.get('experiment_id'))
            return
        saved = self.trajectory.append(record)
        if baseline_round:
            self._best_round_id = saved['round_id']
        self._record('round_record_written', round=saved['round'],
                     round_id=saved['round_id'], hash=saved['chain']['hash'],
                     budget=saved['budget'], hypothesis=saved['hypothesis'],
                     verdict=saved['conclusion']['verdict'])

    def _build_round_record(self, result: dict, proposal: dict, run_started: float,
                            verdict: dict | None, baseline: dict | None,
                            baseline_round: bool) -> dict:
        session = self.session
        dataset_hash = getattr(session, 'dataset_sha256', None) or result.get('dataset_sha256')
        if not isinstance(dataset_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', dataset_hash):
            raise _TrajectoryUnavailable('session does not expose the original dataset SHA-256')
        x, y = getattr(session, 'x', None), getattr(session, 'y', None)
        train_indices = getattr(session, 'train_indices', None)
        validation_indices = getattr(session, 'validation_indices', None)
        test_indices = getattr(session, 'test_indices', None)
        if x is None or y is None or any(
                indices is None for indices in (train_indices, validation_indices, test_indices)):
            raise _TrajectoryUnavailable('session does not expose the fixed train/validation/test split')

        import numpy as np
        from sklearn.metrics import confusion_matrix

        training_y = y.iloc[train_indices]
        class_counts = training_y.value_counts().sort_index()
        if len(class_counts) != 2:
            raise _TrajectoryUnavailable('the round schema only represents binary confusion matrices')
        classes = [int(value) for value in class_counts.index]
        experiment_id = result.get('experiment_id')
        experiment = getattr(session, 'experiments', {}).get(experiment_id)
        if not experiment:
            raise _TrajectoryUnavailable('session does not retain the validation-only fitted estimator')
        pipeline = experiment[0]
        actual = y.iloc[validation_indices]
        predicted = pipeline.predict(x.iloc[validation_indices])
        matrix = confusion_matrix(actual, predicted, labels=classes)
        if matrix.shape != (2, 2):
            raise _TrajectoryUnavailable('validation confusion matrix is not binary')
        tn, fp, fn, tp = (int(value) for value in matrix.ravel())

        class_metrics = result.get('validation_class_metrics') or []
        class_f1 = [float(item['f1']) for item in class_metrics if 'f1' in item]
        if len(class_f1) < 2:
            raise _TrajectoryUnavailable('validation class-wise F1 values are unavailable')
        value = float(result.get('validation_f1', result.get('validation_score')))
        previous_result = baseline or result
        baseline_value = float(previous_result.get('validation_f1', previous_result.get('validation_score')))
        delta = value - baseline_value
        current_best_id = self._best_round_id
        reverted = bool(verdict and verdict.get(
            'reverted', verdict.get('verdict') not in ('improved', 'kept')))
        status = 'reverted' if reverted else 'kept'
        if baseline_round:
            status, reverted, current_best_id = 'kept', False, None

        seed = int(getattr(session, 'seed', 42))
        split_identity = {
            'dataset_sha256': dataset_hash,
            'seed': seed,
            'train': [int(i) for i in train_indices],
            'validation': [int(i) for i in validation_indices],
            'test': [int(i) for i in test_indices],
        }
        split_hash = 'sha256:' + hashlib.sha256(canonical_json(split_identity)).hexdigest()
        positive_id = getattr(session, 'positive_id', None)
        if positive_id not in classes:
            positive_id = int(class_counts.idxmin())
        pos_rate = float(class_counts.loc[positive_id] / class_counts.sum())
        train_x = x.iloc[train_indices]
        missing_rate = {
            str(column): float(rate)
            for column, rate in train_x.isna().mean().items()
            if float(rate) > 0
        }
        profile = {
            'class_counts': {str(key): int(count) for key, count in class_counts.items()},
            'missing_rate': missing_rate,
            'pos_rate_basis': 'training minority class fraction' if getattr(session, 'positive_id', None) is None
                              else 'configured positive class fraction',
            'target_column_recorded': bool(getattr(session, 'target', None)),
            'metric_note': 'headline f1 is macro validation F1; selection objective is also retained in secondary',
        }

        round_number = self.trajectory.next_round
        slug = re.sub(r'[^a-z0-9_]+', '_', str(result.get('model', 'experiment')).lower()).strip('_')
        round_id = f'r{round_number:03d}_{slug or "experiment"}'
        now = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
        expected = proposal.get('expected_direction', 'flat')
        if expected not in ('up', 'down', 'flat'):
            expected = 'flat'
        hypothesis_text = proposal.get('hypothesis')
        falsifier = proposal.get('falsifier')
        if baseline_round:
            hypothesis_text = 'Establish a deterministic baseline for later validation comparisons.'
            falsifier = 'Reject the baseline if execution or validation checks fail.'
            expected = 'flat'
        if not isinstance(hypothesis_text, str) or not hypothesis_text.strip():
            hypothesis_text = 'Evaluate this configuration against the current validation baseline.'
        if not isinstance(falsifier, str) or len(falsifier.strip()) < 10:
            falsifier = 'Reject the hypothesis when the measured validation result contradicts it.'
        confidence = proposal.get('confidence_before')
        confidence = float(confidence) if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else 0.5

        candidate_parameters = result.get('parameters') or {}
        current_proposal = {'model': result.get('model'), 'parameters': candidate_parameters}
        previous_proposal = (
            {'model': (baseline or {}).get('model'),
             'parameters': (baseline or {}).get('parameters') or {}}
            if not baseline_round else None
        )
        diff = diff_proposals(previous_proposal, current_proposal)
        intervention_kind = 'baseline' if baseline_round else (
            'model' if (baseline or {}).get('model') != result.get('model') else 'hyperparameter')

        gates = []
        if verdict:
            for gate in verdict.get('gates', []):
                gates.append({
                    'id': str(gate.get('id', gate.get('name', 'reviewer_gate'))),
                    'pass': bool(gate.get('pass')),
                    'detail': str(gate.get('detail', '')),
                })
        if not gates:
            gates.append({'id': 'execution_succeeded', 'pass': result.get('status') == 'success',
                          'detail': 'The validation-only experiment returned successfully.'})
        gates.append({
            'id': 'fold_and_ci_evidence',
            'pass': False,
            'detail': ('The current executor uses one fixed holdout: per_fold stores observed '
                       'per-class F1 for schema compatibility, and ci95 is the conservative [0,1] '
                       'metric range, not a statistical confidence interval.'),
        })
        gates.append({
            'id': 'cost_reconciliation',
            'pass': False,
            'detail': 'Provider billing is not instrumented; cost_usd is recorded as 0.0 placeholder.',
        })

        token_usage = getattr(getattr(self, 'scientist', None), 'last_tokens', {}) if not baseline_round else {}
        tokens_in = max(0, int(token_usage.get('prompt', 0) or 0))
        tokens_out = max(0, int(token_usage.get('completion', 0) or 0))
        total_tokens = tokens_in + tokens_out
        elapsed = max(0.0, time.monotonic() - run_started)
        selection_score = float(result.get('validation_score', value))
        selection_baseline = float((baseline or result).get('validation_score', baseline_value))
        secondary = {
            'validation_accuracy': float(result.get('validation_accuracy', 0.0)),
            'selection_score': selection_score,
            'selection_baseline': selection_baseline,
            'selection_delta': selection_score - selection_baseline,
        }
        if result.get('validation_minority_precision') is not None:
            secondary['minority_precision'] = float(result['validation_minority_precision'])
        if result.get('validation_minority_recall') is not None:
            secondary['minority_recall'] = float(result['validation_minority_recall'])
        restored_from = current_best_id if reverted else None
        review_name = verdict.get('verdict', 'baseline') if verdict else 'baseline'
        reason = (f'Reviewer {review_name}; macro validation F1 delta {delta:+.6f}; '
                  f'round status {status}.')
        next_hypothesis = (verdict or {}).get('next_hypothesis')
        if not isinstance(next_hypothesis, str) or not next_hypothesis.strip():
            next_hypothesis = ('Test an untried model family against the validation baseline.'
                               if status == 'kept' else
                               'Revise one model parameter and state a falsifier before retrying.')

        command = ('ResearchSession.run(model=' + json.dumps(result.get('model')) +
                   ', parameters=' + json.dumps(candidate_parameters, sort_keys=True) + ')')
        env_material = {
            'python': sys.version,
            'platform': platform.platform(),
            'sklearn': importlib.metadata.version('scikit-learn'),
            'numpy': importlib.metadata.version('numpy'),
        }
        env_hash = 'sha256:' + hashlib.sha256(canonical_json(env_material)).hexdigest()
        try:
            # git 不一定在 PATH 里（这台机器就是装在 Program Files），先找再用
            git_exe = shutil.which('git') or r'C:\Program Files\Git\cmd\git.exe'
            commit = subprocess.run(
                [git_exe, 'rev-parse', 'HEAD'], cwd=Path(__file__).parent,
                check=True, capture_output=True, text=True, timeout=2,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            commit = 'unknown-uncommitted'

        return {
            'schema_version': '1.0',
            'round': round_number,
            'round_id': round_id,
            'run_id': self.run_id,
            'parent_round_id': self.trajectory.last_round_id,
            'started_at': getattr(self, '_round_started_at', datetime.now(timezone.utc)).isoformat(
                timespec='milliseconds'),
            'ended_at': datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
            'status': status,
            'revision': 1,
            'input': {
                'dataset': getattr(session, 'dataset_name', '') or
                           (self.task_name if self.task_name != 'task' else f'dataset:{dataset_hash[:12]}'),
                'sha256': f'sha256:{dataset_hash}',
                'rows': len(y),
                'cols': len(x.columns) + 1,
                'target': str(getattr(session, 'target', '') or ''),
                'pos_rate': pos_rate,
                'profile': profile,
                'split': {
                    'scheme': 'holdout',
                    'k': 2,
                    'seed': seed,
                    'split_hash': split_hash,
                },
            },
            'hypothesis': {
                'text': hypothesis_text.strip(),
                'expected_direction': expected,
                'falsifier': falsifier.strip(),
                'confidence_before': confidence,
                'authored_by': 'Manager baseline' if baseline_round else 'Scientist',
            },
            'intervention': {
                'kind': intervention_kind,
                'proposal': proposal,
                'diff': diff,
            },
            'exec': {
                'command': command,
                'code_commit': commit,
                'env_hash': env_hash,
                'exit_code': 0 if result.get('status') == 'success' else 1,
                'duration_s': max(0.0, float(result.get('duration_seconds', 0.0))),
                'log_path': '',
                'stage': 'experiment_only' if baseline_round else 'with_llm',
            },
            'result': {
                'metric': 'f1',
                'value': value,
                'per_fold': class_f1,
                'std': float(np.std(class_f1)),
                'ci95': [0.0, 1.0],
                'baseline_value': baseline_value,
                'delta': delta,
                'mde': 1.0,
                'delta_significant': False,
                'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
                'secondary': secondary,
            },
            'validation': {
                'gates': gates,
                'redlines': list((verdict or {}).get('redlines', [])),
                'reverted': reverted,
                'restored_from': restored_from,
            },
            'budget': {
                'tokens_in': tokens_in,
                'tokens_out': tokens_out,
                'cost_usd': 0.0,
                'wall_s': elapsed,
                'experiments_used': 1,
                'cumulative': {
                    'tokens': int(self.used['local_tokens']),
                    'wall_s': elapsed,
                    'experiments': int(self.used['experiments']),
                },
                'cap': {
                    'tokens': max(1, int(self.budget.max_local_tokens)),
                    'wall_s': max(1.0, float(self.budget.max_seconds)),
                    'experiments': max(1, int(self.budget.max_experiments)),
                },
                'reconciled_with': {
                    'source': 'ResearchLoop in-process token counters',
                    'tokens': int(self.used['local_tokens']),
                    'diff': 0,
                },
            },
            'conclusion': {
                'verdict': status,
                'reason': reason,
                'next_hypothesis': next_hypothesis,
            },
            'ext': {
                'adapter': 'ResearchLoop fixed-holdout schema adapter',
                'limitations': [
                    'No cross-validation folds or statistical confidence interval are produced by the current executor.',
                    'Provider billing, target-column name, and run log are not available in the current runtime.',
                ],
            },
        }

    def _finish(self, status: str, reason: str, started: float) -> dict:
        results = getattr(self, '_results', [])
        best = max(results, key=lambda r: r.get('validation_score', r.get('validation_f1', 0))) if results else None
        summary = {
            'status': status,
            'stop_reason': reason,
            'elapsed_seconds': round(time.monotonic() - started, 2),
            'experiments': len(results),
            'best_score': best.get('validation_score', best.get('validation_f1')) if best else None,
            'best_model': best.get('model') if best else None,
            'best_parameters': best.get('parameters') if best else None,
            'local_calls': self.used['local_calls'],
            'cloud_calls': self.used['cloud_calls'],
            'local_tokens': self.used['local_tokens'],
            'test_evaluated': False,
            'trajectory_run_id': self.run_id,
            'trajectory_dir': str(self.trajectory.run_dir.resolve()),
        }
        # EvoUndo：版本树落盘（JSON 给程序看，HTML 给人看）
        if getattr(self, 'chain', None) is not None and self.chain.versions:
            self.chain.run_id = self.chain.run_id or self.run_id
            evo_dir = Path('artifacts') / 'evoundo' / self.chain.run_id
            self.chain.dump(evo_dir / 'versions.json')
            self.chain.render_html(evo_dir / 'versions.html')
            summary['evoundo_dir'] = str(evo_dir.resolve())
            summary['evoundo_versions'] = len(self.chain.versions)
            summary['evoundo_rollbacks'] = self.chain.to_dict()['rollbacks']
            summary['evoundo_current'] = self.chain.current_id
            summary['evoundo_best'] = self.chain.best().version_id if self.chain.best() else None
        self._record('finish', summary=summary)
        self._learn(results, best, results[0] if results else None)
        self._settle_advice(results, best)
        return {'summary': summary, 'events': self.events, 'experiments': results}

    def _settle_advice(self, results: list, best: dict | None) -> None:
        """给这次用过的经验记账：它到底管不管用。

        这是"自进化"闭环的最后一步——经验不是写进去就完了，
        每次被用之后都要知道自己有没有帮上忙，不然经验库只会膨胀不会进化。
        """
        if self.experience is None or not getattr(self, '_advice_ids', None):
            return
        if not results or best is None:
            return
        baseline = results[0]
        base = float(baseline.get('validation_score', baseline.get('validation_f1', 0.0)))
        top = float(best.get('validation_score', best.get('validation_f1', 0.0)))
        improved = top > base + 0.01          # 至少比 baseline 高 1 个点才算"经验帮上忙"
        for record_id in self._advice_ids:
            updated = self.experience.record_usage(record_id, improved)
            if updated is not None:
                self._record('experience_settled', id=record_id, improved=improved,
                             used_count=updated.get('used_count'),
                             success_count=updated.get('success_count'),
                             contradicted_count=updated.get('contradicted_count'))

    def _learn(self, results: list, best: dict | None, baseline: dict | None) -> None:
        """把这一轮的发现沉淀成经验。

        只在"真的赢过 baseline 且领先幅度够大"时才记，避免把噪声写成知识。
        """
        if self.experience is None or not best or not baseline or best is baseline:
            return
        base = float(baseline.get('validation_score', baseline.get('validation_f1', 0.0)))
        top = float(best.get('validation_score', best.get('validation_f1', 0.0)))
        delta = top - base
        if delta <= 0.01:
            return
        profile = getattr(self, 'profile', {}) or {}
        condition = {}
        for key in ('imbalanced', 'class_count'):
            if key in profile and profile[key] not in (None, '', [], {}):
                condition[key] = profile[key]
        if not condition:
            return
        params = best.get('parameters') or {}
        advice = (f'在 {condition} 的数据上，{best.get("model")} '
                  f'（{json.dumps(params, ensure_ascii=False, sort_keys=True)}）'
                  f'比 baseline 高 {delta:+.3f}，优先试它')
        try:
            self.experience.add(
                condition=condition,
                advice=advice,
                evidence={'task': self.task_name, 'baseline': round(base, 4),
                          'best': round(top, 4), 'delta': round(delta, 4),
                          'experiments': len(results)},
                source_task=self.task_name,
            )
            self._record('experience_learned', advice=advice)
        except ValueError as exc:
            self._record('experience_rejected', reason=str(exc))
