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
from experience import ExperienceStore, advice_lines
from trajectory import TrajectoryWriter, canonical_json


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
                 trajectory_dir: str | Path | None = None, run_id: str | None = None):
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
            if matched:
                self._record('experience_matched', count=len(matched),
                             ids=[r['id'] for r in matched])

        # 3) Experimenter 跑 baseline
        best = self._run_experiment(plan)
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
            if accepted:
                best, no_improvement = result, 0
                self._best_round_id = self.trajectory.last_round_id or self._best_round_id
            else:
                no_improvement += 1

            # 可疑的"大幅提升"：不采信，但要留下证据
            if verdict['suspicious']:
                self._record('reviewer_flag', reason='suspicious_improvement',
                             score=verdict['candidate_score'])

        return self._finish('completed', stop_reason, started)

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

        previous_parameters = (baseline or {}).get('parameters') or {}
        candidate_parameters = result.get('parameters') or {}
        diff = []
        if baseline_round:
            diff.append({'op': 'add', 'field': 'model',
                         'from': None, 'to': result.get('model')})
            for key, value_before in candidate_parameters.items():
                diff.append({'op': 'add', 'field': f'parameters.{key}',
                             'from': None, 'to': value_before})
        else:
            previous_model = (baseline or {}).get('model')
            if previous_model != result.get('model'):
                diff.append({'op': 'change', 'field': 'model',
                             'from': previous_model, 'to': result.get('model')})
            for key in sorted(set(previous_parameters) | set(candidate_parameters)):
                before, after = previous_parameters.get(key), candidate_parameters.get(key)
                if key not in previous_parameters:
                    diff.append({'op': 'add', 'field': f'parameters.{key}', 'to': after})
                elif key not in candidate_parameters:
                    diff.append({'op': 'remove', 'field': f'parameters.{key}', 'from': before})
                elif before != after:
                    diff.append({'op': 'change', 'field': f'parameters.{key}',
                                 'from': before, 'to': after})
        if not diff:
            diff.append({'op': 'change', 'field': 'proposal',
                         'from': None, 'to': {'model': result.get('model'),
                                              'parameters': candidate_parameters}})
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
            'intervention': {'kind': intervention_kind, 'diff': diff},
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
        self._record('finish', summary=summary)
        self._learn(results, best, results[0] if results else None)
        return {'summary': summary, 'events': self.events, 'experiments': results}

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
