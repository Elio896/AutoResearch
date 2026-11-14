"""Competition workflow: shared-model research, paired policy evolution, audit.

Rules mode is a runnable code baseline, not evidence of autonomous LLM ability.
Acceptance tasks are disjoint from development and are never sent to the policy
proposer. Each evolution round uses a fresh acceptance batch. Test labels are
opened once, after all strategy decisions, on a separate final audit task.
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import time
import uuid
from pathlib import Path

from agents import Budget
from candidate_proposer import propose_from_failures
from executor import ResearchSession
from model_budget import ModelLimits, SharedModel
from research_loop import ResearchLoop
from strategy_decision import code_checks
from strategy_eval import compare, freeze_criteria
from strategy_rollback import freeze_rules, rollback_if_needed
from strategy_store import StrategyStore
from corrections import CorrectionGuard


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(payload,encoding='utf-8')
    try:
        for attempt in range(5):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt==4: break
                time.sleep(.025*(attempt+1))
        if path.name=='state.json':
            # Immutable state journal remains writable when Windows blocks replacement.
            journal=path.with_name('state.'+str(time.time_ns())+'.'+uuid.uuid4().hex+'.json')
            temporary.replace(journal)
            return
        raise PermissionError('Unable to replace research artifact after bounded retries.')
    finally:
        temporary.unlink(missing_ok=True)


def read_saved_state(path):
    path=Path(path)
    candidates=[path]+list(path.parent.glob('state.*.json'))
    for candidate in sorted((p for p in candidates if p.exists()),key=lambda p:p.stat().st_mtime_ns,reverse=True):
        try: return json.loads(candidate.read_text(encoding='utf-8'))
        except (OSError,ValueError): continue
    raise FileNotFoundError('No readable run state.')


class ResearchResources:
    def __init__(self, max_fits=80, max_seconds=180):
        self.max_fits, self.max_seconds = max_fits, max_seconds
        self.fits, self.started = 0, time.monotonic()

    def reserve_fit(self):
        if self.fits >= self.max_fits or self.remaining <= 0:
            raise RuntimeError('Research training/time budget exhausted.')
        self.fits += 1

    @property
    def remaining(self):
        return max(0, self.max_seconds - (time.monotonic() - self.started))


class BudgetSession(ResearchSession):
    def __init__(self, *args, resources, **kwargs):
        self.resources = resources
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        self.resources.reserve_fit()
        return super().run(*args, **kwargs)


class RulePlanner:
    """Explicit deterministic comparator: reacts to installed policy, zero LLMs."""
    source = 'deterministic_rule'

    def __init__(self, policy):
        self.policy, self.index = policy, 0
        self.input_bytes = 0

    def __call__(self, messages, timeout=30):
        self.input_bytes += len(json.dumps(messages).encode('utf-8'))
        rules = self.policy.get('candidate_selection_rules', [])
        explore = any(r.get('kind') == 'explore_families' for r in rules)
        choices = [('random_forest', {'n_estimators': 60, 'max_depth': 6}),
                   ('hist_gradient_boosting', {'max_iter': 60, 'max_depth': 3})] if explore else [
                       ('logistic_regression', {'C': 0.1}), ('logistic_regression', {'C': 10.0})]
        model, parameters = choices[self.index % len(choices)]
        self.index += 1
        proposal = {'model': model, 'parameters': parameters,
                    'hypothesis': 'Explore nonlinear boundaries.' if explore else 'Test regularization sensitivity.',
                    'expected_direction': 'up', 'falsifier': 'Validation macro F1 does not exceed incumbent.',
                    'confidence_before': 0.5}
        if 'missing indicators' in self.policy.get('planning_prompt', ''):
            proposal['preprocessing'] = {'add_missing_indicator': True}
        return {'message': {'content': json.dumps(proposal)},
                'prompt_eval_count': 0, 'eval_count': 0}


class ResearchCampaign:
    def __init__(self, directory, *, mode='rules', client=None, rounds=2,
                 max_calls=36, max_tokens=150000, max_seconds=180, max_fits=80,
                 correction_store=None):
        if mode not in ('rules', 'single_model') or type(rounds) is not int or not 1 <= rounds <= 2:
            raise ValueError('Unknown mode or invalid evolution rounds.')
        if mode == 'single_model' and client is None:
            raise ValueError('Single-model mode requires a configured model endpoint.')
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / 'strategy.json').exists():
            raise ValueError('Use a new campaign directory; existing evidence must not be overwritten.')
        self.mode, self.rounds = mode, rounds
        self.correction_store = correction_store
        self.frozen_corrections = [r for r in correction_store.all() if r['status'] == 'active'] if correction_store else []
        if mode == 'single_model' and any(r['kind'] == 'no_paid_calls' and r['scope'] == 'global'
                                          for r in self.frozen_corrections):
            raise ValueError('User correction forbids paid model calls.')
        self.resources = ResearchResources(max_fits, max_seconds)
        self.model = SharedModel(client, ModelLimits(max_calls, max_tokens, max_seconds)) if client else None
        self.store = StrategyStore(self.root / 'strategy.json')
        self.events = []
        self.criteria = freeze_criteria({
            'min_tasks': 2, 'legal_output_rate': {'min_delta': 0},
            'duplicate_rate': {'max_delta': 0}, 'mean_best_score': {'min_delta': -0.005},
            'calls': {'max_ratio': 1.5}, 'model_tokens': {'max_ratio': 1.5},
            'elapsed_seconds': {'max_ratio': 2.0},
            'min_improvement': {'mean_best_score': 0.002, 'duplicate_rate': 0, 'legal_output_rate': 0}})
        self.rollback_rules = freeze_rules()

    def event(self, kind, **details):
        value = {'event': kind, 'elapsed_seconds': round(time.monotonic() - self.resources.started, 6), **details}
        self.events.append(value)
        with (self.root / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')

    def usage(self):
        return self.model.snapshot() if self.model else {
            'calls': 0, 'known_tokens': 0, 'budget_tokens': 0, 'unreconciled_calls': 0,
            'cost_cny': 0, 'cost_basis': 'no_model_calls', 'records': []}

    def dataset(self, name, seed):
        import pandas as pd
        import numpy as np
        from sklearn.datasets import make_moons
        x, y = make_moons(n_samples=600, noise=0.28, random_state=seed)
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame({'signal_a': x[:, 0], 'signal_b': x[:, 1],
                              'noise': rng.normal(size=len(y)), 'target': y})
        path = self.root / 'datasets' / f'{name}.csv'
        path.parent.mkdir(exist_ok=True)
        frame.to_csv(path, index=False)
        return path

    def research(self, strategy_id, task, label, *, target='target'):
        policy = self.store.get(strategy_id)['changes']
        proposer = self.model.for_role('scientist') if self.model else RulePlanner(policy)
        session = BudgetSession(task, target, resources=self.resources, seed=42)
        matched = [r for r in self.frozen_corrections if r['scope'] == 'global' or
                   r['dataset_sha256'] == session.dataset_sha256]
        snapshot = {'rules': matched, 'dataset_sha256': session.dataset_sha256,
                    'digest': hashlib.sha256(json.dumps(matched, sort_keys=True).encode('utf-8')).hexdigest()}
        loop = ResearchLoop(session, proposer=proposer, strategy_store=self.store,
                            strategy_id=strategy_id, run_id=label,
                            trajectory_dir=self.root / 'trajectory' / label,
                            budget=Budget(max_experiments=min(3, max(1, self.resources.max_fits - self.resources.fits)), max_local_calls=2,
                                          max_seconds=max(0.01, self.resources.remaining)), patience=2,
                            correction_store=self.correction_store,
                            correction_snapshot=snapshot if self.correction_store else None)
        # Verification is allocated separately as matched tasks, not unmetered red-team fits.
        loop.redteam = None
        outcome = loop.run()
        outcome['execution_source'] = 'single_model' if self.model else 'deterministic_rule'
        outcome['dataset_sha256'] = session.dataset_sha256
        outcome['summary']['task_id'] = hashlib.sha256(Path(task).read_bytes()).hexdigest()
        outcome['summary']['proposal_attempts'] = outcome['summary']['local_calls']
        save(self.root / 'runs' / f'{label}.json', outcome)
        self.event('research_completed', label=label, strategy_id=strategy_id,
                   best_score=outcome['summary']['best_score'], evidence=f'runs/{label}.json')
        return outcome, session

    def arm(self, strategy_id, tasks, label):
        before = self.usage()
        began = time.monotonic()
        rows = []
        for index, task in enumerate(tasks):
            if self.resources.remaining <= 0 or self.resources.fits >= self.resources.max_fits:
                rows.append({'status': 'budget_exhausted', 'experiments': 0, 'best_score': None,
                             'task_id': hashlib.sha256(Path(task).read_bytes()).hexdigest(),
                             'attempts': 0, 'legal': 0, 'duplicates': 0})
                continue
            result, _ = self.research(strategy_id, task, f'{label}_{index}')
            proposals = [e for e in result['events'] if e['event'] == 'scientist_proposal']
            failures = [e for e in result['events'] if e['event'] in ('scientist_rejected', 'scientist_error')]
            attempts = len(proposals) + len(failures)
            keys = [json.dumps({k: e['proposal'].get(k) for k in ('model', 'parameters', 'preprocessing')}, sort_keys=True)
                    for e in proposals]
            rows.append({**result['summary'], 'attempts': attempts, 'legal': len(proposals),
                         'duplicates': len(keys) - len(set(keys)), 'evidence': f'runs/{label}_{index}.json'})
        after = self.usage()
        calls = after['calls'] - before['calls']
        attempts = sum(r['attempts'] for r in rows)
        unknown = after['unreconciled_calls'] > before['unreconciled_calls']
        arm = {'strategy_id': strategy_id, 'task_count': len(rows),
               'completed': sum(r['status'] == 'completed' and r['experiments'] == 3 for r in rows),
               'task_ids': [r['task_id'] for r in rows],
               'mean_best_score': (sum(r['best_score'] for r in rows) / len(rows)
                                   if all(r['best_score'] is not None for r in rows) else None),
               'legal_output_rate': sum(r['legal'] for r in rows) / attempts if attempts else None,
               'duplicate_rate': sum(r['duplicates'] for r in rows) / attempts if attempts else None,
               'local_calls': 0, 'cloud_calls': calls,
               'model_tokens': None if unknown else after['known_tokens'] - before['known_tokens'],
               'elapsed_seconds': time.monotonic() - began,
               'cost_cny': None if calls else 0, 'cost_basis': 'unknown' if calls else 'no_model_calls',
               'tasks': rows}
        save(self.root / f'{label}.json', arm)
        return arm

    def candidate(self, development, round_number):
        active = self.store.active()
        events = development['events']
        refs = [f'event_{i:03d}' for i, e in enumerate(events) if e['event'] in
                ('experiment', 'reviewer_verdict', 'scientist_rejected')]
        records = [{'failure_type': 'limited_exploration' if round_number == 1 else 'remaining_validation_error',
                    'measurements': {'best_score': development['summary']['best_score']},
                    'event_refs': refs, 'run_id': development['summary']['trajectory_run_id']}]
        if self.model:
            return propose_from_failures(self.store, {'records': records, 'insufficient': not bool(refs)},
                                         self.model.for_role('strategy_diagnosis'),
                                         timeout=max(0.1, min(45, self.resources.remaining)),
                                         proposed_by='single_model')
        if round_number == 1:
            patch = {'candidate_selection_rules': [{'kind': 'explore_families'}, {'kind': 'reject_duplicates'}]}
            what = 'Explore untried model families before more regularization tuning.'
        else:
            rules = active['changes'].get('candidate_selection_rules', [])
            if any(r.get('kind') == 'compact_context' for r in rules):
                raise ValueError('Context policy already installed; no unsupported change.')
            patch = {'candidate_selection_rules': rules + [{'kind': 'compact_context'}]}
            what = 'Compress planning context while preserving validation quality; require measured token savings.'
        return self.store.propose(patch, reason=what, parent_id=active['strategy_id'],
                                  proposed_by='deterministic_rule',
                                  evidence_run_ids=[development['summary']['trajectory_run_id']],
                                  hypothesis={'what': what, 'why': 'Development validation still has errors.',
                                              'expected_improvement': ('Independent validation macro F1 improves > 0.002.' if round_number == 1
                                                                       else 'Known provider tokens decrease with no material validation regression.'),
                                              'falsifier': 'No measured gain or quality/call/token gate failure.',
                                              'side_effects': 'May increase training time.'})

    def run(self, development_csv=None, target='target'):
        import importlib.metadata
        began = time.monotonic()
        active = self.store.bootstrap({'planning_prompt': 'Use validation feedback and a falsifiable hypothesis.',
                                       'candidate_selection_rules': []}, reason='Frozen initial research policy')
        source = Path(__file__).resolve().parent
        source_files = [*source.glob('*.py'), source / 'requirements.txt', source / 'requirements-api.txt',
                        source / 'tools' / 'run_research_system.py']
        source_hashes = {}
        for file in source_files:
            relative = file.relative_to(source)
            destination = self.root / 'source' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, destination)
            source_hashes[relative.as_posix()] = hashlib.sha256(destination.read_bytes()).hexdigest()
        protocol = {'schema': 1, 'run_id': self.root.name, 'mode': self.mode,
                    'python': platform.python_version(),
                    'packages': {n: importlib.metadata.version(n) for n in ('pandas', 'scikit-learn')},
                    'criteria': self.criteria, 'rollback': self.rollback_rules,
                    'max_fits': self.resources.max_fits, 'max_seconds': self.resources.max_seconds,
                    'model_limits': vars(self.model.limits) if self.model else None,
                    'task_family': 'synthetic noisy moons classification',
                    'round_seeds': [[101, 202], [303, 404]], 'canary_seeds': [501, 502],
                    'development_seed': 11, 'final_audit_seed': 999,
                    'development_target': target, 'custom_development_data': bool(development_csv),
                    'source_sha256': source_hashes,
                    'user_corrections': self.frozen_corrections,
                    'model': (getattr(self.model.client, 'descriptor', {'model': getattr(self.model.client, 'model', None)})
                              if self.model else None),
                    'wall_time_policy': 'Do not start work after deadline; an in-flight sklearn fit is not forcibly terminated.'}
        save(self.root / 'protocol.json', protocol)
        if development_csv:
            dev = self.root / 'development-upload.csv'
            if Path(development_csv).resolve() != dev.resolve():
                shutil.copyfile(development_csv, dev)
        else:
            dev = self.dataset('development', 11)
        dev_sha = hashlib.sha256(dev.read_bytes()).hexdigest()
        if self.model and any(r['kind'] == 'no_paid_calls' and (r['scope'] == 'global' or
                              r['dataset_sha256'] == dev_sha) for r in self.frozen_corrections):
            raise ValueError('User correction forbids paid calls for this development dataset.')
        history = []
        for number in range(1, self.rounds + 1):
            if self.resources.remaining <= 0 or self.resources.fits >= self.resources.max_fits:
                self.event('stopped', reason='time_or_fit_budget')
                break
            active = self.store.active()
            development, _ = self.research(active['strategy_id'], dev, f'round{number}_development', target=target)
            entry = {'round': number, 'incumbent': active['strategy_id'],
                     'development_evidence': f'runs/round{number}_development.json'}
            try:
                candidate = self.candidate(development, number)
            except (ValueError, RuntimeError):
                entry.update({'decision': 'keep', 'reason': 'No valid candidate or model budget exhausted.'})
                history.append(entry)
                self.event('candidate_not_applied', **entry)
                continue
            entry['candidate'] = candidate['strategy_id']
            entry['hypothesis'] = candidate.get('hypothesis')
            entry['candidate_source'] = candidate.get('created_by')
            if self.resources.max_fits - self.resources.fits < 12 or self.resources.remaining <= 0:
                self.store.reject(candidate['strategy_id'], 'Insufficient matched evaluation budget.', by='evidence-gate')
                entry.update({'decision': 'keep', 'reason': 'Insufficient matched evaluation budget.'})
                history.append(entry)
                self.event('stopped', reason='matched_evaluation_budget')
                break
            # Generate acceptance data only AFTER candidate is frozen; proposer never sees its metrics.
            seeds = protocol['round_seeds'][number - 1]
            tasks = [self.dataset(f'round{number}_eval_{s}', s) for s in seeds]
            # Counterbalance order across rounds to reduce systematic timing bias.
            ids = [active['strategy_id'], candidate['strategy_id']]
            if number % 2 == 0:
                ids.reverse()
            arms = {sid: self.arm(sid, tasks, f'round{number}_{sid}') for sid in ids}
            report = compare(arms[active['strategy_id']], arms[candidate['strategy_id']], self.criteria)
            save(self.root / f'round{number}_comparison.json', report)
            checks, problems = code_checks(report)
            entry.update({'decision': 'keep', 'comparison': f'round{number}_comparison.json',
                          'score_delta': (report['new']['mean_best_score'] - report['old']['mean_best_score']
                                          if all(report[a]['mean_best_score'] is not None for a in ('old', 'new')) else None),
                          'evaluation': report})
            if not problems and report['verdict'] == 'upgrade':
                # In the competition, one model suffices: deterministic evidence gate, no fake Jev call.
                self.store.activate(candidate['strategy_id'], checks=checks,
                                    jev={'decision': 'upgrade', 'source': 'deterministic_acceptance',
                                         'criteria_digest': self.criteria['digest']},
                                    by='evidence-gate', reason='Independent paired evaluation passed',
                                    expected_active_id=active['strategy_id'])
                entry['decision'] = 'upgrade'
                canary = [self.dataset(f'round{number}_canary', protocol['canary_seeds'][number - 1])]
                baseline = self.arm(active['strategy_id'], canary, f'round{number}_canary_old')
                current = self.arm(candidate['strategy_id'], canary, f'round{number}_canary_new')
                monitor = rollback_if_needed(self.store, current_id=candidate['strategy_id'],
                                             fallback_id=active['strategy_id'], baseline=baseline, current=current,
                                             rules=self.rollback_rules['rules'])
                save(self.root / f'round{number}_canary.json', monitor)
                entry['monitor'] = monitor
                if monitor['applied']:
                    entry['decision'] = 'rollback'
            else:
                self.store.reject(candidate['strategy_id'], report['reason'], by='evidence-gate')
            history.append(entry)
            self.event('evolution_round_completed', round=number, decision=entry['decision'],
                       candidate=candidate['strategy_id'], score_delta=entry['score_delta'])
            save(self.root / 'progress.json', {'rounds': history, 'usage': self.usage()})

        final_audit = None
        if self.resources.remaining > 0 and self.resources.fits + 3 <= self.resources.max_fits:
            audit_csv = self.dataset('final_audit', 999)
            audit, session = self.research(self.store.active()['strategy_id'], audit_csv, 'final_audit')
            winner = session.experiments
            selected = next((key for key, (_, result) in winner.items() if
                             result['model'] == audit['summary']['best_model'] and
                             result['parameters'] == audit['summary']['best_parameters'] and
                             result.get('preprocessing', {}) == audit['summary']['best_preprocessing']), None)
            if selected:
                final_audit = session.finalize(selected)
                save(self.root / 'final_test.json', final_audit)
        manifest = {str(p.relative_to(self.root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (self.root / 'datasets').glob('*.csv')}
        if development_csv:
            manifest['development-upload.csv'] = hashlib.sha256(Path(development_csv).read_bytes()).hexdigest()
        report = {'schema': 1, 'run_id': self.root.name, 'mode': self.mode, 'rounds': history,
                  'active_strategy': self.store.active(), 'model_usage': self.usage(),
                  'training_fits': self.resources.fits, 'elapsed_seconds': time.monotonic() - began,
                  'final_test': final_audit, 'datasets_sha256': manifest,
                  'limitations': ['Rules mode demonstrates code policy adaptation, not LLM self-improvement.',
                                  'Two paired tasks per round are a smoke evaluation, not statistical proof.',
                                  'Acceptance is limited to noisy-moons classification; no cross-domain claim.',
                                  'Provider billing is unknown without an invoice; reservations are not actual tokens.',
                                  'Model budgets are per campaign; use provider account caps for an account-wide limit.']}
        save(self.root / 'report.json', report)
        self.event('campaign_completed', training_fits=self.resources.fits, rounds=len(history))
        return report
