"""Real-training ablation: no memory vs prompt-only vs executable correction.

Proposal replay is deliberately scripted to repeat mistakes. This measures the
system's enforcement and persistence, not the cognitive ability of an LLM.
"""
import argparse
import hashlib
import json
import sys
import time
import shutil
import platform
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import Budget
from corrections import CorrectionStore, fingerprint
from executor import ResearchSession
from research_loop import ResearchLoop
from research_system import save


class ReplayPlanner:
    source = 'deterministic_rule'

    def __init__(self, proposals):
        self.proposals, self.index = proposals, 0

    def __call__(self, messages, timeout=30):
        p = dict(self.proposals[min(self.index, len(self.proposals) - 1)])
        self.index += 1
        p.update(hypothesis='Replay a known failure to test the guard.', expected_direction='up',
                 falsifier='Validation score does not improve.')
        return {'message': {'content': json.dumps(p)}}


def run_case(path, csv, proposals, correction_store=None, prompt_only=None):
    session = ResearchSession(csv, 'target')
    loop = ResearchLoop(session, proposer=ReplayPlanner(proposals),
                        budget=Budget(max_experiments=len(proposals) + 1, max_local_calls=len(proposals)),
                        correction_store=correction_store, trajectory_dir=path / 'trajectory', run_id=path.name)
    loop.redteam = None
    if prompt_only:
        loop.scientist.corrections = prompt_only
    started = time.monotonic()
    result = loop.run()
    result['measured_seconds'] = time.monotonic() - started
    assert not session.closed and result['summary']['test_evaluated'] is False
    save(path / 'run.json', result)
    return result


def evaluate(directory):
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_moons

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'report.json').exists():
        raise ValueError('Do not overwrite a previous evaluation.')
    seeds = (71, 83, 97)
    protocol = {'seeds': seeds, 'arms': ['no_memory', 'prompt_only', 'enforced'],
                'scenarios': ['duplicate_experiment', 'forbidden_model', 'missing_indicator'],
                'source': 'scripted_proposal_replay_real_sklearn_training',
                'metrics': ['executed_mistakes', 'blocked', 'auto_repaired', 'training_fits', 'validation_f1', 'seconds'],
                'acceptance': {'post_feedback_mistakes': 0, 'unrelated_scope_blocks': 0},
                'holdout_usage': 0, 'scope': 'Three machine-checkable correction types; no guarantee for arbitrary language.'}
    project = Path(__file__).resolve().parents[1]
    source_hashes = {}
    for file in [*project.glob('*.py'), Path(__file__).resolve()]:
        relative = file.relative_to(project)
        destination = root / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, destination)
        source_hashes[relative.as_posix()] = hashlib.sha256(destination.read_bytes()).hexdigest()
    protocol['source_sha256'] = source_hashes
    protocol['python'] = platform.python_version()
    save(root / 'protocol.json', protocol)
    cases = []
    for seed in seeds:
        x, y = make_moons(n_samples=400, noise=0.28, random_state=seed)
        frame = pd.DataFrame({'signal_a': x[:, 0], 'signal_b': x[:, 1], 'target': y})
        csv = root / f'data_{seed}.csv'
        frame.to_csv(csv, index=False)
        missing = frame.copy()
        rng = np.random.default_rng(seed)
        missing.loc[rng.random(len(y)) < np.where(y == 1, 0.5, 0.05), 'signal_a'] = np.nan
        missing_csv = root / f'missing_{seed}.csv'
        missing.to_csv(missing_csv, index=False)
        forest = {'model': 'random_forest', 'parameters': {'n_estimators': 30, 'max_depth': 4}}
        scenarios = [
            ('duplicate_experiment', csv, [forest, forest], '不要重复运行相同的实验', 'reject_duplicates'),
            ('forbidden_model', csv, [forest], '不要使用随机森林', 'forbid_model'),
            ('missing_indicator', missing_csv, [forest], '必须加上缺失指示变量', 'require_missing_indicator'),
        ]
        for scenario, task, proposals, text, kind in scenarios:
            store_path = root / f'{scenario}_{seed}.sqlite3'
            store = CorrectionStore(store_path)
            correction = store.add(text)
            assert correction['status'] == 'active' and correction['kind'] == kind
            # Re-open a new store to prove this does not rely on the previous object.
            restarted = CorrectionStore(store_path)
            observed = {}
            for arm in protocol['arms']:
                result = run_case(root / f'{scenario}_{seed}_{arm}', task, proposals,
                                  correction_store=restarted if arm == 'enforced' else None,
                                  prompt_only=[{'constraint': kind, 'value': correction['value']}] if arm == 'prompt_only' else None)
                experiments = result['experiments']
                if kind == 'reject_duplicates':
                    keys = [fingerprint(e) for e in experiments]
                    mistakes = len(keys) - len(set(keys))
                elif kind == 'forbid_model':
                    mistakes = sum(e['model'] == 'random_forest' for e in experiments)
                else:
                    mistakes = sum(not e.get('preprocessing', {}).get('add_missing_indicator') for e in experiments)
                row = {'scenario': scenario, 'seed': seed, 'arm': arm, 'mistakes': mistakes,
                       'training_fits': len(experiments), 'validation_f1': result['summary']['best_score'],
                       'blocked': sum(e['event'] == 'correction_blocked' for e in result['events']),
                       'auto_repaired': sum(e['event'] == 'correction_applied' for e in result['events']),
                       'seconds': result['measured_seconds'], 'test_evaluated': False,
                       'evidence': f'{scenario}_{seed}_{arm}/run.json'}
                cases.append(row)
                observed[arm] = row
            assert observed['enforced']['mistakes'] == 0
            save(root / f'{scenario}_{seed}_memory.json', {'records': restarted.all(), 'events': restarted.events()})
            if kind == 'reject_duplicates':
                assert observed['no_memory']['validation_f1'] == observed['enforced']['validation_f1']

    # Negative control: a dataset-specific correction must not forbid a different dataset.
    scoped = CorrectionStore(root / 'scope.sqlite3')
    scoped.add('不要使用随机森林', scope='dataset', dataset_sha256=hashlib.sha256((root / 'data_71.csv').read_bytes()).hexdigest())
    unrelated = run_case(root / 'unrelated_scope', root / 'data_83.csv', [forest], scoped)
    false_blocks = sum(e['event'] == 'correction_blocked' for e in unrelated['events'])
    assert false_blocks == 0 and any(e['model'] == 'random_forest' for e in unrelated['experiments'])
    totals = {arm: {'mistakes': sum(r['mistakes'] for r in cases if r['arm'] == arm),
                    'training_fits': sum(r['training_fits'] for r in cases if r['arm'] == arm),
                    'mean_validation_f1': sum(r['validation_f1'] for r in cases if r['arm'] == arm) / 9,
                    'seconds': sum(r['seconds'] for r in cases if r['arm'] == arm)} for arm in protocol['arms']}
    report = {'protocol': protocol, 'cases': cases, 'totals': totals, 'unrelated_scope_blocks': false_blocks,
              'next_session_cases': 9, 'post_feedback_mistake_rate': totals['enforced']['mistakes'] / 9,
              'known_model_calls': 0, 'holdout_evaluations': 0,
              'total_training_fits': sum(r['training_fits'] for r in totals.values()) + len(unrelated['experiments']),
              'datasets_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.csv')},
              'limitations': ['Scripted proposals simulate repeated model mistakes; no live LLM improvement is claimed.',
                              'Zero recurrence applies to checked execution constraints in these nine cases.',
                              'Unknown feedback stays pending; semantic errors outside the vocabulary are not guaranteed.']}
    save(root / 'report.json', report)
    print(json.dumps({'totals': totals, 'unrelated_scope_blocks': false_blocks, 'cases': 9}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    evaluate(args.output)
