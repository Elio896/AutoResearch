"""Live LLM context adaptation evaluation: no replay, fake client, or fallback.

Measures raw proposal violations BEFORE enforcement separately from executed
violations AFTER enforcement. This tests contextual adaptation, not weight updates.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
import math
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from agents import Budget
from corrections import CorrectionStore, CorrectionGuard
from executor import ResearchSession
from light_model import from_env
from model_budget import ModelLimits, SharedModel
from research_loop import ResearchLoop
from research_system import save


def evaluate(output, max_calls=18, max_tokens=150000, max_seconds=240):
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'report.json').exists():
        raise ValueError('Use a new evidence directory.')
    protocol = {'mode': 'live_model_only', 'seeds': [711, 833, 977],
                'arms': ['no_memory', 'prompt_only', 'persistent_memory_guard'],
                'feedback': '必须加上缺失指示变量', 'proposals_per_arm': 2,
                'metrics': ['raw_proposal_mistakes', 'executed_mistakes', 'invalid_or_missing_proposals',
                            'validation_f1', 'actual_tokens', 'elapsed_seconds'],
                'limits': {'calls': max_calls, 'tokens': max_tokens, 'seconds': max_seconds},
                'acceptance': 'All 18 proposals valid and fitted; fewer raw mistakes; mean validation F1 must not drop more than 0.01.',
                'min_validation_delta': -0.01,
                'interpretation': 'Context adaptation with retrieved feedback, not model-weight training or a general guarantee.'}
    hashes = {}
    for file in [*root.glob('*.py'), Path(__file__).resolve()]:
        relative = file.relative_to(root)
        destination = directory / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, destination)
        hashes[relative.as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
    protocol['source_sha256'] = hashes
    save(directory / 'protocol.json', protocol)
    try:
        client = from_env()
    except (ValueError, RuntimeError):
        report = {'status': 'blocked', 'reason': 'Invalid server model configuration. Check that the API key is the actual provider key, not Chinese example text.',
                  'actual_model_calls': 0, 'learning_effect': None, 'passed': False}
        save(directory / 'report.json', report)
        print(json.dumps(report))
        return report
    if client is None:
        report = {'status': 'blocked', 'reason': 'No server model credentials configured.',
                  'actual_model_calls': 0, 'learning_effect': None, 'passed': False,
                  'required_configuration': ['LIGHT_MODEL_API_KEY or DEEPSEEK_API_KEY',
                                             'LIGHT_MODEL_BASE_URL and LIGHT_MODEL_NAME for the selected provider']}
        save(directory / 'report.json', report)
        print(json.dumps(report))
        return report

    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_moons
    client.max_calls = max_calls
    shared = SharedModel(client, ModelLimits(max_calls, max_tokens, max_seconds))
    protocol['provider'] = client.descriptor
    save(directory / 'protocol.json', protocol)
    rows = []
    for index, seed in enumerate(protocol['seeds']):
        x, y = make_moons(n_samples=400, noise=.28, random_state=seed)
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame({'signal_a': x[:, 0], 'signal_b': x[:, 1], 'target': y})
        frame.loc[rng.random(len(y)) < np.where(y == 1, .5, .05), 'signal_a'] = np.nan
        csv = directory / f'data_{seed}.csv'
        frame.to_csv(csv, index=False)
        memory_path = directory / f'memory_{seed}.sqlite3'
        memory = CorrectionStore(memory_path)
        memory.add(protocol['feedback'])
        memory = CorrectionStore(memory_path)  # New object: persisted next-session retrieval.
        instructions = CorrectionGuard(memory.snapshot()).instructions()
        order = list(protocol['arms'])
        if index % 2:
            order.reverse()
        for arm in order:
            session = ResearchSession(csv, 'target')
            loop = ResearchLoop(session, proposer=shared.for_role(arm),
                                correction_store=memory if arm == 'persistent_memory_guard' else None,
                                trajectory_dir=directory / 'trajectory' / f'{seed}_{arm}',
                                run_id=f'{seed}_{arm}',
                                budget=Budget(max_experiments=3, max_local_calls=2, max_seconds=max_seconds))
            loop.redteam = None
            if arm == 'prompt_only':
                loop.scientist.corrections = instructions
            started = time.monotonic()
            outcome = loop.run()
            proposals = [e['proposal'] for e in outcome['events'] if e['event'] == 'scientist_proposal']
            raw_mistakes = sum(p.get('preprocessing', {}).get('add_missing_indicator') is not True for p in proposals)
            executed_mistakes = sum(p.get('preprocessing', {}).get('add_missing_indicator') is not True
                                    for p in outcome['experiments'])
            row = {'seed': seed, 'arm': arm, 'valid_proposals': len(proposals),
                   'completed_experiments': len(outcome['experiments']),
                   'invalid_or_missing_proposals': 2 - len(proposals), 'raw_proposal_mistakes': raw_mistakes,
                   'executed_mistakes': executed_mistakes,
                   'validation_f1': outcome['summary']['best_score'],
                   'elapsed_seconds': time.monotonic() - started, 'test_evaluated': False}
            rows.append(row)
            save(directory / f'{seed}_{arm}.json', outcome)
            save(directory / 'progress.json', {'cases': rows, 'usage': shared.snapshot()})
            assert not session.closed
    totals = {arm: {key: sum(row[key] for row in rows if row['arm'] == arm)
                    for key in ('valid_proposals', 'invalid_or_missing_proposals', 'raw_proposal_mistakes', 'executed_mistakes')}
              for arm in protocol['arms']}
    complete = (all(value['valid_proposals'] == 6 for value in totals.values()) and
                all(row['completed_experiments'] == 3 for row in rows))
    scores_available = all(isinstance(row['validation_f1'], (int, float)) and math.isfinite(row['validation_f1']) for row in rows)
    complete = complete and scores_available
    scores = {arm: (sum(row['validation_f1'] for row in rows if row['arm'] == arm) / 3 if scores_available else None)
              for arm in protocol['arms']}
    quality_delta = scores['persistent_memory_guard'] - scores['no_memory'] if scores_available else None
    quality_passed = quality_delta is not None and quality_delta >= protocol['min_validation_delta']
    before = totals['no_memory']['raw_proposal_mistakes']
    after = totals['persistent_memory_guard']['raw_proposal_mistakes']
    if not complete:
        effect = 'inconclusive_missing_or_invalid_real_responses'
    elif before == 0:
        effect = 'no_baseline_errors_no_improvement_demonstrated'
    elif after < before and not quality_passed:
        effect = 'raw_adaptation_but_quality_regressed_not_accepted'
    elif after < before:
        effect = 'observed_context_adaptation_on_this_small_sample'
    else:
        effect = 'no_raw_proposal_improvement_demonstrated'
    report = {'status': 'completed', 'passed': complete and before > after and quality_passed,
              'learning_effect': effect, 'totals': totals, 'cases': rows,
              'mean_validation_f1': scores, 'quality_delta': quality_delta, 'quality_gate_passed': quality_passed,
              'model_usage': shared.snapshot(), 'holdout_evaluations': 0,
              'limitations': ['Only contextual adaptation is tested, not permanent model-weight learning.',
                              'Guard prevention is reported separately; it cannot be counted as LLM learning.',
                              'Three seeds are not statistical proof or cross-domain generalization.']}
    save(directory / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('cases', 'model_usage')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-calls', type=int, default=18)
    parser.add_argument('--max-tokens', type=int, default=150000)
    parser.add_argument('--max-seconds', type=int, default=240)
    args = parser.parse_args()
    result = evaluate(args.output, args.max_calls, args.max_tokens, args.max_seconds)
    sys.exit(0 if result.get('passed') else 2)
