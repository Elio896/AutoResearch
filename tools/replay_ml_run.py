"""Replay saved sklearn experiments without any LLM, network or test-set access."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from executor import ResearchSession


def replay(directory):
    directory=Path(directory).resolve()
    report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
    if report.get('execution_kind')=='answer_followup':
        evidence=report.get('context',{}).get('evidence_run_id')
        if not isinstance(evidence,str) or len(evidence)!=32 or any(c not in '0123456789abcdef' for c in evidence):raise ValueError('Invalid evidence run reference')
        directory=directory.parent/evidence
        report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
    protocol=report.get('protocol') or {}
    if not protocol and (directory.parent/'report.json').exists():
        protocol=json.loads((directory.parent/'report.json').read_text(encoding='utf-8')).get('protocol',{})
    best=report.get('best') or {};target=protocol.get('target') or (report.get('research_outputs')or{}).get('target')
    seed=best.get('seed',protocol.get('seed',42))
    if not target or not report.get('experiments'):raise ValueError('No saved training experiments to replay')
    data=directory/'data.csv';digest=hashlib.sha256(data.read_bytes()).hexdigest()
    expected=best.get('dataset_sha256') or protocol.get('dataset_sha256')
    if expected!=digest:raise ValueError('Dataset hash differs from recorded evidence')
    session=ResearchSession(data,target,seed=seed,feature_decisions=best.get('feature_review'),metric=best.get('selection_metric','f1_macro'),pos_label=best.get('pos_label','1'))
    checks=[]
    for prior in report['experiments']:
        result=session.run(prior['model'],prior['parameters'],preprocessing=prior.get('preprocessing'))
        delta=abs(result['validation_score']-prior['validation_score'])
        checks.append({'original_experiment':prior['experiment_id'],'score_delta':delta,'passed':delta<1e-9})
    return {'run_id':directory.name,'dataset_sha256':digest,'seed':seed,'target':target,'experiments':checks,
            'all_passed':all(r['passed'] for r in checks),'llm_calls':0,'test_evaluated':False,
            'scope':'Recomputes recorded validation results, not stochastic external LLM replies.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--output');args=parser.parse_args()
    result=replay(args.directory)
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True));sys.exit(0 if result['all_passed'] else 1)
