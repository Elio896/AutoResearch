"""Summarize preserved acceptance evidence; never turn missing costs into zero."""
import argparse,csv,json,statistics
from pathlib import Path


def summarize(root):
    root=Path(root);load=lambda name:json.loads((root/name).read_text(encoding='utf-8')) if (root/name).exists() else []
    protocol=load('protocol.json');qa=load('qa-results.json');matrix=load('matrix-results.json');memory=load('memory-results.json')
    lines=['# Frozen V23 acceptance evidence','',
        'Backend fingerprint: `'+protocol['version']['source_fingerprint']+'`.',
        'All model calls are real. Scores are validation results used for selection, not independent test estimates. Prices are official-rate calculations, not reconciled invoices.','',
        '## Question acceptance','',f'Submitted: {len(qa)} / 30. Structural evidence checks passed: {sum(r["evidence_checks_passed"] for r in qa)}. This is not a semantic correctness score.','',
        '| Question | Status | Fits | Download rows | Evidence checks |',
        '|---|---|---:|---:|---|']
    for r in qa:
        bad=[k for k,v in r['evidence_checks'].items() if not v]
        lines.append(f'| {r["case"]} | {r["status"]} | {r.get("training_fits")} | {r.get("download_rows",0)} | {", ".join(bad) or "pass"} |')
    flat=[]
    lines+=['','## Paired model comparison','','Identical data, split, limits and local algorithms; no memory. Acquisition is excluded for both arms. Failed or unfinished arms are retained.','',
        '| Dataset | Seed | Mode | Research status | Macro F1 | Minority F1 | Known tokens | Official USD | Seconds |',
        '|---|---:|---|---|---:|---:|---:|---:|---:|']
    for row in matrix:
        for arm in row['report'].get('arms',[]):
            m=arm['metrics'];cost=(m.get('costs')or{}).get('calculated_total_usd')
            r={'dataset':row['dataset'],'seed':row['seed'],'mode':arm['mode'],'status':m.get('research_status',m['status']),
                'macro_f1':m.get('validation_macro_f1'),'minority_f1':m.get('minority_f1'),'known_tokens':m.get('known_tokens'),
                'official_usd':cost,'seconds':m.get('wall_seconds'),'baseline_only':m.get('baseline_only',True)}
            flat.append(r);lines.append('| '+' | '.join(str(r[k]) for k in ('dataset','seed','mode','status','macro_f1','minority_f1','known_tokens','official_usd','seconds'))+' |')
    ready_pairs=[]
    for row in matrix:
        ready_pairs +=row['report'].get('aggregate',{}).get('paired_deltas_routed_minus_high_only',[])
    means={}
    for key in ('macro_f1_delta','token_savings_percent','cost_savings_percent','time_savings_percent','speedup'):
        values=[p[key] for p in ready_pairs if p.get('comparable') and isinstance(p.get(key),(int,float))]
        means[key]={'n':len(values),'mean':statistics.mean(values) if values else None}
    lines+=['','Descriptive paired means (negative savings means greater use):','', '```json',json.dumps(means,indent=2),'```',
            '', '## Historical experience ablation','','Same acquisition history is copied into two isolated arms. Acquisition overhead is recorded separately. History is revalidated. No weight learning.','',
            '| Seed | Arm | History reused | Macro F1 | Fits | Calls | Official USD | Seconds |',
            '|---|---|---|---:|---:|---:|---:|---:|']
    for row in memory:
        for mode,r in row['arms'].items():
            lines.append('| '+' | '.join(map(str,[row['seed'],mode,(r.get('experience')or{}).get('reused'),r.get('validation_macro_f1'),r.get('training_fits'),r.get('model_calls'),r.get('official_calculated_usd'),r.get('request_wall_seconds')]))+' |')
    summary={'qa_submitted':len(qa),'qa_structural_passed':sum(r['evidence_checks_passed'] for r in qa),
        'paired_comparisons':len(matrix),'paired_descriptive_means':means,'memory_seeds':len(memory),
        'validation_only':True,'semantic_review_required':True,'no_weight_learning':True,'invoice_reconciled':False}
    (root/'evidence-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (root/'evidence-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if flat:
        with (root/'paired-metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();print(json.dumps(summarize(a.directory)))
