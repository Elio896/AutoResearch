"""Replay presentation only. No model calls, rerouting, new fits or quality claims."""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_context import present_answer

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    archive=Path(a.archive).resolve();rows=[]
    for file in sorted(archive.glob('*/report.json')):
        comparison=json.loads(file.read_text(encoding='utf-8'))
        if len(comparison.get('arms',[]))!=2:continue
        for arm in comparison['arms']:
            path=file.parent/(str(arm['seed'])+'-'+arm['mode'])/'report.json'
            if not path.is_file():continue
            report=json.loads(path.read_text(encoding='utf-8'));before=report.get('review') or {}
            outputs=report.get('research_outputs') or {};link=outputs.get('prediction_download_url')
            # An existing CSV can be linked only if actually present in this arm.
            # Comparison arms lack a user-run download endpoint; never invent one.
            if not (path.parent/'predictions.csv').is_file():link=None
            after=present_answer(copy.deepcopy(before),comparison['goal'],outputs,link)
            refs=after['answer_evidence']['references']
            rows.append({'comparison_id':file.parent.name,'seed':arm['seed'],'mode':arm['mode'],
                'research_status_unchanged':report.get('research_status'),'decision_before':before.get('decision'),'decision_after':after.get('decision'),
                'chars_before':len(before.get('answer') or ''),'chars_after':len(after['answer']),
                'evidence_references':refs,'scope_supported':after['answer_evidence']['scope_supported'],
                'unmet_requirements':after.get('unmet_requirements',[]),'answer_after':after['answer']})
    result={'scope':'postprocessing_replay_only','paid_api_calls':0,'new_training_fits':0,'rerouting_retested':False,
        'unique_pairs':len({(r['comparison_id'],r['seed']) for r in rows}),'arms':len(rows),
        'limitations':['Not a new paired model experiment. No improvement in answer correctness or research completion inferred.',
            'The prior keyword rubric penalizes legitimate association-only no-download outputs and technical original field names; it is not used as quality ground truth.',
            'No missing predictions endpoint or missing direct answer is fabricated. Historical failures retain original research status.'], 'rows':rows}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))

if __name__=='__main__':main()
