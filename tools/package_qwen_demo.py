"""Curated original-Qwen source delivery, excluding credentials and private runs."""
import argparse,hashlib,json,re,zipfile,os
from pathlib import Path

EXCLUDED={'.git','.venv','venv','__pycache__','artifacts','incoming','bundle_extract','outputs','work','.agent-log','.vscode','node_modules','data','runtime'}
SUFFIXES={'.py','.cmd','.ps1','.md','.txt','.html','.css','.js','.cjs','.json','.csv'}
SECRET=re.compile(rb'(?:sk-[A-Za-z0-9]{28,}|gh[pousr]_[A-Za-z0-9]{28,}|apikey_[A-Za-z0-9_]{40,})')

def build(root,target):
    root=Path(root);files={}
    for directory,folders,names in os.walk(root):
        folders[:]=[x for x in folders if x not in EXCLUDED and not x.startswith('.')]
        for name in names:
            p=Path(directory)/name;rel=p.relative_to(root)
            if p.suffix.lower() not in SUFFIXES or name.startswith('.'):continue
            if len(rel.parts)>1 and rel.parts[0] not in {'static','fixtures','tools','tests','docs','bench'}:continue
            raw=p.read_bytes()
            if SECRET.search(raw):raise ValueError('Review possible credential in '+rel.as_posix())
            files[rel.as_posix()]=raw
    # Never ship stale historical success curves as a fresh run.
    files['static/evolution.json']=b'{"steps":[],"summary":{},"notes":["Run the original Qwen demo to generate actual results."]}'
    catalog=root/'data/catalog';index=catalog/'index.json'
    if index.exists():
        rows=[r for r in json.loads(index.read_text(encoding='utf-8')) if r['id'] in {'iris','wine','breast_cancer','customer_churn'}]
        for row in rows:
            p=catalog/(row['id']+'.csv');raw=p.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Public catalog checksum mismatch')
            files['data/catalog/'+p.name]=raw
        files['data/catalog/index.json']=json.dumps(rows,ensure_ascii=False,indent=2).encode()
    required=['qwen_demo.py','Setup.cmd','requirements-api.txt','tools/setup_demo.py','docs/qwen-demo-submission.md']+['fixtures/graded/'+n+'.csv' for n in ['g1_easy','g2_mild','g3_medium','g4_hard','g5_harder','g6_extreme']]
    if any(p not in files for p in required):raise ValueError('Missing required submission files')
    manifest={'basis':'original GitHub49efe09 + minimal Qwen demo fixes','qwen_model':'qwen2.5:3b-instruct-q4_K_M','contains_credentials':False,'contains_private_runs':False,'files':{p:hashlib.sha256(raw).hexdigest() for p,raw in files.items()}}
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p,raw in files.items():z.writestr('EvoResearch/'+p,raw)
        z.writestr('EvoResearch/SUBMISSION-MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(target) as z:
        if z.testzip():raise ValueError('ZIP integrity error')
    return {'file':str(target.resolve()),'files':len(files),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(build(Path(__file__).resolve().parents[1],a.output)))
