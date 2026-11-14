"""Small offline index of real tabular datasets and provenance."""
import hashlib
import json
import re
import threading
from pathlib import Path
from data_source import DataSourceError

_LOCK=threading.RLock()
DEFAULT_ROOT=Path(__file__).parent/'data/catalog'
class LocalCatalog:
    provider='local_catalog'
    def __init__(self,root=None):
        self.root=Path(root or DEFAULT_ROOT)
        self.root.mkdir(parents=True,exist_ok=True)
        with _LOCK:
            if not (self.root/'index.json').exists():
                from sklearn.datasets import load_iris,load_wine,load_breast_cancer
                for key,loader,aliases in [
                    ('iris',load_iris,['iris','鸢尾花','花卉']),
                    ('wine',load_wine,['wine','葡萄酒','酒类']),
                    ('breast_cancer',load_breast_cancer,['breast cancer','cancer','乳腺癌','肿瘤'])]:
                    data=loader(as_frame=True)
                    self.add(key,data.frame.to_csv(index=False),'target',aliases,
                        {'provider':'scikit-learn bundled real dataset','name':key,
                         'metadata_url':'https://scikit-learn.org/stable/datasets/toy_dataset.html',
                         'note':'Bundled benchmark dataset; research demonstration only.'})
    def records(self):
        path=self.root/'index.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    def add(self,key,csv,target,aliases,provenance):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',key): raise ValueError('Invalid local ID')
        import pandas as pd,io
        frame=pd.read_csv(io.StringIO(csv))
        if target not in frame or len(frame)>25000 or len(frame.columns)>200 or len(csv.encode())>2000000:
            raise ValueError('Dataset outside local catalog bounds')
        raw=csv.encode('utf-8')
        with _LOCK:
            (self.root/(key+'.csv')).write_bytes(raw)
            rows=[r for r in self.records() if r['id']!=key]
            rows.append({'id':key,'name':provenance.get('name',key),'target':target,
                'aliases':aliases,'rows':len(frame),'columns':len(frame.columns),
                'sha256':hashlib.sha256(raw).hexdigest(),'provenance':provenance})
            temp=self.root/'index.tmp'
            temp.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
            temp.replace(self.root/'index.json')
    def search(self,query):
        if not isinstance(query,str) or not query.strip(): return []
        query=query.lower().replace('_',' ').replace('-',' ')
        ranked=[]
        for row in self.records():
            aliases=[x.lower().replace('_',' ').replace('-',' ') for x in row['aliases']+[row['name']]]
            import csv
            from research_context import FIELD_NAMES
            columns=self.schema(row['id'])
            aliases += [str(c).lower().replace('_',' ').replace('-',' ') for c in columns if c!=row['target'] and len(c)>5]
            aliases += [label for c,label in FIELD_NAMES.items() if c in columns and c!=row['target']]
            if 'international_plan' in columns: aliases += ['国际套餐','international plan','telecom','电信']
            if 'number_customer_service_calls' in columns: aliases += ['客服','流失客户','未流失客户']
            score=max((3 if a==query else 2 if a in query else 1 if query in a else 0) for a in aliases)
            if score: ranked.append((score,row))
        ranked.sort(key=lambda x:-x[0])
        return [{'id':r['id'],'name':r['name'],'rows':r['rows'],'target':r['target']} for _,r in ranked[:8]]
    def schema(self,key):
        import csv
        row=next((r for r in self.records() if r['id']==str(key)),None)
        if row is None: return []
        with (self.root/(row['id']+'.csv')).open(encoding='utf-8-sig',newline='') as stream:
            return next(csv.reader(stream),[])
    def download(self,key):
        row=next((r for r in self.records() if r['id']==str(key)),None)
        if row is None: raise DataSourceError('local_data','本地库没有该数据集。')
        raw=(self.root/(row['id']+'.csv')).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise DataSourceError('local_data','本地数据校验失败，请重新导入。')
        return raw.decode('utf-8'),row['target'],{**row['provenance'],'provider':self.provider,
            'original_source':row['provenance'],'dataset_id':row['id'],'sha256':row['sha256'],'offline':True}
