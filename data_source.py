"""Bounded public dataset search/download tools for the source LLM."""
import io
import json
import re
from urllib import request, parse, error
from cloud_provider import NoRedirect
from ml_engine import object_response


class DataSourceError(ValueError):
    def __init__(self,stage,reason,http_status=None,needs_input=False):
        self.stage,self.reason,self.http_status=stage,reason,http_status
        self.needs_input=needs_input
        super().__init__(reason)


class OpenMLCatalog:
    def fetch(self,url,limit=5_000_000):
        for attempt in range(4):
            parts=parse.urlsplit(url)
            if parts.scheme!='https' or parts.hostname not in ('www.openml.org','openml.org') or parts.username or parts.password or parts.fragment:
                raise DataSourceError('dataset_download','数据源跳转到未授权地址，已停止。')
            try:
                with request.build_opener(NoRedirect()).open(url,timeout=20) as response:
                    raw=response.read(limit+1)
                if len(raw)>limit: raise DataSourceError('dataset_download','数据超过当前下载大小限制。')
                return raw
            except error.HTTPError as exc:
                code=exc.code
                location=exc.headers.get('Location')
                exc.close()
                if code in (301,302,303,307,308) and location:
                    url=parse.urljoin(url,location)
                    continue
                raise DataSourceError('public_data_request',f'公开数据源 OpenML 返回 HTTP {code}，尚未取得数据。',code) from None
            except (error.URLError,TimeoutError,OSError):
                raise DataSourceError('public_data_request','无法连接 OpenML 数据源：网络、TLS 或请求超时。模型接口成功不代表数据源可访问。') from None
        raise DataSourceError('public_data_request','数据源跳转次数过多，已停止。')

    def search(self,query):
        if not isinstance(query,str) or not re.fullmatch(r'[A-Za-z0-9 _-]{1,80}',query):
            raise ValueError('Source must provide a short English dataset keyword.')
        url='https://www.openml.org/api/v1/json/data/list/limit/8/data_name/'+parse.quote(query,safe='')
        try:
            data=json.loads(self.fetch(url,100000))
        except DataSourceError as exc:
            if exc.http_status==412: return []
            raise
        return [{'id':str(r['did']),'name':r['name']} for r in data.get('data',{}).get('dataset',[])][:8]

    def download(self,dataset_id):
        if not re.fullmatch(r'[0-9]{1,10}',str(dataset_id)): raise ValueError('Invalid dataset ID.')
        meta_url='https://www.openml.org/api/v1/json/data/'+str(dataset_id)
        meta=json.loads(self.fetch(meta_url,100000))['data_set_description']
        if meta.get('visibility','public')!='public': raise ValueError('Only public datasets are supported.')
        target=meta.get('default_target_attribute')
        if not isinstance(target,str) or not target or ',' in target: raise ValueError('Dataset needs a single declared target.')
        raw=self.fetch(meta['url'])
        import pandas as pd
        if meta.get('format','').lower()=='arff':
            from scipy.io import arff
            values,_=arff.loadarff(io.StringIO(raw.decode('utf-8')))
            frame=pd.DataFrame(values)
            for column in frame.select_dtypes(include='object'):
                frame[column]=frame[column].map(lambda x:x.decode('utf-8') if isinstance(x,bytes) else x)
        elif meta.get('format','').lower()=='csv': frame=pd.read_csv(io.BytesIO(raw))
        else: raise ValueError('Only CSV/ARFF source datasets are supported.')
        if len(frame)>25000 or len(frame.columns)>200: raise ValueError('Dataset exceeds lightweight ML resource bounds.')
        if target not in frame: raise ValueError('Declared target column is missing.')
        csv=frame.to_csv(index=False)
        if len(csv.encode('utf-8'))>2_000_000: raise ValueError('Converted CSV exceeds 2 MB.')
        return csv,target,{'provider':'OpenML','dataset_id':str(dataset_id),'name':meta.get('name'),
                          'metadata_url':meta_url,'download_url':meta['url'],'license':meta.get('licence'),
                          'note':'Public catalog provenance; no guarantee of suitability or permission for every downstream use.'}


def collect_dataset(goal,gateway,catalog=None,emit=None):
    emit=emit or (lambda event: None)
    if catalog is None:
        from local_catalog import LocalCatalog
        catalog=LocalCatalog()
    source=gateway.for_role('source')
    records=catalog.records() if hasattr(catalog,'records') else []
    available='、'.join(str(r.get('name',r['id'])) for r in records)
    direct=catalog.search(goal) if records else []
    if records and hasattr(catalog,'schema'):
        from research_context import scope_gap
        if direct:
            gaps=[scope_gap(goal,catalog.schema(r['id']),r) for r in direct]
            if all(gaps):
                raise DataSourceError('dataset_scope',gaps[0]+' 当前可用数据集：'+available+'。',needs_input=True)
    from research_context import is_followup
    if records and not direct and is_followup(goal):
        raise DataSourceError('dataset_search','这句话像是对上一轮结果的追问，但本次没有关联上一轮任务。请打开原任务继续提问，或说明要用哪个数据集。当前可用数据集：'+available+'；也可上传 CSV。',needs_input=True)
    catalog_context=[{**{k:r.get(k) for k in ('id','name','aliases','target')},
                     'fields':catalog.schema(r['id'])[:50] if hasattr(catalog,'schema') else []} for r in records]
    query=object_response(source([
        {'role':'system','content':'You are the data-source agent for tabular classification. Translate the research need into one local dataset catalog search. Return only JSON {"query":"short English dataset name keyword"}. Do not invent data, URLs, or results.'},
        {'role':'user','content':json.dumps({'question':goal,'available_catalog':catalog_context},ensure_ascii=False) if records else goal}],timeout=30))
    keyword=query.get('query')
    if not isinstance(keyword,str) or not re.fullmatch(r'[A-Za-z0-9 _-]{1,80}',keyword):
        raise DataSourceError('dataset_search','来源模型未给出有效的英文搜索关键词。')
    words=keyword.strip().split()
    variants=list(dict.fromkeys([keyword.strip(),'_'.join(words),words[-1]]))[:3]
    candidates=[]
    attempted=[]
    for index,term in enumerate(variants):
        if index: emit({'event':'source_search_retry','query':term,'reason':'上一关键词没有结果，改用数据集命名变体或核心词'})
        emit({'event':'source_query','query':term})
        attempted.append(term)
        candidates=catalog.search(term)
        emit({'event':'source_candidates','count':len(candidates),'query':term})
        if candidates:
            query['query']=term
            break
    if not candidates: candidates=direct
    if not candidates:
        raise DataSourceError('dataset_search','当前数据目录没有匹配这项研究的数据；已搜索：'+ '、'.join(attempted)+'。'+('当前可用数据集：'+available+'。' if available else '')+'请说明研究对象或上传匹配的 CSV，尚未开始训练。',needs_input=True)
    choice=object_response(source([
        {'role':'system','content':'Select a suitable dataset ONLY from supplied catalog results. Return JSON {"dataset_id":"one supplied id","reason":"why it matches"}. If none is suitable, return {"dataset_id":null,"reason":"clarification needed"}.'},
        {'role':'user','content':json.dumps({'request':goal,'catalog_results':candidates})}],timeout=30))
    if str(choice.get('dataset_id')) not in {str(c['id']) for c in candidates}:
        raise DataSourceError('dataset_selection','找到候选数据，但尚不能确认适合研究。请补充研究对象和目标列。'+('当前可用数据集：'+available+'。' if available else ''),needs_input=True)
    emit({'event':'source_selected','name':next(c['name'] for c in candidates if str(c['id'])==str(choice['dataset_id'])),'reason':str(choice.get('reason',''))[:500]})
    emit({'event':'source_downloading'})
    csv,target,provenance=catalog.download(choice['dataset_id'])
    provenance['search_query']=query['query']
    provenance['attempted_queries']=attempted
    provenance['selection_reason']=str(choice.get('reason',''))[:500]
    return csv,target,provenance
