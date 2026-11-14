"""Bounded official-page refresh, validated parsing, last-good cache, no LLM."""
import copy,hashlib,json,math,re,threading,time
from datetime import datetime,timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib import request
from cloud_provider import NoRedirect
from model_costs import reference_prices,DEEPSEEK_URL,JEV_URL
from research_system import save

class OfficialHTML(HTMLParser):
    def __init__(self):
        super().__init__();self.text=[];self.tables=[];self.table=None;self.row=None;self.cell=None;self.ignore=0
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style'):self.ignore+=1
        if tag=='table':self.table=[]
        if tag=='tr' and self.table is not None:self.row=[]
        if tag in ('td','th') and self.row is not None:self.cell=[]
    def handle_endtag(self,tag):
        if tag in ('script','style'):self.ignore=max(0,self.ignore-1)
        if tag in ('td','th') and self.cell is not None:
            self.row.append(' '.join(self.cell));self.cell=None
        if tag=='tr' and self.row is not None:
            self.table.append(self.row);self.row=None
        if tag=='table' and self.table is not None:self.tables.append(self.table);self.table=None
    def handle_data(self,data):
        if not self.ignore:self.text.append(data)
        if self.cell is not None:self.cell.append(data)

def read_page(url):
    if url not in (DEEPSEEK_URL,JEV_URL):raise ValueError('Unapproved pricing source')
    req=request.Request(url,headers={'User-Agent':'EvoResearch-PriceCheck/1.0','Cache-Control':'no-cache','Accept':'text/html'})
    with request.build_opener(NoRedirect()).open(req,timeout=10) as response:
        raw=response.read(2_000_001)
    if len(raw)>2_000_000:raise ValueError('Pricing page too large')
    return raw.decode('utf-8'),hashlib.sha256(raw).hexdigest()

def parse_deepseek(html):
    parser=OfficialHTML();parser.feed(html)
    candidates=[t for t in parser.tables if t and 'deepseek-flash' in ' '.join(t[0]) and 'deepseek-v4-pro' in ' '.join(t[0])]
    if len(candidates)!=1:raise ValueError('Pricing model columns changed')
    table=candidates[0]
    headers=[i for i,c in enumerate(table[0]) if 'deepseek-flash' in c or 'deepseek-v4-pro' in c]
    if len(table[0])!=3 or headers!=[1,2]:raise ValueError('Unsupported model order')
    if 'deepseek-flash' not in table[0][1] or 'deepseek-v4-pro' not in table[0][2]:raise ValueError('Unsupported model order')
    off=[];peak=[];off_labels=[]
    for row in table[1:]:
        line=' '.join(row)
        amounts=[float(x) for x in re.findall(r'\$\s*([0-9]+(?:\.[0-9]+)?)',line)]
        if 'OFF-PEAK' in line.upper():
            if len(amounts)!=2:raise ValueError('Invalid price row')
            off.append(amounts);off_labels.append(line.upper())
        elif 'PEAK' in line.upper() and amounts:
            if len(amounts)!=2:raise ValueError('Invalid peak price row')
            peak.append(amounts)
    text=' '.join(' '.join(parser.text).split())
    if len(off)!=3 or len(peak)!=3 or not all(x in text for x in ('CACHE HIT','CACHE MISS','OUTPUT','01:00','04:00','06:00','10:00','Monday','Friday')):raise ValueError('Pricing schedule changed')
    if not all('1M' in line and label in line for line,label in zip(off_labels,('CACHE HIT','CACHE MISS','OUTPUT'))):raise ValueError('Pricing units or row order changed')
    if any(not math.isfinite(v) or not 0<v<10000 for row in off+peak for v in row):raise ValueError('Invalid rates')
    ratios=[peak[i][j]/off[i][j] for i in range(3) for j in range(2)]
    if any(abs(v-ratios[0])>1e-6 for v in ratios):raise ValueError('Peak pricing structure changed')
    return {'rates':{'deepseek-flash':[r[0] for r in off],'deepseek-v4-pro':[r[1] for r in off]},'peak_multiplier':ratios[0]}

def parse_jev(html):
    parser=OfficialHTML();parser.feed(html);text=' '.join(parser.text)
    matches=re.findall(r'Jev costs\s*\$\s*([0-9]+(?:\.[0-9]+)?)\s*per million input tokens',text,re.I)
    if len(set(matches))!=1 or not re.search(r'Output tokens are free',text,re.I):raise ValueError('Jev pricing structure changed')
    amount=float(matches[0])
    if not 0<amount<10000:raise ValueError('Invalid Jev price')
    return {'input_per_million':amount,'output_per_million':0}

class PriceFeed:
    interval=300
    def __init__(self,path,auto_refresh=False):
        self.auto_refresh=auto_refresh
        self.path=Path(path);self.lock=threading.RLock();self.refreshing=False;self.last_attempt=0;self.last_errors={}
        self.prices=reference_prices()
        if self.path.exists():
            try:
                data=json.loads(self.path.read_text(encoding='utf-8'))
                for name in self.prices:
                    entry=data['prices'][name]
                    if entry.get('source_url')!=self.prices[name]['source_url']:raise ValueError()
                ds=data['prices']['deepseek'];jev=data['prices']['typesafe']
                rates=ds['rates']
                if set(rates)!={'deepseek-flash','deepseek-v4-pro'} or any(not isinstance(v,list) or len(v)!=3 or any(type(n) not in (float,int) or not math.isfinite(n) or not 0<n<10000 for n in v) for v in rates.values()):raise ValueError()
                if type(ds['peak_multiplier']) not in (int,float) or not 1<=ds['peak_multiplier']<100:raise ValueError()
                if type(jev['input_per_million']) not in (float,int) or not math.isfinite(jev['input_per_million']) or not 0<jev['input_per_million']<10000 or jev['output_per_million']!=0:raise ValueError()
                self.prices=data['prices'];self.last_attempt=data.get('last_attempt',0);self.last_errors=data.get('last_errors',{})
                if not isinstance(self.last_attempt,(float,int)) or not 0<=self.last_attempt<=time.time()+60:self.last_attempt=0
            except (OSError,ValueError,KeyError,TypeError):pass
    def snapshot_for_request(self):
        with self.lock:
            value=copy.deepcopy(self.prices)
            for provider,entry in value.items():
                if provider in self.last_errors or time.time()-self.last_attempt>self.interval:
                    if entry.get('status')=='live':entry['status']='stale'
            return value
    def status(self):
        return {'prices':self.snapshot_for_request(),'refreshing':self.refreshing,
            'refresh_interval_seconds':self.interval,'last_attempt':self.last_attempt,'errors':dict(self.last_errors)}
    def refresh_async(self,force=False):
        with self.lock:
            if self.refreshing or (not force and time.time()-self.last_attempt<self.interval):return
            self.refreshing=True
        threading.Thread(target=self.refresh,daemon=True,name='official-price-refresh').start()
    def refresh(self):
        updates={};errors={}
        try:
            for name,url,parser in [('deepseek',DEEPSEEK_URL,parse_deepseek),('typesafe',JEV_URL,parse_jev)]:
                try:
                    html,digest=read_page(url);entry=parser(html)
                    now=datetime.now(timezone.utc).isoformat()
                    updates[name]={**entry,'source_url':url,'fetched_at':now,'verified_date':now[:10],'sha256':digest,'status':'live'}
                except Exception:errors[name]='官方页面暂不可达或价格结构变化，保留上次价格并标记过期。'
            with self.lock:
                proposed={**self.prices,**updates}
                checked=time.time()
                try:save(self.path,{'prices':proposed,'last_attempt':checked,'last_errors':errors})
                except OSError:errors['cache']='价格缓存保存失败；已抓取的价格仅在本次进程使用。'
                self.prices=proposed;self.last_attempt=checked;self.last_errors=errors
        finally:
            with self.lock:self.refreshing=False
