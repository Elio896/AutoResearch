"""Measured, narrowly scoped experiment lessons; no extra LLM calls."""
import hashlib,json,sqlite3,time
from contextlib import contextmanager
from corrections import fingerprint

class LessonStore:
    def __init__(self,path,session,source_run=None):
        self.path=str(path)
        self.source_run=source_run
        context={'dataset':session.dataset_sha256,'target':session.target,'seed':session.seed,
                 'metric':session.metric,'positive_label':session.pos_label,
                 'features':session.risk_report['decisions'],'excluded':session.risk_report['excluded_columns']}
        self.scope=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS lessons(scope TEXT, config TEXT, outcome TEXT, body TEXT, updated REAL, PRIMARY KEY(scope,config,outcome))')
    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=5)
        try:
            with db: yield db
        finally: db.close()
    def record(self,proposal,outcome,score=None,reference=None,error_type=None):
        config=fingerprint(proposal) if proposal else 'execution:'+str(error_type)
        labels={'baseline':'记录本轮起点，尚不能据此认定方案最优。',
                'improved':'该配置在当前验证划分上优于比较方案，可作为后续候选。',
                'no_gain':'该配置未超过比较方案，下次避免原样重复，优先尝试其他配置。',
                'failed':'该次调用或实验执行失败，先检查错误原因；不能据此认定该模型效果差。'}
        row={'source_run':self.source_run,'outcome':outcome,'model':proposal.get('model') if proposal else None,
             'parameters':proposal.get('parameters',{}) if proposal else {},
             'preprocessing':proposal.get('preprocessing',{}) if proposal else {},
             'score':score,'reference_score':reference,'error_type':error_type,
             'summary':labels[outcome],'scope':'仅适用于相同数据、目标、划分、指标和特征处理；不作跨领域结论。'}
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO lessons VALUES(?,?,?,?,?)',(self.scope,config,outcome,json.dumps(row,ensure_ascii=False),time.time()))
        return row
    def snapshot(self):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute('SELECT body FROM lessons WHERE scope=? ORDER BY updated DESC LIMIT 20',(self.scope,))]
    def should_skip(self,proposal,current_score):
        with self.connect() as db:
            row=db.execute('SELECT body FROM lessons WHERE scope=? AND config=? AND outcome=?',(self.scope,fingerprint(proposal),'no_gain')).fetchone()
        if not row:return False
        score=json.loads(row[0]).get('score')
        return score is not None and score<=current_score
