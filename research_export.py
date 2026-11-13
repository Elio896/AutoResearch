"""Read-only, explicit research evidence exports; no fits or model calls."""
import html
import io
import json
import zipfile

RESULT_FIELDS=('status','experiment_id','model','parameters','seed','dataset_sha256','split_sizes','excluded_columns','feature_review','validation_f1','validation_accuracy','validation_score','selection_metric','pos_label','validation_f1_minority','validation_minority_precision','validation_minority_recall','validation_class_metrics','duration_seconds')
USAGE_FIELDS=('model_calls','reported_local_tokens','cloud_calls','reported_cloud_tokens','cloud_usage_complete','cloud_cost','jev_calls','jev_cost','total_remote_calls','elapsed_seconds','stop_reason','status','planning_policy','baseline_policy')

def build_manifest(task):
    job=task.get('research_job')
    if task['state']=='finalizing' or (job and job['state'] in ('queued','running')) or any(x['status']=='running' for x in task['experiments']):
        raise RuntimeError('Research is active; wait for a stable export snapshot.')
    report=task.get('research_report') or {}
    sources={};pending='rule_baseline'
    for event in report.get('events',[]):
        if event['event']=='proposal_accepted':pending='local_llm'
        elif event['event']=='cloud_proposal_accepted':pending='cloud_llm'
        elif event['event']=='experiment':
            sources[event['result']['experiment_id']]=pending;pending='rule_baseline'
        elif event['event']=='experiment_failed':pending='rule_baseline'
    experiments=[]
    for item in task['experiments']:
        result=item.get('result') or {}
        experiments.append({'number':item['number'],'status':item['status'],'proposal':item['proposal'],
            'source':sources.get(result.get('experiment_id'),'manual' if not job else 'unresolved'),
            'result':{key:result[key] for key in RESULT_FIELDS if key in result}})
    risk=task.get('risk_report')
    unsafe=bool(task.get('legacy_results_at_risk') or (risk and risk.get('blocked')))
    success=[x['result'] for x in experiments if x['status']=='success']
    best=max(success,key=lambda x:x.get('validation_score',x['validation_f1'])) if success and not unsafe else None
    return {'schema_version':1,'task':{key:task[key] for key in ('id','target','dataset_hash','seed','max_experiments','versions','state','created_at','metric','pos_label')},
        'feature_decisions':task.get('feature_decisions',{}),'risk_report':risk,'legacy_results_at_risk':bool(task.get('legacy_results_at_risk')),
        'conclusion_blocked':unsafe,'experiments':experiments,'best_validation':best,
        'usage':{key:report.get(key) for key in USAGE_FIELDS},'research_settings':job['settings'] if job else None,
        'events':[{key:event[key] for key in ('event','attempt','status','reason','policy') if key in event} for event in task.get('events',[])],
        'test_evaluated':task['state']=='finalized' and task['final_result'] is not None,'final_result':task['final_result'],
        'limitations':['Dataset and model weights are not included.','No trained estimator artifact is saved; reproduction refits training data.','Dependency versions are recorded; original source revision is not persisted per historical task.','Validation selection is not proof of improved generalization or Agent superiority.','Null cost or usage means unreported, not zero.','Export does not train, call models, finalize or change task state.']}

def clean(value):
    return html.escape(str(value)).replace('|','\\|').replace('\n',' ').replace('\r',' ').replace('`','&#96;')

def markdown(manifest):
    task=manifest['task'];usage=manifest['usage']
    lines=['# EvoResearch 研究记录','',f"任务：{clean(task['id'])}",f"状态：{clean(task['state'])}",f"目标列：{clean(task['target'])}",f"选择指标：{clean(task['metric'])}；正类标签：{clean(task['pos_label'])}",f"数据 SHA256：{clean(task['dataset_hash'])}",f"随机种子：{task['seed']}；分层切分：训练/验证/测试 60/20/20",f"实验预算：{task['max_experiments']}；已尝试：{len(manifest['experiments'])}",'']
    if manifest['conclusion_blocked']:lines+=['**存在未解决的数据风险或历史泄漏风险：以下数值仅为历史记录，不能作为可靠研究结论。**','']
    lines+=['## 实验记录','','| 实验 | 来源 | 模型 | 状态 | 验证选择分数 | 参数 |','|---|---|---|---|---:|---|']
    names={'rule_baseline':'规则基线','local_llm':'本地 LLM 提案','cloud_llm':'云端 LLM 提案','manual':'手动实验','unresolved':'来源待确认'}
    for item in manifest['experiments']:
        result=item['result'];score=result.get('validation_score',result.get('validation_f1'))
        lines.append(f"| {item['number']} | {names[item['source']]} | {clean(item['proposal']['model'])} | {clean(item['status'])} | {score if score is not None else '未取得'} | {clean(json.dumps(item['proposal']['parameters'],ensure_ascii=False))} |")
    lines+=['','## 预算与调用','',f"策略：{clean(usage.get('baseline_policy') or '未记录/手动')}；本地调用：{clean(usage.get('model_calls') if usage.get('model_calls') is not None else '未记录')}；云端调用：{clean(usage.get('cloud_calls') if usage.get('cloud_calls') is not None else '未记录')}；Jev 调用：{clean(usage.get('jev_calls') if usage.get('jev_calls') is not None else '未记录')}",f"本地 Token 报告：{clean(json.dumps(usage.get('reported_local_tokens'),ensure_ascii=False))}",f"云端成本：{clean(usage.get('cloud_cost') if usage.get('cloud_cost') is not None else '未知')}；Jev 成本：{clean(usage.get('jev_cost') if usage.get('jev_cost') is not None else '未知')}",'','## 最终测试','']
    if manifest['test_evaluated']:
        final=manifest['final_result'];lines += [f"已执行一次最终测试。选定实验：{clean(final['experiment_id'])}；{clean(task['metric'])}：{final.get('test_score',final['test_f1'])}。",'存在风险时，该结果仍不能作为可靠结论。' if manifest['conclusion_blocked'] else '此后不允许继续调参或重复测试。']
    else:lines+=['未取得成功的最终测试结果。本导出未执行测试，不能把验证集指标称为测试成绩。']
    lines+=['','## 复现说明','','完整配置、逐类指标和风险决定见 manifest.json；依赖版本：'+clean(json.dumps(task['versions'],ensure_ascii=False)),'使用相同代码与原始 CSV，运行包内 replay_validation.py 可重新训练成功候选并输出验证指标。默认不访问测试集。原任务的模型文件和 CSV 不在本包内。历史任务未保存源码修订号，因此不能保证跨代码版本逐位一致。','','## 边界','']
    lines += ['- '+clean(item) for item in manifest['limitations']]
    return '\n'.join(lines)+'\n'

REPLAY='''"""Replay recorded successful validation experiments using the EvoResearch checkout."""
import argparse,hashlib,json,sys
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--project',required=True,help='EvoResearch source checkout directory')
parser.add_argument('--csv',required=True,help='Original CSV, not included in export')
args=parser.parse_args()
manifest=json.loads(Path(__file__).with_name('manifest.json').read_text(encoding='utf-8'))
sys.path.insert(0,str(Path(args.project).resolve()))
from executor import ResearchSession
from task_store import runtime_versions
from orchestrator import validate_proposal
task=manifest['task']
if hashlib.sha256(Path(args.csv).read_bytes()).hexdigest()!=task['dataset_hash']:raise SystemExit('Dataset SHA256 mismatch')
if runtime_versions()!=task['versions']:raise SystemExit('Dependency versions differ; restore recorded environment')
session=ResearchSession(args.csv,task['target'],seed=task['seed'],metric=task['metric'],pos_label=task['pos_label'],feature_decisions=manifest['feature_decisions'])
for experiment in manifest['experiments']:
    if experiment['status']=='success':
        proposal=validate_proposal(json.dumps(experiment['proposal']))
        print(json.dumps({'recorded_number':experiment['number'],'replayed_validation':session.run(**proposal)},ensure_ascii=False))
# Deliberately no finalize: exporting/replaying never automatically opens the test set.
'''

def bundle(manifest):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        archive.writestr('REPORT.md',markdown(manifest))
        archive.writestr('replay_validation.py',REPLAY)
    return stream.getvalue()
