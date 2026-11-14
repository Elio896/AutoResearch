'use strict';
const $=id=>document.getElementById(id);let lastWorkflowRun=null;let contextRun=null;let run=null,csv=null,waiting=false,backendReady=false,lastRenderKey=null;
$('start').textContent='开始研究';
function rememberRun(id){if(!/^[a-f0-9]{32}$/.test(id))throw new Error('Invalid run reference');localStorage.setItem('ml-run',id);const url=new URL(location.href);url.searchParams.set('run',id);history.replaceState(null,'',url);}
async function api(url,body){const r=await fetch(url,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const d=await r.json();if(r.status===404 && ['/ml/credentials','/ml/connection-check'].includes(url))throw new Error('当前运行的是旧版服务，密钥设置接口不存在，并非 API Key 无效。请重新运行 Start-Connected-ML.cmd 加载新版。');if(!r.ok)throw new Error(typeof d.detail==='string'?d.detail:'请求参数或接口异常。');return d;}
const names={source_collection_started:'来源站开始搜集',source_collected:'来源数据已获取',planning_started:'来源站分析与拆解',plan:'研究任务已规划',route_reused:'复用已验证派发，节省分类调用',route:'分类站派发',proposal:'工作站提案',experiment:'真实训练完成',verification:'程序验证',rollback:'拒绝退化并恢复最优方案',escalation:'升级高成本工作站',worker_failed:'工作站失败',classification_deferred:'分类不确定，暂缓派发',final_review:'高性能模型审核完成',strategy_pinned:'固定本轮策略版本（自动升级未启用）',classification_completed:'分类完成',stopped:'预算停止',correction_applied:'纠错约束已应用'};
function render(s){if(!['queued','running','failed','needs_input','blocked'].includes(s.status)){contextRun=run;$('context-note').textContent='追问将引用任务 '+run.slice(0,8)+' 的数据和结果，不会仅为解释而重新训练。';}const key=JSON.stringify([run,s.status,s.events?.length,s.report?.best?.validation_score,!!s.partial,!!s.report?.account_billing,s.reason]);if(key===lastRenderKey)return;lastRenderKey=key;$('status').textContent={needs_revision:'研究需修正',queued:'排队中',running:'工作站运行中',completed:'链路执行完成',failed:'链路未完成 · 保留部分证据',needs_input:'需要补充信息',blocked:'派发或执行受阻',partial_completed:'已完成可支持的关联分析（非因果）'}[s.status]||s.status;const report=s.report||s.partial;const ev=s.events||report?.events||[];renderWorkflow(s,ev);$('events').replaceChildren(...ev.map(e=>{const li=document.createElement('li');li.textContent=eventSummary(e)+(e.task?' · '+e.task:'')+(e.tier?' · '+e.tier:'')+(e.category?' · '+e.category:'')+(e.reason?' · '+e.reason:'');return li;}));if(!report){$('actual-tokens').textContent='—';$('score').textContent='—';renderAccountBilling(null);}if(report){renderLessons(report.lessons||[]);renderResearchAnswer(report);$('result-placeholder').hidden=true;$('score').textContent=report.best?.validation_score?.toFixed(4)??'—';const u=report.usage;renderCosts(u?.costs,report.account_billing);$('actual-tokens').textContent=u?.known_tokens??'—';$('usage').textContent=u?`${u.calls} / ${u.known_tokens}`:'—';$('usage-note').textContent=u?.unreconciled_calls?`${u.unreconciled_calls} 次用量未对账，预留 ${u.reserved_unknown_tokens} token；费用未知。`:'按真实 usage 记账；费用未核算。';$('details').textContent=$('details').parentElement.open?JSON.stringify(report,null,2):'展开后查看详细记录；全部预测可通过 CSV 或完整报告下载。';window.currentResearchReport=report;if(report.review){$('review-state').textContent={approve:'模型审核通过',revise:'需要修正',reject:'审核不通过'}[report.review.decision];if(report.review_performed===false)$('review-state').textContent='尚未审核：派发或执行未完成';$('review-summary').textContent=report.review.summary;$('provenance').textContent=report.source?.provider==='user_csv'?'数据来源：用户 CSV':`数据来源：${report.source?.provider||'未知'} · ${report.source?.name||''} · ${report.source?.metadata_url||''}`;}}if(['failed','blocked','needs_input'].includes(s.status))$('message').textContent=(s.reason?.startsWith('Research stopped;')?'这是旧版任务留下的失败记录：当时公开数据采集发生 HTTP 错误，旧版未记录状态码，不能据此判断密钥无效。':s.reason)||'请检查部分证据和接口配置。';$('export').hidden=!s.report;$('export').href=`/ml/runs/${run}/export`;}
async function poll(){waiting=true;try{const s=await api(`/ml/runs/${run}?compact=true`);render(s);if(['queued','running'].includes(s.status)){setTimeout(poll,document.hidden?5000:1500);}else{waiting=false;$('start').disabled=false;}}catch(e){$('message').textContent=e.message;waiting=false;$('start').disabled=false;}}
$('mode').onchange=()=>{$('csv-input').hidden=$('mode').value!=='csv';$('source-note').hidden=$('mode').value==='csv';};
$('file').onchange=async()=>{const f=$('file').files[0];if(f){if(f.size>2000000){$('message').textContent='CSV 最大为 2 MB。';return;}csv=await f.text();}};
$('sample').onclick=async()=>{try{const d=await api('/ml/sample');csv=d.csv_text;$('target').value=d.target;$('message').textContent='示例 CSV 已载入。';}catch(e){$('message').textContent=e.message;}};
$('form').onsubmit=async e=>{e.preventDefault();if(waiting)return;if(!backendReady){$('message').textContent='当前后台尚未通过版本核对，不能使用历史经验功能。请运行 Start-Local-Research.cmd 更新后台。';return;}try{const body={economy:true,max_rounds:3,patience:2,goal:$('goal').value,max_calls:Number($('calls').value),max_tokens:Number($('tokens').value),max_fits:Number($('fits').value),max_expert_calls:Number($('experts').value),evaluate_test:$('final-test').checked};if($('use-context').checked&&contextRun&&$('mode').value!=='csv')body.previous_run_id=contextRun;if($('mode').value==='csv'){if(!csv)throw new Error('请上传 CSV 或载入示例。');body.csv_text=csv;body.target=$('target').value;}waiting=true;$('start').disabled=true;$('message').textContent='';$('research-answer').hidden=true;$('review-state').textContent='尚未审核';$('review-summary').textContent='等待当前研究结果。';const d=await api('/ml/runs',body);run=d.run_id;rememberRun(run);poll();}catch(e){waiting=false;$('start').disabled=false;$('message').textContent=e.message;}};
api('/ml/config').then(c=>{$('config').textContent=c.configured?'API 角色已配置：'+Object.entries(c.roles).filter(([r])=>r!=='planner').map(([r,m])=>`${r} = ${m}`).join(' · '):'服务器待配置：'+c.reason;}).catch(e=>{$('config').textContent=e.message;});
const linkedRun=new URLSearchParams(location.search).get('run');
const previousRun=linkedRun||localStorage.getItem('ml-run');
if(previousRun&&/^[a-f0-9]{32}$/.test(previousRun)){$('context-note').textContent='追问将引用任务 '+previousRun.slice(0,8)+' 的结果。';api(`/ml/runs/${previousRun}?compact=true`).then(s=>{if(run)return;if(['completed','partial_completed','needs_revision'].includes(s.status)){contextRun=previousRun;$('context-note').textContent='已关联上一轮任务 '+previousRun.slice(0,8)+'，可直接追问。';}if(linkedRun||['queued','running'].includes(s.status)){run=previousRun;render(s);if(['queued','running'].includes(s.status))poll();}else{$('message').textContent='上一次任务已结束。提交新需求后，将显示本次执行结果。';}}).catch(()=>localStorage.removeItem('ml-run'));}


$('key-form').onsubmit=async e=>{e.preventDefault();const b=e.submitter;if(b)b.disabled=true;try{const saved=await api('/ml/credentials',{deepseek:$('deepseek-key').value.trim()||null,jev:$('jev-key').value.trim()||null});if(saved.persisted!==true)throw new Error('当前后台尚未支持加密保存，不能确认密钥已持久保存。请运行启动器更新后台后在设置中重新保存。');$('deepseek-key').value='';$('jev-key').value='';await loadCredentialStatus();if(saved.configured){$('config').textContent='密钥已加密保存，正在实际验证模型连接…';await verifyConnection();}else{$('key-message').textContent='已加密保存，请补充另一项密钥。';}}catch(e){$('key-message').textContent=e.message;}finally{if(b)b.disabled=false;}};
async function verifyConnection(){const b=$('check-connection');b.disabled=true;$('key-message').textContent='正在实际调用三个模型，请稍候…';try{const r=await api('/ml/connection-check',{});$('key-message').textContent=r.results.map(x=>`${x.model}：${x.connected?'连接成功':x.reason+(x.http_status?' HTTP '+x.http_status:'')}`).join('；');$('config').textContent=r.connected?'三个模型均已实际连接成功':'存在连接失败，请查看密钥设置中的具体原因';}catch(e){$('key-message').textContent=e.message;throw e;}finally{b.disabled=false;}}
$('check-connection').onclick=()=>verifyConnection().catch(()=>{});

const flowRoles=['source','classifier','low','high','reviewer'];
function eventRole(e,prior){if(e.event.startsWith('route_recheck'))return 'classifier';if(['clarification_required','routing_blocked'].includes(e.event))return 'high';if(e.event.startsWith('source_')||['planning_started','plan'].includes(e.event))return 'source';if(['classification_started','classification_deferred','route'].includes(e.event))return 'classifier';if(['review_started','final_review'].includes(e.event))return 'reviewer';if(e.event==='escalation')return 'high';return e.tier||prior||'source';}
function eventSummary(e){const labels={iteration_started:'开始新一轮研究',research_replanned:'依据上一轮证据修正实验计划',iteration_completed:'本轮研究完成',iteration_stopped:'达到停止条件，保留最佳方案',duplicate_proposal_skipped:'跳过已验证的重复方案',context_loaded:'载入上一轮研究上下文',classification_uncertain:'分类不确定，准备升级复核',method_scope_review:'正在检查研究问题与方法的适配范围',analysis_scope_adjusted:'因果证据不足，继续可支持的关联分析',association_started:'正在执行本地关联统计',association_completed:'关联分析已完成',route_recheck_started:'高级模型正在复核派发方案',route_recheck_resolved:'高级复核完成，重新派发',route_recheck_unresolved:'高级复核仍需补充信息',route_recheck_failed:'高级复核失败，流程停止',clarification_required:'请补充必要信息',routing_blocked:'派发或执行未完成',experience_loaded:'载入历史最佳方案并重新验证',experience_retained:'沿用已验证的最佳方案',experience_rejected:'历史方案复验失败，恢复基线',lesson_recorded:'保存实测经验',lessons_consulted:'参考历史经验',lesson_prevented_repeat:'避免重复无效实验',data_ready:'数据已整理，交给分类站派发',source_search_retry:'正在调整关键词重新搜索',source_query:'搜索关键词',source_candidates:'找到候选数据',source_selected:'选择数据',source_downloading:'下载数据',source_collection_started:'来源站正在理解需求并搜集数据',source_collected:'数据已获取，准备整理',planning_started:'来源站正在整理研究目标',plan:'研究任务已整理',classification_started:'分类站正在判断任务类型与适配工作站',worker_started:'工作站正在分析并提出实验方案',training_started:'程序正在执行真实训练',review_started:'审核站正在检查实验与证据',route:'分类完成，派发至工作站',escalation:'任务升级至高级工作站',final_review:'最终审核完成',experiment:'真实训练完成',worker_failed:'工作站执行失败',classification_deferred:'分类置信度不足，暂缓派发'};return (labels[e.event]||names[e.event]||e.event)+(e.round?'（第 '+e.round+'/'+(e.total_rounds||3)+' 轮）':'')+(e.query?'：'+e.query:'')+(e.count!==undefined?'（'+e.count+' 个）':'')+(e.name?'：'+e.name:'')+(e.reason?' · '+e.reason:'')+(e.score!==undefined?' · 验证 F1 '+Number(e.score).toFixed(4):'');}
const FLOW_TEXT={'source-classifier':'来源 → 分类站：数据画像（行数、类别、缺失率）',
 'classifier-low':'分类站 → 常规工作站（本地）：任务单 + 能力判断',
 'classifier-high':'分类站 → 高级工作站：任务单 + 能力判断',
 'low-high':'常规工作站 → 高级工作站：这个做不了，升级',
 'low-reviewer':'常规工作站 → 审核站：实验记录（模型、参数、验证分）',
 'high-reviewer':'高级工作站 → 审核站：诊断提案与实验结果'};
let lastFlowKey='';
function animateFlow(edges){
  const key=edges.slice().sort().join(',');
  if(key===lastFlowKey) return;
  lastFlowKey=key;
  const paths=[...document.querySelectorAll('.flow-lines path')];
  paths.forEach(p=>p.classList.remove('on'));
  edges.filter(e=>document.getElementById('edge-'+e)).forEach((e,i)=>{
    setTimeout(()=>{const p=document.getElementById('edge-'+e);if(!p)return;
      p.classList.add('on');
      const cap=document.getElementById('flow-caption');
      if(cap) cap.textContent=FLOW_TEXT[e]||e;},i*700);
  });
}
function renderWorkflow(state,events){if(lastWorkflowRun!==run){for(const role of flowRoles)$('action-'+role).textContent='等待本轮任务';lastWorkflowRun=run;}let current='source';const visited=new Set(),edges=new Set();let lastWorker=null;for(const e of events){current=eventRole(e,current);visited.add(current);if(current==='classifier')edges.add('source-classifier');if(e.event==='route'){lastWorker=e.tier;edges.add('classifier-'+e.tier);}if(e.event==='escalation'){lastWorker='high';edges.add('low-high');}if(e.event==='route_recheck_started')edges.add('classifier-high');if(current==='reviewer'&&lastWorker)edges.add(lastWorker+'-reviewer');if(edges.size)setTimeout(()=>animateFlow([...edges]),0);if(e.event==='association_started')$('model-'+current).textContent='本地统计工具（无需调用模型）';const el=$('action-'+current);if(el){el.textContent=eventSummary(e);el.title=el.textContent;}}
 const running=['queued','running'].includes(state.status);for(const r of flowRoles){const n=$('node-'+r);n.classList.toggle('visited',visited.has(r));n.classList.toggle('needs-input',state.status==='needs_input'&&r===current);n.classList.toggle('active',running&&r===current);n.classList.toggle('error',['failed','blocked'].includes(state.status)&&r===current);n.querySelector('.node-state').textContent=running&&r===current?'正在执行':['failed','blocked'].includes(state.status)&&r===current?'执行中断':state.status==='needs_input'&&r===current?'需补充信息':visited.has(r)?'已参与':'等待任务';}
 document.querySelectorAll('.flow-lines path').forEach(p=>{const key=p.id.slice(5);p.classList.toggle('visited',edges.has(key));p.classList.toggle('flowing',running&&edges.has(key)&&(key.endsWith(current)||key==='classifier-'+lastWorker&&current===lastWorker));});
 $('flow-state').textContent=running?'实时执行中':['failed','blocked','needs_input'].includes(state.status)?'流程已暂停':state.status==='needs_revision'?'研究结果仍需修正':state.status==='partial_completed'?'关联分析完成，因果结论不支持':state.status==='completed'?'本轮执行结束':'等待任务';$('flow-summary').textContent=['failed','blocked','needs_input'].includes(state.status)?(state.reason||state.report?.clarification?.reason||'流程中断，请查看错误详情'):events.length?eventSummary(events[events.length-1]):'等待执行事件…';}
api('/ml/config').then(c=>{for(const r of flowRoles)if(c.roles?.[r])$('model-'+r).textContent=c.roles[r];}).catch(()=>{});

$('start').disabled=true;
api('/ml/version').then(v=>{backendReady=v.version==='ml-chain-23'&&!v.restart_required;$('start').disabled=!backendReady;if(!backendReady){$('backend-warning').hidden=false;$('backend-warning').textContent='当前后台仍是 '+v.version+'，历史最佳方案和经验功能未加载。请运行 Start-Local-Research.cmd 更新服务，刷新后此提示会消失。';$('status').textContent='后台需要更新';}}).catch(()=>{$('backend-warning').hidden=false;$('backend-warning').textContent='无法核对后台版本，已暂停提交，避免再次执行旧流程。';});

api('/ml/catalog').then(c=>{$('catalog-note').textContent='本地可用数据：'+c.datasets.map(d=>d.aliases.find(a=>/[\u4e00-\u9fff]/.test(a))||d.name).join('、')+'。也可直接上传 CSV。';}).catch(()=>{});

function userFacingAnswer(report){
 let answer=report.review?.answer||'';
 if(report.research_outputs?.records_predicted)answer=answer.replace(/未使用训练时未见过的全新数据。/g,'这些验证记录未用于模型拟合，但参与了方案选择；不是未来客户或独立最终测试。');
 const goal=report.research_outputs?.question||report.protocol?.request?.goal||'';
 if(!goal.includes('开发者面板'))return answer;
 const technical=/exp_\d+|random_forest|hist_gradient_boosting|logistic_regression|n_estimators|max_depth|min_samples_leaf|max_iter|learning_rate|class_weight|模型参数|调参日志|超参数/i;
 const structured=answer.replace(/(发现了什么|证据是什么|有哪些局限|直接回答|数据证据|预测结果|局限)[：:]/g,'$1：\n');
 const filtered=structured.split(/\n\s*\n/).map(p=>p.split(/(?<=[。！？])|\n/).filter(s=>!technical.test(s)).join('\n').trim()).filter(Boolean).join('\n\n');
 let summary='';
 if(report.training_fits>0&&report.iteration&&['approved','partial_completed'].includes(report.research_status)){
  const rounds=report.iteration.rounds||[],first=rounds[0]?.incumbent_before,last=rounds.at(-1)?.incumbent_after;
  if(rounds.length&&typeof first==='number'&&typeof last==='number')summary=(report.experience?.reused?'本轮已复用并重新验证历史方案。':'本轮已验证基线方案。')+'完成 '+rounds.length+' 轮研究；'+(last>first?'新方案通过验证，已保留更优结果。':'新方案未超过本轮起点，保留原有最佳方案。');
  const o=report.research_outputs;if(o?.records_predicted)summary+='已为 '+o.records_predicted+' 条验证记录生成逐条预测。';
 }
 return [summary,filtered].filter(Boolean).join('\n\n')||'技术细节请在开发者面板查看；实际数据结果与交付文件见下方。';
}

function renderResearchAnswer(report){const o=report.research_outputs;$('research-answer').hidden=!report.review;$('direct-answer').textContent=userFacingAnswer(report)||'这份报告尚未提供针对原问题的最终回答。技术审核摘要可在开发者面板查看。';renderAssociation(o?.association_evidence);$('prediction-block').hidden=!o||!o.records_predicted;if(!o){$('research-limits').replaceChildren();$('experience-note').textContent='';return;}$('experience-note').textContent=report.experience?.reused?'已复用历史验证最佳方案，本轮先重训验证；来源任务：'+report.experience.source_run:'本轮从默认基线开始；审核通过的最佳方案将供下次相同数据和目标复用。';$('prediction-summary').textContent='预测范围：'+o.records_predicted+' 条验证记录；'+Object.entries(o.prediction_counts).map(([label,count])=>'类别 '+label+'：'+count+' 条').join('，');$('prediction-download').href=o.prediction_download_url||`/ml/runs/${run}/predictions`;$('prediction-rows').replaceChildren(...o.predictions.slice(0,20).map(r=>{const tr=document.createElement('tr');for(const val of [r.source_row,r.actual_label,r.predicted_label,r.predicted_probability===undefined?'—':Number(r.predicted_probability).toFixed(3)]){const td=document.createElement('td');td.textContent=val;tr.append(td);}return tr;}));$('feature-answer').textContent=o.feature_evidence.map(f=>f.feature+'（打乱后宏 F1 下降 '+f.macro_f1_drop.toFixed(4)+'）').join('；')||'没有正向重要性证据。';$('research-limits').replaceChildren(...o.limitations.map(t=>{const li=document.createElement('li');li.textContent=t;return li;}));}

function renderLessons(lessons){$('lessons-panel').hidden=!lessons.length;$('lesson-list').replaceChildren(...lessons.map(l=>{const li=document.createElement('li');li.textContent=(l.model?l.model+'：':'执行记录：')+l.summary+(l.score===null?'':' 验证得分 '+l.score.toFixed(4))+(l.reference_score===null?'':'，比较得分 '+l.reference_score.toFixed(4))+' '+l.scope;return li;}));}

$('details').parentElement.addEventListener('toggle',()=>{if($('details').parentElement.open&&window.currentResearchReport)$('details').textContent=JSON.stringify(window.currentResearchReport,null,2);});
document.addEventListener('visibilitychange',()=>document.body.classList.toggle('page-hidden',document.hidden));

$('force-reset').onclick=()=>{localStorage.removeItem('ml-run');localStorage.removeItem('ml-comparison');sessionStorage.setItem('ml-page-reset','1');location.replace(location.pathname);};
if(sessionStorage.getItem('ml-page-reset')){sessionStorage.removeItem('ml-page-reset');$('message').textContent='页面已重置。后台任务如仍在运行会继续执行，密钥和历史报告保留。';}

async function loadCredentialStatus(){try{const s=await api('/ml/credentials/status');$('credential-status').textContent=s.load_error||('DeepSeek：'+(s.deepseek_configured?'已保存':'未保存')+'；Jev：'+(s.jev_configured?'已保存':'未保存')+'。重启服务和重新打开浏览器后自动加载。');$('deepseek-key').placeholder=s.deepseek_configured?'已保存，留空不修改':'首次配置请输入密钥';$('jev-key').placeholder=s.jev_configured?'已保存，留空不修改':'首次配置请输入密钥';}catch(e){$('credential-status').textContent='暂时无法读取密钥保存状态。';}}
loadCredentialStatus();

let comparisonId=null,comparisonBusy=false;
const compareNames={routed:'多智能体协作',high_only:'高成本单模型'};
function compareNumber(v,n=3){return typeof v==='number'?v.toFixed(n):'—';}
function renderComparison(s){$('compare-status').textContent=s.status==='completed'?'对比已结束；请同时检查成功组和失败组':s.status==='failed'?(s.reason||'对比失败'):s.status==='queued'?'对比任务排队中':`${compareNames[s.mode]||'准备中'} · 种子 ${s.seed??'—'} · 已完成 ${s.completed_arms||0}/${s.total_arms||'—'} 组 · ${s.event||'正在执行'}`;
 if(!s.report)return;const r=s.report;$('compare-rows').replaceChildren(...r.arms.map(a=>{const m=a.metrics,tr=document.createElement('tr');const vals=[compareNames[a.mode]+' / '+a.seed,m.status==='completed'?(m.research_status==='approved'?'完成/审核通过':m.baseline_only?'未完成研究 · 仅基线':'完成/需修正'):'失败：'+m.error_type,m.known_input_tokens??'—',m.known_output_tokens??'—',m.usage_complete?'完整':m.status==='failed'?'见报告':'有未对账调用',costText(m.costs),'待对账',compareNumber(m.wall_seconds),(m.baseline_only?'基线 '+compareNumber(m.validation_macro_f1,4):compareNumber(m.validation_macro_f1,4)),compareNumber(m.minority_f1,4),compareNumber(m.minority_pr_auc_average_precision,4),m.calls===undefined?'—':m.calls+' / '+m.training_fits+'（失败调用 '+m.failed_calls+'）',m.delivery_checks_passed??'—'];for(const v of vals){const td=document.createElement('td');td.textContent=v;tr.append(td);}return tr;}));
 const pairs=r.aggregate.paired_deltas_routed_minus_high_only;const valid=pairs.filter(p=>p.comparable===true);const mean=k=>{const xs=valid.map(p=>p[k]).filter(v=>typeof v==='number');return xs.length?xs.reduce((a,b)=>a+b,0)/xs.length:null;};$('compare-summary').textContent=`可比较配对 ${valid.length}/${pairs.length}${valid.length<pairs.length?'（其余分组未完成研究或分析范围不一致，不参与平均差）':''}；协作减单模型的平均差：官方价核算费用 ${compareNumber(mean('calculated_usd_delta'),8)} USD（非账单实扣），已对账 Token ${compareNumber(mean('known_tokens_delta'),0)}，耗时 ${compareNumber(mean('wall_seconds_delta'),2)} 秒，宏 F1 ${compareNumber(mean('macro_f1_delta'),4)}。负 Token/时间差才代表本次节省。少量重复不能证明普遍优越；费用尚未对账。`;
 $('compare-answers').replaceChildren(...r.arms.filter(a=>a.metrics.final_answer).map(a=>{const article=document.createElement('article'),h=document.createElement('h3'),p=document.createElement('p');h.textContent=compareNames[a.mode]+' · 种子 '+a.seed+' · 实际回答';p.textContent=a.metrics.final_answer;article.append(h,p);if(a.metrics.costs){const fees=document.createElement('p');fees.className='hint';fees.textContent='模型费用（官方价核算，非账单实扣）：'+a.metrics.costs.models.map(m=>m.model+'（输入 '+m.known_input_tokens+'，输出 '+m.known_output_tokens+' Token） '+(typeof m.calculated_usd==='number'?'$'+m.calculated_usd.toFixed(8):'费用不完整/范围见下载报告')).join('；');article.append(fees);}return article;}));const rel=r.aggregate.relative_summary;if(rel){const value=k=>compareNumber(rel[k]?.mean,2);$('compare-summary').textContent+=` 相对全高成本模式：Token 节省 ${value('token_savings_percent')}%；官方价估算费用节省 ${value('cost_savings_percent')}%；耗时节省 ${value('time_savings_percent')}%；速度比 ${value('speedup')}。负数表示增加；未知费用不计算；${rel.paired_runs} 个配对观察，不构成优越性结论。`;} $('compare-export').hidden=false;$('compare-export').href=`/ml/comparisons/${comparisonId}/export`;}
async function pollComparison(){try{const s=await api(`/ml/comparisons/${comparisonId}`);renderComparison(s);if(['queued','running'].includes(s.status)){setTimeout(pollComparison,document.hidden?5000:1500);}else{comparisonBusy=false;$('compare-start').disabled=false;}}catch(e){$('compare-status').textContent=e.message;comparisonBusy=false;$('compare-start').disabled=false;}}
$('compare-start').onclick=async()=>{if(comparisonBusy)return;if(!backendReady){$('compare-status').textContent='请先加载匹配版本的后台。';return;}const goal=$('goal').value.trim();if(goal.length<3){$('compare-status').textContent='先在研究需求中填写要比较的问题。';return;}try{const body={economy:true,max_rounds:3,patience:2,goal,dataset_id:$('compare-dataset').value,seeds:$('compare-repeats').value==='3'?[42,7,2026]:[42],max_calls:Number($('calls').value),max_tokens:Number($('tokens').value),max_fits:Math.max(2,Number($('fits').value))};if($('mode').value==='csv'){if(!csv)throw new Error('请先上传 CSV。');body.csv_text=csv;body.target=$('target').value;}comparisonBusy=true;$('compare-start').disabled=true;$('compare-status').textContent='正在提交真实对比…';$('compare-rows').replaceChildren();$('compare-answers').replaceChildren();$('compare-export').hidden=true;const d=await api('/ml/comparisons',body);comparisonId=d.comparison_id;localStorage.setItem('ml-comparison',comparisonId);pollComparison();}catch(e){comparisonBusy=false;$('compare-start').disabled=false;$('compare-status').textContent=e.message;}};

const restoredComparison=new URLSearchParams(location.search).get('comparison')||localStorage.getItem('ml-comparison');
if(restoredComparison&&/^[a-f0-9]{32}$/.test(restoredComparison)){comparisonId=restoredComparison;comparisonBusy=true;$('compare-start').disabled=true;pollComparison();}

if(new URLSearchParams(location.search).get('developer')==='1'||new URLSearchParams(location.search).has('comparison'))$('developer-panel').open=true;

function costText(cost){
  if(!cost) return '未知';
  if(typeof cost.calculated_total_usd==='number') return '$'+cost.calculated_total_usd.toFixed(8);
  if(typeof cost.calculated_min_usd==='number'&&typeof cost.calculated_max_usd==='number')
    return '$'+cost.calculated_min_usd.toFixed(8)+'–$'+cost.calculated_max_usd.toFixed(8);
  const models=cost.models||[];
  const known=models.reduce((a,m)=>a+(typeof m.calculated_usd==='number'?m.calculated_usd:0),0);
  const unknown=models.filter(m=>typeof m.calculated_usd!=='number'&&!isLocalModel(m.model));
  if(known>0){
    const names=unknown.map(m=>m.model).filter(Boolean);
    return '$'+known.toFixed(8)+'（已知部分'+(names.length?'；未对账：'+names.join('、'):'')+'）';
  }
  return unknown.length?'不完整，不能算总费用':'$0.00000000（全部为本地推理）';
}
function isLocalModel(name){
  if(!name) return false;
  const n=String(name).toLowerCase();
  return n.startsWith('qwen')||n.includes('ollama')||n.includes(':local')||n.includes('-local');
}
function renderCosts(cost,billing){renderAccountBilling(billing);$('model-cost-total').textContent=cost?'官方公开单价核算合计：'+costText(cost)+'；账单实扣未获取。':'这份旧记录未采集费用所需信息，不能补造精确费用。';$('model-cost-rows').replaceChildren(...(cost?.models||[]).map(m=>{const tr=document.createElement('tr');const amount=isLocalModel(m.model)?'本地推理 · $0':typeof m.calculated_usd==='number'?'$'+m.calculated_usd.toFixed(8):typeof m.calculated_min_usd==='number'?'$'+m.calculated_min_usd.toFixed(8)+'–$'+m.calculated_max_usd.toFixed(8):'未知';for(const text of [m.model+' / '+m.calls,m.known_input_tokens+(m.unknown_usage_calls?'（不完整）':''),m.provider==='typesafe'?'不适用':m.requests.some(r=>r.cache_hit_tokens===null)?'未完整返回':m.known_cache_hit_tokens+' / '+m.known_cache_miss_tokens,m.known_output_tokens,amount,'未对账']){const td=document.createElement('td');td.textContent=text;tr.append(td);}return tr;}));$('model-cost-basis').textContent=cost?JSON.stringify({basis:cost.basis,verified:cost.price_verified_date,models:cost.models.map(m=>({model:m.model,sources:m.price_sources,conditions:m.requests.map(r=>({rates:r.rates,warnings:r.warnings}))}))},null,2):'无费用核算记录';}

let pricePolling=false;
async function refreshPriceStatus(force=false){if(pricePolling&&!force)return;pricePolling=true;try{const s=await api(force?'/ml/pricing/refresh':'/ml/pricing',force?{}:undefined);$('price-refresh').disabled=s.refreshing;const descriptions=Object.entries(s.prices).map(([name,p])=>name+'：'+(p.status==='live'?'已更新':p.status==='stale'?'价格已过期/刷新失败':'内置参考价，尚未在线核验')+' · '+(p.fetched_at?new Date(p.fetched_at).toLocaleString():'无在线更新时间'));$('price-status').textContent=(s.refreshing?'正在后台获取官方价格；':'')+descriptions.join('；')+'。自动检查间隔 '+s.refresh_interval_seconds/60+' 分钟。历史任务价格快照不变。';$('price-current').textContent=JSON.stringify(s.prices,null,2);if(s.refreshing)setTimeout(()=>refreshPriceStatus(),2000);}catch(e){$('price-status').textContent='价格刷新暂不可用，保留上次核定价格；不会把旧价格标成实时。';$('price-refresh').disabled=false;}finally{pricePolling=false;}}
$('price-refresh').onclick=()=>refreshPriceStatus(true);
refreshPriceStatus();
setInterval(()=>{if(!document.hidden)refreshPriceStatus();},300000);

function renderAssociation(evidence){$('association-block').hidden=!evidence||!evidence.features?.length;if(!evidence)return;$('association-note').textContent='仅使用训练样本 '+evidence.observations+' 条；目标类别 '+evidence.positive_label+'。这是分组观察比例，不是因果效应。';const rows=[];for(const f of evidence.features)for(const g of f.groups){const tr=document.createElement('tr');for(const text of [f.feature,g.group,g.n+(g.n<30?'（小样本）':''),g.positive_count,(g.rate*100).toFixed(1)+'%',(g.ci95_lower*100).toFixed(1)+'%–'+(g.ci95_upper*100).toFixed(1)+'%']){const td=document.createElement('td');td.textContent=text;tr.append(td);}rows.push(tr);}$('association-rows').replaceChildren(...rows);}

function renderAccountBilling(billing){
 const actual=billing?.task_actual_charge;
 $('actual-spend').textContent=actual?actual.amount+' '+actual.currency:'待对账';
 const observations=billing?.observations||[];if(observations.length&&observations.every(r=>Number(r.decrease)===0)){$('account-window').textContent='快照精度内未观察到余额变化，不能据此认定零费用；本轮精确实扣仍待账单核对。';return;}
 $('account-window').textContent=observations.length?'实际账户窗口变化：'+observations.map(r=>r.decrease+' '+r.currency).join(' / ')+'；含并发消费、充值或赠额变动，非本轮精确实扣。':'尚无可用账户余额观测；不以官方单价估算冒充实扣。';
}

/* --- iterative test: per-round charts (yuyu) --- */
let EVO_DATA=null, EVO_METRIC='tokens';
const EVO_ROUND_METRICS={
  tokens:{label:'每轮 token 消耗量',short:'token',color:'#b7ff50',
          pick:g=>g.reduce((a,s)=>a+(s.tokens||0),0),fmt:v=>String(Math.round(v)),unit:'token',
          note:'这一轮 7 个任务真正消耗的 token 合计；转回的步骤不调用本地模型，记为 0'},
  seconds:{label:'每轮耗时',short:'秒',color:'#7fd1ff',
           pick:g=>g.reduce((a,s)=>a+(s.seconds||0),0),fmt:v=>v.toFixed(1),unit:'秒',
           note:'这一轮所有步骤的墙钟时间合计'},
  failures:{label:'每轮错误步骤',short:'次',color:'#ff8a5c',
            pick:g=>g.filter(s=>s.attempted&&!s.ok).length,fmt:v=>String(Math.round(v)),unit:'次',
            note:'这一轮里本地尝试失败的步数；能力表学会之后不再增加'},
  evidence:{label:'有证据路由',short:'%',color:'#ffd166',
            pick:g=>g.filter(s=>s.decision!=='unproven_explore').length/g.length*100,fmt:v=>Math.round(v)+'%',unit:'%',
            note:'这一轮里"先查能力表再派"的步数占比；越高说明越不是瞎试'}
};

function evoRoundChart(steps){
  const metric=EVO_ROUND_METRICS[EVO_METRIC];
  const W=680,H=250,L=72,R=24,T=32,B=58;
  const rounds=[...new Set(steps.map(s=>s.round))];
  const data=rounds.map(r=>{const g=steps.filter(s=>s.round===r);
    return {round:r,value:metric.pick(g),fails:g.filter(s=>s.attempted&&!s.ok).length,
            calls:g.filter(s=>s.attempted).length,tokens:g.reduce((a,s)=>a+(s.tokens||0),0)};});
  const max=Math.max(1,...data.map(d=>d.value));
  const bw=(W-L-R)/data.length;
  let out='';
  [0,0.25,0.5,0.75,1].forEach(f=>{
    const y=T+(H-T-B)*(1-f);
    out+=`<line x1="${L}" y1="${y}" x2="${W-R}" y2="${y}" stroke="#16402a"/>`;
    out+=`<text x="${L-8}" y="${y+4}" fill="#7fbf98" font-size="11" text-anchor="end">${metric.fmt(max*f)}</text>`;
  });
  data.forEach((d,i)=>{
    const h=(H-T-B)*(d.value/max);
    const x=L+i*bw+bw*0.22,w=bw*0.56,y=H-B-h;
    out+=`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" fill="${metric.color}" opacity="0.85"/>`;
    out+=`<text x="${x+w/2}" y="${y-10}" fill="${metric.color}" font-size="15" font-weight="700" text-anchor="middle">${metric.fmt(d.value)}</text>`;
    out+=`<text x="${x+w/2}" y="${H-B+22}" fill="#9fd6b3" font-size="12" text-anchor="middle">第 ${d.round} 轮</text>`;
    out+=`<text x="${x+w/2}" y="${H-B+38}" fill="#7fbf98" font-size="11" text-anchor="middle">${d.tokens} token · 失败 ${d.fails}/${d.calls}</text>`;
  });
  out+=`<text x="${W-R}" y="${T-12}" fill="${metric.color}" font-size="12" text-anchor="end">${metric.label}（每轮）</text>`;
  return out;
}

async function renderEvolution(){
  if(!document.getElementById('evolution-card')) return;
  let data;
  try{data=await (await fetch('/static/evolution.json',{cache:'no-store'})).json();}catch(e){return;}
  const steps=data.steps||[];
  const note=document.getElementById('evo-note');
  if(!steps.length){note.textContent='数据为空。';return;}
  EVO_DATA=data;
  const s=data.summary||{};
  note.textContent='模型 '+(data.model||'')+' · 模式 '+(data.mode_label||'')+' · '+steps.length+' 步 / '+
    (data.rounds||0)+' 轮 · 通过 '+(s.passed||0)+' · 失败 '+(s.wasted_local_calls||0)+
    ' 次 · 转交 '+(s.delegated||0)+' 次 · token 合计 '+(s.tokens||0);
  document.getElementById('evo-chart').innerHTML=evoRoundChart(steps);
  document.getElementById('evo-caption').textContent=EVO_ROUND_METRICS[EVO_METRIC].note;
  document.querySelectorAll('.evo-tab').forEach(b=>b.classList.toggle('is-on',b.dataset.metric===EVO_METRIC));
  document.getElementById('evo-rows').getElementsByTagName('tbody')[0].innerHTML=steps.map(x=>
    `<tr><td>${x.step}</td><td>${x.round}</td><td>${x.dataset}</td><td>${x.decision}</td>`+
    `<td>${x.attempted?(x.ok?'通过':'失败'):'转交'}</td><td>${x.seconds}s</td><td>${x.tokens||0}</td></tr>`).join('');
}

async function evolutionCall(path,body){
  const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
  if(!r.ok) throw new Error((await r.json().catch(()=>({}))).detail||('HTTP '+r.status));
  return r.json();
}
async function pollEvolution(){
  const status=document.getElementById('evo-status');
  for(let i=0;i<180;i++){
    const s=await (await fetch('/ml/evolution/status',{cache:'no-store'})).json();
    status.textContent=s.state==='running'?('测验中… '+s.elapsed_hint):(s.state==='done'?'测验完成':'测验'+s.state);
    if(s.state!=='running'){await renderEvolution();return;}
    await new Promise(r=>setTimeout(r,2000));
  }
  status.textContent='测验超时，看日志。';
}
function bindEvolution(){
  const faulty=document.getElementById('evo-run-faulty');
  if(!faulty) return;
  faulty.onclick=async()=>{try{await evolutionCall('/ml/evolution/run',{mode:'faulty'});pollEvolution();}catch(e){document.getElementById('evo-status').textContent='失败：'+e.message;}};
  document.getElementById('evo-run-clean').onclick=async()=>{try{await evolutionCall('/ml/evolution/run',{mode:'clean'});pollEvolution();}catch(e){document.getElementById('evo-status').textContent='失败：'+e.message;}};
  document.getElementById('evo-rollback').onclick=async()=>{try{const r=await evolutionCall('/ml/evolution/rollback',{});document.getElementById('evo-status').textContent=r.restored||'已回退';await renderEvolution();}catch(e){document.getElementById('evo-status').textContent='失败：'+e.message;}};
  document.querySelectorAll('.evo-tab').forEach(b=>{b.onclick=()=>{EVO_METRIC=b.dataset.metric;renderEvolution();};});
}
document.addEventListener('DOMContentLoaded',()=>{renderEvolution();bindEvolution();});
