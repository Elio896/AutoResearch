'use strict';
const $=id=>document.getElementById(id);
let current=null, busy=false;
const labels={before:'纠错前',after:'下一会话',revoked:'撤销后负对照'};
async function request(url,body){const r=await fetch(url,{method:body===undefined?'GET':'POST',headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw new Error(typeof data.detail==='string'?data.detail:'请求失败，请检查输入。');return data;}
function paint(){
 $('start').disabled=busy;
 $('remember').disabled=busy||current?.phase!=='before';
 $('next').disabled=busy||current?.phase!=='feedback';
 $('revoke').disabled=busy||current?.phase!=='after';
 $('rollback').disabled=busy||!current||!!current.strategy_rollback;
 if(!current)return;
 $('state').textContent=busy?'执行真实实验…':({before:'等待纠错',feedback:'记忆已保存',after:'下一会话已验证',revoked:'撤销负对照完成'})[current.phase];
 $('before-count').textContent=current.before?.duplicate_fits??'—';$('after-count').textContent=current.after?.duplicate_fits??'—';$('after-score').textContent=current.after?current.after.score.toFixed(4):'—';
 const rows=[];for(const key of ['before','after','revoked']){const r=current[key];if(!r)continue;const tr=document.createElement('tr');for(const value of [labels[key],r.fits,r.duplicate_fits,r.blocked,r.score.toFixed(4),r.seconds.toFixed(2)]){const td=document.createElement('td');td.textContent=value;tr.append(td);}rows.push(tr);}$('rows').replaceChildren(...rows);
 $('evidence').textContent=JSON.stringify({dataset_sha256:current.dataset_sha256,source:current.source,feedback:current.feedback,memory_events:current.memory_events,research_events:current.after?.events??current.before?.events},null,2);
 $('feedback-status').textContent=current.feedback?({active:'已启用：下一会话将重新加载此规则。',pending:'反馈待明确；可修改为具体要求后重新保存。',rejected:'不允许的规则，未启用。',rejected_conflict:'规则冲突，未启用。'})[current.feedback.status]:'请指出重复实验的问题。';
 $('download').hidden=false;$('download').href=`/showcase/runs/${current.run_id}/evidence`;
 if(current.strategy_rollback)$('rollback-result').textContent=`已在独立副本从 ${current.strategy_rollback.from} 恢复到 ${current.strategy_rollback.to}，回滚事件已保存。`;
}
async function action(kind){if(busy)return;busy=true;paint();$('message').textContent='正在执行，请稍候…';try{if(kind==='start')current=await request('/showcase/runs',{});else current=await request(`/showcase/runs/${current.run_id}/${kind}`,kind==='feedback'?{text:$('feedback').value}:{});localStorage.setItem('evoresearch-showcase',current.run_id);$('message').textContent='已完成，结果来自本次实际执行。';}catch(e){$('message').textContent=e.message;}finally{busy=false;paint();}}
$('start').onclick=()=>action('start');$('remember').onclick=()=>action('feedback');$('next').onclick=()=>action('next');$('revoke').onclick=()=>action('revoke');$('rollback').onclick=()=>action('strategy-rollback');
const saved=localStorage.getItem('evoresearch-showcase');if(saved&&/^[0-9a-f]{32}$/.test(saved))request(`/showcase/runs/${saved}`).then(data=>{current=data;paint();}).catch(()=>{});
