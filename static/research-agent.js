'use strict';
const $ = id => document.getElementById(id);
let currentRun = null;
let polling = false;
let currentDatasetSha = null;
let currentRunVerified = false;
const number = (value, digits = 4) => value == null ? '未知' : Number(value).toFixed(digits);
function node(tag, text, cls) { const e = document.createElement(tag); if (text != null) e.textContent = text; if (cls) e.className = cls; return e; }
async function request(url, options) { const r = await fetch(url, options); const value = await r.json(); if (!r.ok) throw new Error(typeof value.detail === 'string' ? value.detail : '请求参数不正确，请检查预算与数据。'); return value; }
function renderRound(row) {
  const section = node('section', null, 'round');
  const head = node('div', null, 'round-head'); head.append(node('h3', `第 ${row.round} 轮 · ${row.incumbent} → ${row.candidate || '无有效候选'}`));
  const labels = {upgrade: '证据通过 · 升级', keep: '保留当前策略', rollback: '监控退化 · 回滚'};
  head.append(node('span', labels[row.decision] || row.decision, 'pill ' + (row.decision === 'upgrade' ? 'up' : 'keep'))); section.append(head);
  if (row.hypothesis) section.append(node('p', row.hypothesis.what));
  const evaluation = row.evaluation;
  if (evaluation) {
    const table = node('table'), thead = node('thead'), tr = node('tr');
    ['对照指标', '当前策略', '候选策略'].forEach(t => tr.append(node('th', t))); thead.append(tr); table.append(thead);
    const body = node('tbody');
    [['验证宏 F1', number(evaluation.old.mean_best_score), number(evaluation.new.mean_best_score)],
     ['任务完成', `${evaluation.old.completed}/${evaluation.old.task_count}`, `${evaluation.new.completed}/${evaluation.new.task_count}`],
     ['合法提案率', number(evaluation.old.legal_output_rate, 2), number(evaluation.new.legal_output_rate, 2)],
     ['模型调用', evaluation.old.cloud_calls, evaluation.new.cloud_calls],
     ['实际 token', evaluation.old.model_tokens ?? '未知', evaluation.new.model_tokens ?? '未知'],
     ['执行时间 / 秒', number(evaluation.old.elapsed_seconds, 2), number(evaluation.new.elapsed_seconds, 2)]].forEach(values => { const r = node('tr'); values.forEach(v => r.append(node('td', v))); body.append(r); });
    table.append(body); const wrap = node('div', null, 'table-wrap'); wrap.append(table); section.append(wrap);
    const failed = [...evaluation.failed_checks, ...evaluation.unknown_checks];
    section.append(node('div', failed.length ? `未通过 / 证据不足：${failed.join('、')}` : '✓ 所有冻结检查通过；至少一项指标有可测改善。', 'checkline' + (failed.length ? ' fail' : '')));
  } else section.append(node('p', row.reason || '尚未获得完整对照证据。'));
  if (row.monitor) section.append(node('p', `上线后独立监控：${row.monitor.verdict} · ${row.monitor.reason}`));
  return section;
}
const eventNames = {research_completed: '研究任务完成', evolution_round_completed: '策略评测完成', campaign_completed: '研究闭环完成', candidate_not_applied: '候选未应用', stopped: '预算停止'};
function render(state) {
  currentRunVerified = true;
  $('status').textContent = ({queued:'已排队', running:'正在研究', completed:'研究完成', failed:'未完成 · 证据已保留'})[state.state] || state.state;
  const report = state.report || state.progress;
  const rounds = report?.rounds || [];
  $('empty').hidden = rounds.length > 0;
  $('round-results').replaceChildren(...rounds.map(renderRound));
  $('rounds').textContent = `${rounds.length} / 2`;
  $('decisions').textContent = rounds.length ? rounds.map(r => `第${r.round}轮：${({upgrade:'升级',keep:'保留',rollback:'回滚'})[r.decision] || r.decision}`).join(' · ') : '等待独立评测';
  if (state.report) {
    const r = state.report, u = r.model_usage;
    $('score').textContent = r.final_test ? number(r.final_test.test_score) : '未评估';
    $('usage').textContent = `${u.calls} / ${u.unreconciled_calls ? '部分未知' : u.known_tokens}`;
    $('cost').textContent = u.calls ? '费用未知 · 以供应商账单为准' : '无模型调用 · 不产生 API token';
    $('fits').textContent = `${r.training_fits} 次`;
    $('duration').textContent = `全程 ${number(r.elapsed_seconds, 2)} 秒 · ${r.mode === 'rules' ? '程序规则，非模型输出' : '单模型共享预算'}`;
    const development = Object.entries(r.datasets_sha256 || {}).find(([name]) => name.includes('development'));
    currentDatasetSha = development ? development[1] : null;
  }
  const active = state.current_strategy;
  $('active-policy').textContent = active ? `当前 ${active.strategy_id}` : '策略准备中';
  $('restore').disabled = state.state !== 'completed' || !active?.parent_id;
  $('export').hidden = !['completed','failed'].includes(state.state);
  $('export').href = `/system/campaigns/${currentRun}/export`;
  if (state.events?.length) $('events').replaceChildren(...state.events.slice(-25).map(e => {
    const li = node('li'); li.append(node('time', `${number(e.elapsed_seconds, 1)}s`), node('strong', eventNames[e.event] || e.event));
    const detail = e.label || (e.round ? `第 ${e.round} 轮 · ${e.decision}` : '');
    if (detail) li.append(node('span', ` · ${detail}`)); return li;
  }));
  if (state.state === 'failed') $('error').textContent = state.reason;
}
async function poll() {
  if (polling || !currentRun) return;
  polling = true;
  try {
    const state = await request(`/system/campaigns/${currentRun}`); render(state);
    const busy = ['queued','running'].includes(state.state); $('start').disabled = busy;
    if (busy) setTimeout(poll, 1200);
  } catch(e) { $('error').textContent = e.message; $('start').disabled = false; }
  finally { polling = false; }
}
$('campaign-form').addEventListener('submit', async e => {
  e.preventDefault(); $('error').textContent = ''; $('start').disabled = true;
  try {
    const body = {mode:$('mode').value, max_calls:Number($('max-calls').value), max_tokens:Number($('max-tokens').value), max_seconds:Number($('max-seconds').value), max_fits:Number($('max-fits').value), target:$('target').value};
    const file = $('development-file').files[0];
    if (file) { if (file.size > 2000000) throw new Error('CSV 文件不能超过 2 MB。'); body.csv_text = await file.text(); }
    const result = await request('/system/campaigns', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    currentRun = result.run_id; localStorage.setItem('evoresearch-campaign', currentRun);
    $('score').textContent = '—'; $('usage').textContent = '—'; $('fits').textContent = '—'; await poll();
  } catch(error) { $('error').textContent = error.message; $('start').disabled = false; }
});
$('restore').addEventListener('click', async () => {
  $('restore').disabled = true;
  try { await request(`/system/campaigns/${currentRun}/rollback`, {method:'POST'}); $('restore-note').textContent = '已恢复上一稳定策略。原始实验和测试记录保持可追溯。'; await poll(); }
  catch(e) { $('error').textContent = e.message; }
});
$('mode').addEventListener('change', () => { $('mode-note').textContent = $('mode').value === 'rules' ? '真实训练与评测；规划和策略候选由程序规则产生，不代表 LLM 自主研究能力。' : '规划与失败诊断共用同一个模型及调用/token 预算。凭据从服务器环境读取。'; });
request('/system/config').then(config => { const option = $('mode').options[1]; option.disabled = !config.single_model_configured; option.textContent = config.single_model_configured ? '单模型 Agent · 已配置服务' : '单模型 Agent · 待配置服务'; }).catch(e => { $('error').textContent = e.message; });
currentRun = localStorage.getItem('evoresearch-campaign');
if (currentRun && /^[0-9a-f]{32}$/.test(currentRun)) poll();

async function loadFeedback() {
  const data = await request('/system/feedback');
  const active = data.records.filter(r => r.status === 'active');
  const blocked = data.events.filter(e => e.kind === 'blocked').length;
  const repaired = data.events.filter(e => e.kind === 'auto_repaired').length;
  $('feedback-count').textContent = `${active.length} 条生效 · 已拦截 ${blocked} · 已修正 ${repaired}`;
  $('feedback-records').replaceChildren(...data.records.slice(-12).reverse().map(r => {
    const card = node('article', null, 'feedback-record');
    const status = {active:'已启用', pending:'待明确规则', rejected:'不允许的修改', rejected_conflict:'冲突 · 未启用', revoked:'已撤销'};
    card.append(node('strong', status[r.status] || r.status), node('p', r.text), node('small', r.scope === 'global' ? '之后所有研究' : '仅匹配的数据摘要'));
    if (r.status === 'active') {
      const revoke = node('button', '撤销');
      revoke.type = 'button';
      revoke.addEventListener('click', async () => {
        try { await request(`/system/feedback/${r.id}/revoke`, {method:'POST'}); await loadFeedback(); }
        catch(e) { $('feedback-notice').textContent = e.message; }
      });
      card.append(revoke);
    }
    return card;
  }));
}
$('feedback-form').addEventListener('submit', async e => {
  e.preventDefault();
  try {
    const kind = $('feedback-kind').value || null;
    const body = {text:$('feedback-text').value, kind, scope:$('feedback-scope').value};
    if (kind === 'forbid_model') body.value = $('feedback-value').value;
    if (body.scope === 'dataset') {
      if (!currentDatasetSha) throw new Error('请先运行一次研究，才能绑定开发数据的摘要。');
      body.dataset_sha256 = currentDatasetSha;
    }
    if (currentRun && currentRunVerified) body.source_run_id = currentRun;
    const result = await request('/system/feedback', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    $('feedback-notice').textContent = result.status === 'active' ? '已记住。下次研究开始时会加载，并在训练前执行检查；当前正在运行的研究保持原快照。' : result.status === 'pending' ? '已保存，但含义尚不明确。请选择具体纠错方式后重新提交。' : '这条反馈与执行边界或现有约束冲突，未启用。';
    await loadFeedback();
  } catch(e) { $('feedback-notice').textContent = e.message; }
});
loadFeedback().catch(e => { $('feedback-notice').textContent = e.message; });
request('/system/innovations').then(data => {
  $('innovation-cards').replaceChildren(...data.designs.map(item => { const card=node('article'); card.append(node('h3',item.title),node('p',item.detail)); return card; }));
  const r = data.evaluation;
  $('innovation-evidence').textContent = r ? `真实训练纠错回归：${r.next_session_cases} 个下次会话案例；无记忆 ${r.totals.no_memory.mistakes} 次错误，只有提示 ${r.totals.prompt_only.mistakes} 次，执行约束 ${r.totals.enforced.mistakes} 次；无关数据误拦截 ${r.unrelated_scope_blocks} 次。提案为脚本回放，非真实模型能力测试。` : '尚无纠错评测结果。设计说明不能代替实验证据。';
}).catch(e => { $('innovation-evidence').textContent=e.message; });
