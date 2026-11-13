'use strict';
const $ = id => document.getElementById(id);
const state = {csv: '', task: null, selected: localStorage.getItem('evoresearch-task') || '', busy: false, refreshing: false, serverReady: false};
const reasons = {experiment_budget: '实验预算已用完，保留验证集表现最好的方案。', local_call_budget: '本地调用预算已用完，需要进一步诊断。', validation_plateau: '验证指标连续未改善，需要进一步诊断。', local_unavailable: '本地模型暂不可用，已保留现有结果。', time_budget: '时间预算已用完，停止发起新工作。', experiment_failed: '候选训练失败，请查看实验记录。'};
const modelNames = {logistic_regression: '逻辑回归', random_forest: '随机森林'};
function toast(message, error = false) {
  $('toast').textContent = message; $('toast').className = 'toast' + (error ? ' error' : ''); $('toast').hidden = false;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => { $('toast').hidden = true; }, 6500);
}
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) {
    const messages = {'Autonomous research is active; manual actions are locked.': '自主研究正在进行，请等待完成。', 'Another local research job is active. Wait for it to finish.': '另一个研究任务正在运行，请稍后再试。', 'Task is not open; final test cannot be repeated.': '该任务已封存，不能重复评估。', 'Invalid dataset or experiment configuration.': '请检查 CSV 目标列、缺失标签和每类样本数量。'};
    throw new Error(messages[result.detail] || (typeof result.detail === 'string' ? result.detail : '请求失败，请检查填写内容。'));
  }
  return result;
}
function headerFields(csv) {
  const fields = []; let field = '', quoted = false;
  for (let i = 0; i < csv.length; i++) {
    const c = csv[i];
    if (c === '"') { if (quoted && csv[i + 1] === '"') { field += '"'; i++; } else quoted = !quoted; }
    else if (c === ',' && !quoted) { fields.push(field); field = ''; }
    else if ((c === '\n' || c === '\r') && !quoted) break;
    else field += c;
  }
  if (quoted) throw new Error('CSV 表头的引号没有闭合。');
  fields.push(field);
  if (fields.length < 2 || fields.some(name => !name.trim()) || new Set(fields).size !== fields.length) throw new Error('CSV 需要至少两列，列名不能为空或重复。');
  return fields;
}
function setCSV(csv, name, target) {
  csv = csv.replace(/^\uFEFF/, '');
  const size = new TextEncoder().encode(csv).length;
  if (size > 2000000) throw new Error('文件超过 2 MB，请选择更小的 CSV。');
  const fields = headerFields(csv); state.csv = csv;
  $('file-name').textContent = name; $('file-info').textContent = `${fields.length} 列 · ${(size / 1024).toFixed(1)} KB`;
  $('target-column').replaceChildren(...fields.map(name => { const option = document.createElement('option'); option.value = name; option.textContent = name; return option; }));
  $('target-column').value = fields.includes(target) ? target : fields.at(-1);
  $('target-column').disabled = false; updateControls();
}
async function loadFile(file) {
  if (!file) return;
  if (file.size > 2000000) { toast('文件超过 2 MB，请选择更小的 CSV。', true); return; }
  try { setCSV(new TextDecoder('utf-8', {fatal: true}).decode(await file.arrayBuffer()), file.name); }
  catch (error) { toast(error instanceof TypeError ? '请将 CSV 另存为 UTF-8 编码后再上传。' : error.message, true); }
}
function activeJob(task = state.task) { return ['queued', 'running'].includes(task?.research_job?.state); }
function updateControls() {
  $('start-research').disabled = !state.serverReady || !state.csv || state.busy || activeJob();
  const hasResult = state.task?.experiments.some(exp => exp.status === 'success');
  $('finalize').disabled = !state.serverReady || state.busy || !hasResult || state.task?.state !== 'open' || activeJob() || state.task?.risk_report?.blocked || state.task?.experiments.some(exp => exp.status === 'running');
  $('use-demo').disabled = state.busy;
  $('start-existing').disabled = !state.serverReady || state.busy || !state.task || state.task.state !== 'open' || state.task.experiments.length > 0 || Boolean(state.task.research_job) || state.task.risk_report?.blocked;
  $('save-review').disabled = !state.serverReady || state.busy || !state.task || state.task.state !== 'open' || state.task.experiments.length > 0 || activeJob();
}
function element(tag, text, className) { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (className) node.className = className; return node; }
function render(task) {
  state.task = task;
  const risk = task.risk_report;
  $('risk-panel').hidden = !risk || (!risk.findings?.length && !risk.blocked);
  $('risk-title').textContent = task.legacy_results_at_risk ? '历史结果存在数据风险，不能作为可靠结论' : risk?.blocked ? '先审核数据风险，再运行实验' : '数据风险已人工处理';
  $('risk-summary').textContent = task.legacy_results_at_risk ? '原始实验与分数保持不变。需要新建任务、处理风险后重新训练；不能覆盖历史配置。' : risk?.blocked ? '候选信号不是泄漏证明。请根据特征来源逐列选择排除或保留，并填写理由。已确认的泄漏只能排除。' : '所有实验使用同一份处理配置。审核通过不代表已排除所有潜在泄漏。';
  const lockedReview = task.state !== 'open' || task.experiments.length > 0 || activeJob(task);
  const signature = JSON.stringify([task.id, risk?.findings, risk?.decisions, lockedReview]);
  if ($('risk-findings').dataset.signature !== signature) {
    $('risk-findings').dataset.signature = signature;
    const signalNames = {identifier_name: '列名像标识符', unique_ordered_integer: '近乎唯一的有序整数', matches_training_target: '在训练集中与标签一致', strong_training_association: '训练集中与标签高度相关', unique_numeric_separates_training_classes: '训练样本按该列排序后类别完全分段'};
    $('risk-findings').replaceChildren(...(risk?.findings || []).map(finding => {
      const row = element('div', undefined, 'risk-finding'); row.dataset.column = finding.column;
      row.append(element('strong', `${finding.column} · ${finding.level === 'confirmed' ? '已确认泄漏' : '待核实候选'}`));
      row.append(element('small', finding.level === 'confirmed' ? finding.evidence : finding.signals.map(signal => signalNames[signal] || signal).join('；')));
      const select = document.createElement('select'); select.setAttribute('aria-label', `${finding.column} 处理方式`);
      [['', '请选择处理方式'], ['exclude', '排除该特征'], ['keep', '确认合法并保留']].forEach(([value, label]) => { const option = element('option', label); option.value = value; if (value === 'keep' && finding.level === 'confirmed') option.disabled = true; select.append(option); });
      const saved = risk.decisions?.[finding.column]; select.value = saved?.action || ''; select.disabled = lockedReview;
      const reason = document.createElement('input'); reason.type = 'text'; reason.maxLength = 500; reason.placeholder = '说明来源和决定理由（至少 3 个字符）'; reason.value = saved?.reason || ''; reason.disabled = lockedReview; reason.setAttribute('aria-label', `${finding.column} 决定理由`);
      row.append(select, reason); return row;
    }));
  }
  $('existing-task-controls').hidden = false;
  $('existing-task-hint').textContent = task.experiments.length > 0 ? '此任务已有实验。自主研究必须从新任务开始，请在左侧新建。' : risk?.blocked ? '处理完风险后，可启动这个已保存的任务。' : task.research_job ? '该任务已启动过自主研究，可查看完整轨迹。' : '这是未运行的任务，可以直接启动；实验次数沿用创建时预算。';
  const successful = task.experiments.filter(exp => exp.status === 'success');
  const best = successful.reduce((a, b) => !a || b.result.validation_f1 > a.result.validation_f1 ? b : a, null);
  $('best-score').textContent = best ? best.result.validation_f1.toFixed(4) : '—';
  $('best-model').textContent = best ? `${best.result.experiment_id} · ${modelNames[best.result.model] || best.result.model}` : '等待第一份实验结果';
  if (task.legacy_results_at_risk) { $('best-score').textContent = '待复核'; $('best-model').textContent = '历史指标保留，不能直接作为研究结论'; }
  $('experiment-count').replaceChildren(document.createTextNode(`${task.experiments.length} `), element('em', `/ ${task.max_experiments}`));
  const report = task.research_report;
  const attempts = Math.max(0, ...task.events.filter(event => ['proposal_accepted', 'proposal_rejected', 'local_error'].includes(event.event)).map(event => event.attempt || 0));
  $('model-calls').textContent = report?.model_calls ?? attempts;
  $('token-count').textContent = report ? `输入 ${report.reported_local_tokens?.prompt || 0} / 输出 ${report.reported_local_tokens?.completion || 0} tokens` : activeJob(task) ? '本地提案中 · 等待完整用量' : '尚无完整调用记录';
  $('test-state').textContent = task.state === 'finalized' ? '已评估' : task.state === 'open' ? '已隔离' : '已锁定';
  $('test-detail').textContent = task.final_result ? `最终测试 F1 ${task.final_result.test_f1.toFixed(4)}` : task.state === 'open' ? '未用于研究迭代' : '最终评估已锁定，不能重试';
  if (task.legacy_results_at_risk) { $('test-state').textContent = '结果有风险'; $('test-detail').textContent = '历史评估保留，不作为可靠结论'; }
  const status = risk?.blocked ? '需审核数据风险' : task.state === 'finalized' ? '已封存' : activeJob(task) ? '研究进行中' : report?.status === 'escalation_required' ? '需要进一步诊断' : report ? '研究已完成' : task.research_job?.state === 'failed' ? '研究中断' : '等待开始';
  $('run-status').textContent = status; $('run-status').className = 'badge' + (activeJob(task) ? ' running' : status === '需要进一步诊断' || status === '研究中断' ? ' warning' : '');
  $('empty-results').hidden = task.experiments.length > 0; $('results-content').hidden = !task.experiments.length;
  $('task-caption').textContent = `任务 ${task.id.slice(0, 8)} · 预测 ${task.target} · 固定随机种子 ${task.seed}`;
  $('experiment-rows').replaceChildren(...task.experiments.map(exp => {
    const row = document.createElement('tr'), label = document.createElement('td');
    label.append(element('strong', modelNames[exp.proposal.model] || exp.proposal.model), element('small', `exp_${String(exp.number).padStart(3, '0')}`)); label.title = JSON.stringify(exp.proposal.parameters);
    const score = element('td', exp.result ? exp.result.validation_f1.toFixed(4) : '—', 'score');
    const duration = element('td', exp.result ? `${exp.result.duration_seconds.toFixed(2)}s` : '—');
    const outcome = document.createElement('td'); outcome.append(element('span', task.legacy_results_at_risk ? '历史结果·有风险' : exp === best ? '当前最佳' : exp.status === 'success' ? '已完成' : exp.status === 'failed' ? '失败' : '训练中', exp === best && !task.legacy_results_at_risk ? 'row-best' : 'row-status'));
    row.append(label, score, duration, outcome); return row;
  }));
  $('research-outcome').textContent = report ? reasons[report.stop_reason] || '本轮研究已结束，请查看研究轨迹。' : activeJob(task) ? '正在用真实指标检验方案，请稍候。' : '实验结果已保存，可以查看或确认最终评估。';
  $('event-count').textContent = `${task.events.length} 条`;
  const descriptions = {
    research_queued: ['Manager / 调度', '任务已入队', '按设定预算执行本地研究。'],
    research_started: ['Manager / 调度', '研究开始', '先建立可比较的基线。'],
    experiment: ['Experimenter / 执行器', '真实训练完成', ''],
    proposal_accepted: ['Scientist / 本地千问', '候选方案通过校验', ''],
    proposal_rejected: ['Reviewer / 规则校验', '拒绝无效或重复方案', '该方案没有进入训练。'],
    local_error: ['Manager / 调度', '本地模型调用受阻', '失败请求也会计入调用预算。'],
    research_stopped: ['Manager / 调度', '本轮研究停止', ''],
    research_failed: ['Manager / 调度', '研究意外中断', '已完成的实验仍保留。'],
    experiment_failed: ['Reviewer / 规则校验', '训练未能完成', '保留之前的有效结果。']
  };
  $('timeline').replaceChildren(...task.events.map((event, index) => {
    const [role, title, detail] = descriptions[event.event] || ['研究记录', event.event, ''];
    const item = document.createElement('li'); item.append(element('small', `${String(index + 1).padStart(2, '0')} / ${role}`), element('strong', title));
    let message = detail;
    if (event.event === 'experiment') message = `${modelNames[event.result.model] || event.result.model} · 验证 F1 ${event.result.validation_f1.toFixed(4)}`;
    if (event.event === 'proposal_accepted') message = `${modelNames[event.proposal.model] || event.proposal.model} · ${JSON.stringify(event.proposal.parameters)}`;
    if (event.event === 'research_stopped') message = reasons[event.reason] || '已停止新增实验。';
    item.append(element('p', message)); return item;
  }));
  if (!task.events.length) $('timeline').append(element('li', '该任务暂无 Agent 轨迹；手动实验记录在中间列表。', 'timeline-empty'));
  $('final-explanation').textContent = task.final_result ? `已封存 · 最终测试 F1 ${task.final_result.test_f1.toFixed(4)}，不会再参与调参。` : '最终评估会选择验证集表现最好的方案，并封存这个任务。';
  if (task.legacy_results_at_risk) $('final-explanation').textContent = '历史结果已标记风险。需要新建任务并明确处理泄漏后重新研究。';
  updateControls();
}
async function loadTask(id) {
  state.selected = id; localStorage.setItem('evoresearch-task', id);
  const task = await api(`/tasks/${encodeURIComponent(id)}`); if (state.selected === id) render(task);
  document.querySelectorAll('.task-item').forEach(button => button.classList.toggle('selected', button.dataset.taskId === id));
}
async function refreshTasks() {
  const tasks = await api('/tasks');
  $('task-list').replaceChildren(...tasks.slice().reverse().map(task => {
    const button = element('button', undefined, 'task-item' + (state.selected === task.id ? ' selected' : '')); button.dataset.taskId = task.id;
    button.append(element('strong', `预测 ${task.target}`), element('small', `${task.id.slice(0, 8)} · ${task.state === 'finalized' ? '已封存' : task.state === 'open' ? '可查看' : '已锁定'}`));
    button.onclick = () => loadTask(task.id).catch(error => toast(error.message, true)); return button;
  }));
  if (!tasks.length) $('task-list').append(element('p', '还没有任务，试试示例数据。', 'muted')); return tasks;
}
$('csv-file').onchange = event => loadFile(event.target.files[0]);
['dragenter', 'dragover'].forEach(name => $('drop-zone').addEventListener(name, event => { event.preventDefault(); $('drop-zone').classList.add('dragging'); }));
['dragleave', 'drop'].forEach(name => $('drop-zone').addEventListener(name, event => { event.preventDefault(); $('drop-zone').classList.remove('dragging'); if (name === 'drop') loadFile(event.dataTransfer.files[0]); }));
$('use-demo').onclick = async () => { try { const demo = await api('/demo'); setCSV(demo.csv_text, demo.name, demo.target); toast('示例已载入，可以开始研究。'); } catch (error) { toast(error.message, true); } };
$('refresh-tasks').onclick = () => refreshTasks().catch(error => toast(error.message, true));
$('research-form').onsubmit = async event => {
  event.preventDefault(); if (state.busy || !state.csv || activeJob()) return;
  state.busy = true; updateControls();
  try {
    const created = await api('/tasks', {csv_text: state.csv, target: $('target-column').value, max_experiments: Number($('max-experiments').value)});
    await loadTask(created.task_id); await refreshTasks();
    if (state.task.risk_report?.blocked) { toast('数据风险需要审核，暂未启动训练。'); return; }
    await api(`/tasks/${created.task_id}/research`, {max_calls: Number($('max-calls').value), max_seconds: Number($('max-seconds').value)});
    await loadTask(created.task_id); toast('研究已启动，实验结果会自动更新。');
  } catch (error) { toast(error.message, true); }
  finally { state.busy = false; updateControls(); }
};
$('finalize').onclick = () => $('final-dialog').showModal();
$('save-review').onclick = async () => {
  if (!state.selected || state.busy) return;
  const decisions = {...(state.task.risk_report?.decisions || {})};
  for (const row of document.querySelectorAll('.risk-finding')) {
    const action = row.querySelector('select').value;
    const reason = row.querySelector('input').value.trim();
    if (!action || reason.length < 3) { toast('请为每个候选列选择处理方式，并说明理由。', true); return; }
    decisions[row.dataset.column] = {action, reason};
  }
  state.busy = true; updateControls();
  try { await api(`/tasks/${state.selected}/review`, {decisions}); await loadTask(state.selected); toast('决定已保存；可以点击“启动此任务研究”。'); }
  catch (error) { toast(error.message, true); }
  finally { state.busy = false; updateControls(); }
};
$('start-existing').onclick = async () => {
  if (state.busy || $('start-existing').disabled) return;
  const id = state.selected; state.busy = true; updateControls();
  try { await api(`/tasks/${id}/research`, {max_calls: Number($('max-calls').value), max_seconds: Number($('max-seconds').value)}); await loadTask(id); toast('已启动此任务的自主研究。'); }
  catch (error) { toast(error.message, true); }
  finally { state.busy = false; updateControls(); }
};
$('cancel-final').onclick = () => $('final-dialog').close();
$('confirm-final').onclick = async () => {
  $('final-dialog').close(); if (!state.selected || state.busy) return;
  state.busy = true; updateControls(); const id = state.selected;
  try { await api(`/tasks/${id}/finalize`, {}); await loadTask(id); await refreshTasks(); toast('最终评估已完成，任务已封存。'); }
  catch (error) { toast(error.message, true); }
  finally { state.busy = false; updateControls(); }
};
async function initialize() {
  try { await api('/health'); const schema = await api('/openapi.json'); if (!schema.paths?.['/tasks/{task_id}/review']) throw new Error('当前服务未加载泄漏守卫，请打开 http://127.0.0.1:8002/ 或重启新版服务。'); state.serverReady = true; $('connection').textContent = '● 服务已连接'; const tasks = await refreshTasks(); const selected = tasks.find(task => task.id === state.selected) || tasks.at(-1); if (selected) await loadTask(selected.id); updateControls(); }
  catch (error) { state.serverReady = false; updateControls(); $('connection').textContent = '服务未就绪'; toast(error.message || '无法连接后端，请确认本地服务正在运行。', true); }
}
setInterval(async () => {
  if (!state.serverReady || !state.selected || state.refreshing || document.hidden) return;
  state.refreshing = true;
  try { const task = await api(`/tasks/${state.selected}`); if (task.id === state.selected) render(task); $('connection').textContent = '● 服务已连接'; }
  catch { $('connection').textContent = '连接中断 · 正在重试'; }
  finally { state.refreshing = false; }
}, 1800);
initialize();
