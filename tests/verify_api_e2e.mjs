// EvoResearch API 端到端验证
import fs from 'node:fs';

const BASE = 'http://127.0.0.1:8000';
const csv = fs.readFileSync('C:/workspace/EvoResearch/data/iris.csv', 'utf8');

async function call(method, path, body) {
  const r = await fetch(BASE + path, {
    method,
    headers: { 'content-type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  let data;
  try { data = await r.json(); } catch { data = null; }
  return { status: r.status, data };
}

console.log('1) health:', JSON.stringify((await call('GET', '/health')).data));

const created = await call('POST', '/tasks', { csv_text: csv, target: 'target', max_experiments: 3 });
console.log('2) create task:', created.status, JSON.stringify(created.data));
const taskId = created.data?.task_id;
if (!taskId) { console.log('ABORT: no task_id'); process.exit(1); }

const exp1 = await call('POST', `/tasks/${taskId}/experiments`, { model: 'logistic_regression', parameters: {} });
console.log('3) experiment 1:', exp1.status, JSON.stringify(exp1.data).slice(0, 400));

const exp2 = await call('POST', `/tasks/${taskId}/experiments`, {
  model: 'random_forest',
  parameters: { n_estimators: 50, max_depth: 5 },
});
console.log('4) experiment 2:', exp2.status, JSON.stringify(exp2.data).slice(0, 400));

const shown = await call('GET', `/tasks/${taskId}`);
console.log('5) history:', shown.status, JSON.stringify(shown.data).slice(0, 600));

const fin = await call('POST', `/tasks/${taskId}/finalize`);
console.log('6) finalize:', fin.status, JSON.stringify(fin.data).slice(0, 400));

const fin2 = await call('POST', `/tasks/${taskId}/finalize`);
console.log('7) finalize again (should be 409):', fin2.status, JSON.stringify(fin2.data).slice(0, 200));

const list = await call('GET', '/tasks');
console.log('8) task list:', list.status, 'count =', Array.isArray(list.data) ? list.data.length : 'n/a');
