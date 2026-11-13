// Real headless Edge verification of our local workbench. No npm packages.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawn} from 'node:child_process';

const base = process.env.EVORESEARCH_TEST_URL || 'http://127.0.0.1:8001';
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'evoresearch-edge-'));
const browser = spawn('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', [
  '--headless=new', '--disable-gpu', '--no-sandbox', '--no-first-run', '--no-default-browser-check',
  `--user-data-dir=${profile}`, '--remote-debugging-port=0', base
], {windowsHide: true, stdio: 'ignore'});
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(test, timeout = 20000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) { const value = await test(); if (value) return value; await pause(200); }
  throw new Error('Timed out waiting for workbench state');
}
let socket;
try {
  const active = path.join(profile, 'DevToolsActivePort');
  await until(() => fs.existsSync(active));
  const port = fs.readFileSync(active, 'utf8').split('\n')[0];
  const tabs = await until(async () => {
    try { return (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).filter(tab => tab.type === 'page' && tab.url.startsWith(base)); }
    catch { return false; }
  });
  socket = new WebSocket(tabs[0].webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.addEventListener('open', resolve, {once: true}); socket.addEventListener('error', reject, {once: true}); });
  let nextId = 0; const pending = new Map(); const errors = [];
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.text);
    if (pending.has(message.id)) { const {resolve, reject, timer} = pending.get(message.id); clearTimeout(timer); pending.delete(message.id); message.error ? reject(new Error(JSON.stringify(message.error))) : resolve(message.result); }
  });
  socket.addEventListener('close', event => { for (const entry of pending.values()) { clearTimeout(entry.timer); entry.reject(new Error(`Browser websocket closed: ${event.code}`)); } pending.clear(); });
  function cdp(method, params = {}) {
    return new Promise((resolve, reject) => { const id = ++nextId; const timer = setTimeout(() => { pending.delete(id); reject(new Error(`Browser command timeout: ${method}`)); }, 15000); pending.set(id, {resolve, reject, timer}); socket.send(JSON.stringify({id, method, params})); });
  }
  async function evaluate(expression) {
    const result = await cdp('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  }
  await cdp('Runtime.enable'); await cdp('Page.enable');
  await cdp('Emulation.setDeviceMetricsOverride', {width: 1600, height: 1100, deviceScaleFactor: 1, mobile: false});
  await until(() => evaluate("document.getElementById('connection')?.textContent.includes('已连接')"));
  await evaluate("document.getElementById('use-demo').click()");
  await until(() => evaluate("!document.getElementById('start-research').disabled"));
  const previousId = await evaluate("localStorage.getItem('evoresearch-task')");
  await evaluate("document.getElementById('max-experiments').value='2';document.getElementById('max-calls').value='2';document.getElementById('max-seconds').value='45';document.getElementById('start-research').click()");
  const taskId = await until(async () => { const id = await evaluate("localStorage.getItem('evoresearch-task')"); return id && id !== previousId ? id : false; });
  const finalTask = await until(async () => {
    const task = await (await fetch(`${base}/tasks/${taskId}`)).json();
    return ['completed', 'failed'].includes(task.research_job?.state) ? task : false;
  }, 60000);
  assert.equal(finalTask.research_job.state, 'completed');
  assert.ok(finalTask.experiments.length >= 1);
  assert.equal(finalTask.research_report.cloud_calls, 0);
  assert.equal(finalTask.final_result, null);
  await until(() => evaluate("document.querySelectorAll('#experiment-rows tr').length>0"));
  const desktopOverflow = await evaluate('document.documentElement.scrollWidth > innerWidth');
  const screenshot = await cdp('Page.captureScreenshot', {format: 'png', captureBeyondViewport: true});
  fs.writeFileSync('artifacts/workbench-desktop.png', Buffer.from(screenshot.data, 'base64'));
  await cdp('Page.reload');
  await until(() => evaluate("document.querySelectorAll('#experiment-rows tr').length>0"));
  assert.equal(await evaluate("localStorage.getItem('evoresearch-task')"), taskId);
  await cdp('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 1, mobile: true});
  const mobileOverflow = await evaluate('document.documentElement.scrollWidth > innerWidth');
  const mobileShot = await cdp('Page.captureScreenshot', {format: 'png', captureBeyondViewport: true});
  fs.writeFileSync('artifacts/workbench-mobile.png', Buffer.from(mobileShot.data, 'base64'));
  assert.equal(desktopOverflow, false); assert.equal(mobileOverflow, false);
  assert.deepEqual(errors, []);
  const result = {task_id: taskId, experiments: finalTask.experiments.length,
    best_validation_f1: Math.max(...finalTask.experiments.filter(e => e.result).map(e => e.result.validation_f1)),
    local_calls: finalTask.research_report.model_calls, cloud_calls: 0, test_evaluated: false,
    reload_restored: true, desktop_overflow: desktopOverflow, mobile_overflow: mobileOverflow, page_errors: errors};
  fs.writeFileSync('artifacts/workbench_verification.json', JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result, null, 2));
  await cdp('Browser.close');
} finally {
  socket?.close();
  browser.kill();
}
