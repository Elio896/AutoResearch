// 测 Ollama 的 TTFT / tok/s / JSON 合规率
const BASE = 'http://127.0.0.1:11434';
const MODEL = 'qwen2.5:3b-instruct-q4_K_M';

async function chat(messages, extra = {}) {
  const t0 = Date.now();
  const r = await fetch(BASE + '/api/chat', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: MODEL, stream: false, messages, ...extra }),
  });
  const j = await r.json();
  return { j, wall: (Date.now() - t0) / 1000 };
}

// 1) TTFT + tok/s（长 prompt）
const longText = ('实验报告说明：本次实验研究表格数据的分类任务，数据包含数值特征与类别特征，' +
  '需要处理缺失值、异常值、类别不平衡以及潜在的数据泄漏问题。我们计划比较逻辑回归、' +
  '随机森林与梯度提升三类模型，并以宏平均 F1 作为主指标，同时记录训练耗时与内存占用。' +
  '每一轮实验需记录假设、配方、结果、判定与成本五项证据，未通过校验的实验不得写入最终结论。').repeat(4);

const r1 = await chat([
  { role: 'user', content: '把下面这段文字读完，然后只回答 OK：\n' + longText },
], { options: { num_predict: 64 } });

const j1 = r1.j;
const promptTok = j1.prompt_eval_count || 0;
const promptNs = j1.prompt_eval_duration || 0;
const evalTok = j1.eval_count || 0;
const evalNs = j1.eval_duration || 0;

console.log('=== TTFT / 速度 ===');
console.log('prompt_eval_count    :', promptTok);
console.log('prompt_eval_duration :', promptNs, 'ns');
console.log('TTFT                 :', (promptNs / 1e9).toFixed(3), 's');
console.log('eval_count           :', evalTok);
console.log('eval_duration        :', evalNs, 'ns');
console.log('tok/s                :', evalNs ? (evalTok / (evalNs / 1e9)).toFixed(2) : 'n/a');
console.log('总墙钟               :', r1.wall.toFixed(2), 's');

// 2) JSON 合规率
const sysPrompt = '只输出 JSON，格式 {"rid":"R1|R2|R3","why":"十五字以内"}';
const userPrompt = '候选：R1 标准化+逻辑回归；R2 目标编码+GBDT；R3 缺失填充+随机森林。选一个。';

async function jsonRate(useFormat, n = 20) {
  let ok = 0;
  const fails = [];
  for (let i = 0; i < n; i++) {
    const extra = useFormat ? { format: 'json' } : {};
    const { j } = await chat([
      { role: 'system', content: sysPrompt },
      { role: 'user', content: userPrompt },
    ], extra);
    const content = j.message?.content ?? '';
    try {
      JSON.parse(content);
      ok++;
    } catch {
      fails.push(content.slice(0, 40));
    }
  }
  return { ok, n, fails };
}

console.log('\n=== JSON 合规率（20 次）===');
const withFmt = await jsonRate(true, 20);
console.log(`带 format:"json" : ${withFmt.ok}/20`);
const noFmt = await jsonRate(false, 20);
console.log(`不带 format      : ${noFmt.ok}/20`);
if (noFmt.fails.length) console.log('失败样例:', JSON.stringify(noFmt.fails.slice(0, 3)));
