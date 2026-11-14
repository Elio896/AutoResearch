// Verify the actual display formatter against saved live answers, without APIs.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync('static/ml-app.js','utf8');
const start=source.indexOf('function userFacingAnswer('),end=source.indexOf('function renderResearchAnswer(',start);
assert(start>=0&&end>start);const context={};vm.createContext(context);
vm.runInContext(source.slice(start,end),context);
let checked=0;
const root=process.argv[2]||'artifacts/contest-final-v23-r3-20261115';
const rows=JSON.parse(fs.readFileSync(root+'/qa-results.json','utf8'));
for(const name of ['qa_10','qa_30']){
 const row=rows.find(r=>r.case===name);assert(row);
 const answer=context.userFacingAnswer({review:{answer:row.answer},research_outputs:{question:'调参记录放在开发者面板。'}});
 assert(answer.length>50);assert(!/exp_\d+|random_forest|hist_gradient_boosting|logistic_regression/i.test(answer));
 for(const word of name==='qa_10'?['发现了什么','证据是什么','有哪些局限']:['直接回答','证据','预测结果','局限'])assert(answer.includes(word));
 assert(answer.includes('/predictions'));checked++;
}
const ordinary=rows.find(r=>r.case==='qa_2').answer;
assert.equal(context.userFacingAnswer({review:{answer:ordinary}}),ordinary);
const scope=rows.find(r=>r.case==='qa_8');
const clarified=context.userFacingAnswer({review:{answer:scope.answer},research_outputs:{records_predicted:1000}});
assert(!clarified.includes('未使用训练时未见过的全新数据'));
assert(clarified.includes('未用于模型拟合'));
const live=JSON.parse(fs.readFileSync('artifacts/ml/cf600d90450f48f69faf9510bf877ed4/report.json','utf8'));
const visible=context.userFacingAnswer(live);
assert(visible.includes('完成 2 轮研究'));
assert(visible.includes('保留原有最佳方案'));
assert(visible.includes('1000 条验证记录'));
assert(!/exp_\d+|hist_gradient_boosting|max_iter/.test(visible));
const followup=JSON.parse(fs.readFileSync('artifacts/ml/a3f1c6b5fde9417a96d642af7068e5dd/report.json','utf8'));
const formatted=context.userFacingAnswer(followup);
for(const h of ['发现了什么','证据是什么','有哪些局限'])assert(formatted.includes(h));
let saved,updated;
const routeContext={URL,location:{href:'http://127.0.0.1:8018/?run='+'a'.repeat(32)+'&developer=1#developer-panel'},
 localStorage:{setItem:(key,id)=>{saved={key,id};}},history:{replaceState:(_a,_b,url)=>{updated=String(url);}}};
vm.createContext(routeContext);
vm.runInContext(source.slice(source.indexOf('function rememberRun('),source.indexOf('async function api(')),routeContext);
routeContext.rememberRun('b'.repeat(32));
assert.equal(saved.id,'b'.repeat(32));assert.equal(new URL(updated).searchParams.get('run'),'b'.repeat(32));
assert.equal(new URL(updated).searchParams.get('developer'),'1');
assert.throws(()=>routeContext.rememberRun('../other'));
console.log(JSON.stringify({live_answers_checked:checked,ordinary_answer_preserved:true,llm_calls:0,passed:true}));
