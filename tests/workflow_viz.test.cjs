const assert=require('node:assert/strict');
const {derive,demoEvents}=require('../static/workflow-viz.js');
let p=derive([{event:'route',tier:'low'},{event:'worker_started',tier:'low'},{event:'escalation',tier:'high'},{event:'worker_started',tier:'high'}],true);
assert.equal(p.current,'high');assert.deepEqual(p.activeEdges,['low-high']);
p=derive([{event:'route_recheck_started'}],true);
assert.equal(p.current,'high');assert.deepEqual(p.activeEdges,['classifier-high']);
p=derive([{event:'route',tier:'low'},{event:'escalation',tier:'high'},{event:'worker_started',tier:'high'},
 {event:'route',tier:'classifier',target:'high'},{event:'worker_started',tier:'high'}],true);
assert.deepEqual(p.activeEdges,['classifier-high'],'new direct dispatch must not animate an old escalation edge');
p=derive([{event:'route',tier:'high'},{event:'worker_started',tier:'high'},{event:'verification',tier:'reviewer'}],true);
assert.deepEqual(p.activeEdges,['high-reviewer']);
assert.deepEqual(derive([{event:'worker_started',tier:'high'}],false).activeEdges,[],'finished runs must not pretend to be live');
let e=demoEvents({steps:[{round:1,step:1,tier:'high',attempted:false,high_attempted:false,delivered:false}]});
assert.equal(e.filter(x=>x.event==='worker_started').length,0,'a delegated but unexecuted task is not a model invocation');
e=demoEvents({model:'qwen2.5:3b-instruct-q4_K_M',steps:[{round:1,step:1,tier:'low',attempted:true,high_attempted:true,low_score:.4,high_score:.5,delivered:true}]});
assert.equal(e.filter(x=>x.event==='worker_started'&&x.tier==='high').length,1);
assert.equal(e.at(-1).tier,'reviewer');
console.log('Workflow visualization: seven routing/animation invariants passed.');
