(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.WorkflowViz=api;})(typeof window!=='undefined'?window:this,function(){
  const roles=['source','classifier','low','high','reviewer'];
  const edgeNames=['source-classifier','classifier-low','classifier-high','low-high','low-reviewer','high-reviewer'];
  function derive(events,running){
    const visited=new Set(),edges=new Set(),actions={};let current='source',worker=null,activeEdge=null,dispatchEdge=null;
    for(const e of events||[]){
      const name=e.event||'';let role=e.tier||current;
      if(name.startsWith('route_recheck')){role='high';edges.add('classifier-high');activeEdge='classifier-high';}
      else if(name.startsWith('source_')||['planning_started','plan','data_ready'].includes(name))role='source';
      else if(name.startsWith('classification')||name==='route')role='classifier';
      else if(['review_started','final_review','verification','iteration_completed'].includes(name))role='reviewer';
      else if(name==='escalation')role='high';
      if(!roles.includes(role))role=current;
      if(name==='route'){
        edges.add('source-classifier');worker=e.target||e.tier;
        if(['low','high'].includes(worker)){activeEdge='classifier-'+worker;dispatchEdge=activeEdge;edges.add(activeEdge);}
      }
      if(name==='escalation'){activeEdge='low-high';dispatchEdge=activeEdge;edges.add(activeEdge);worker='high';}
      if(name==='worker_started'){
        worker=role;const handoff=dispatchEdge&&dispatchEdge.endsWith(role)?dispatchEdge:'classifier-'+role;
        activeEdge=handoff;if(edgeNames.includes(handoff))edges.add(handoff);
      }
      if(['training_started','experiment','worker_failed'].includes(name)&&['low','high'].includes(role))worker=role;
      if(role==='reviewer'&&worker){activeEdge=worker+'-reviewer';edges.add(activeEdge);}
      current=role;visited.add(role);actions[role]=e;
    }
    return {current,worker,visited:[...visited],edges:[...edges],activeEdges:running&&activeEdge?[activeEdge]:[],actions};
  }
  function demoEvents(data){
    if(data.events?.length)return data.events;
    // Earlier verified reports have steps but no live phase events. Replay only those actual calls.
    const events=[];
    for(const s of data.steps||[]){const common={round:s.round,dataset:s.dataset,step:s.step};
      events.push({...common,event:'data_ready',tier:'source'},{...common,event:'route',tier:'classifier',target:s.tier,reason:s.decision});
      if(s.attempted){events.push({...common,event:'worker_started',tier:'low',model:data.model});if(Number.isFinite(s.low_score))events.push({...common,event:'training_started',tier:'low'});}
      if(s.high_attempted){if(s.attempted)events.push({...common,event:'escalation',tier:'high'});events.push({...common,event:'worker_started',tier:'high'});if(Number.isFinite(s.high_score))events.push({...common,event:'training_started',tier:'high'});}
      if(s.delivered)events.push({...common,event:'verification',tier:'reviewer'},{...common,event:'iteration_completed',tier:'reviewer',ok:s.ok});
    }
    return events;
  }
  const cache={},researchModels={};let selected=null,manual=false,replayTimer=null,replaying=false,originalTitles={};
  const $=id=>typeof document==='undefined'?null:document.getElementById(id);
  function labels(e){
    const names={data_ready:'准备原版数据',source_query:'搜索数据',source_collected:'数据收集完成',route:'按能力记录派发',worker_started:'正在请求模型',training_started:'执行真实训练',experiment:'训练结果已产生',escalation:'低站不足，交高级站',verification:'程序验证并记录结果',iteration_completed:'本步实验已结束',worker_failed:'本次调用或执行失败',stopped:'已达到预算上限',route_recheck_started:'高级模型正在复核派发',route_recheck_resolved:'高级复核完成',review_started:'审核模型正在检查证据',final_review:'审核完成'};
    return (names[e?.event]||'执行阶段：'+(e?.event||'等待'))+(e?.round?' · 第 '+e.round+' 轮':'')+(e?.dataset?' · '+e.dataset:'')+(Number.isFinite(e?.score)?' · F1 '+e.score.toFixed(3):'');
  }
  function ensurePackets(){
    const svg=document.querySelector('.flow-lines');if(!svg||$('flow-packets'))return;
    const ns='http://www.w3.org/2000/svg',group=document.createElementNS(ns,'g');group.id='flow-packets';group.setAttribute('aria-hidden','true');
    for(const key of edgeNames){const path=$('edge-'+key);if(!path)continue;const channel=document.createElementNS(ns,'g');channel.dataset.edge=key;channel.classList.add('flow-packet-channel');
      for(let i=0;i<3;i++){const point=document.createElementNS(ns,'rect');point.setAttribute('x','-3');point.setAttribute('y','-3');point.setAttribute('width','6');point.setAttribute('height','6');point.setAttribute('rx','1');
        const motion=document.createElementNS(ns,'animateMotion');motion.setAttribute('path',path.getAttribute('d'));motion.setAttribute('dur','1.6s');motion.setAttribute('begin',(i*.53)+'s');motion.setAttribute('repeatCount','indefinite');point.append(motion);channel.append(point);}
      group.append(channel);
    }svg.append(group);
  }
  function draw(events,live,isReplay){
    if(!$('node-high'))return;ensurePackets();const data=cache[selected]||{},plan=derive(events,live),isDemo=selected==='demo';
    const records=(data.usage||data.report?.usage||data.partial?.usage)?.records||[];
    const models=isDemo?{source:'原版数据 · 文件工具',classifier:'能力记忆 · 本地规则',low:data.model||'qwen2.5:3b-instruct-q4_K_M',high:data.high_model||data.usage?.records?.find(r=>r.role==='high')?.model||researchModels.high||'已配置高级模型',reviewer:'scikit-learn · 程序验证'}:researchModels;
    for(const role of roles){const node=$('node-'+role);if(!originalTitles[role])originalTitles[role]=node.querySelector('h3').textContent;
      node.querySelector('h3').textContent=isDemo&&role==='reviewer'?'验证 / 记录站':originalTitles[role];
      const calls=records.filter(r=>r.role===role);const hasCalls=calls.length>0;
      if(hasCalls)$('model-'+role).textContent=calls[calls.length-1].model;else if(models[role])$('model-'+role).textContent=models[role];
      node.classList.toggle('visited',plan.visited.includes(role)||hasCalls);node.classList.toggle('active',live&&plan.current===role);node.classList.toggle('expensive-call',(role==='high'&&events.some(e=>e.event==='route_recheck_started'||e.event==='worker_started'&&e.tier==='high'))||calls.some(r=>r.model===researchModels.high));
      node.classList.remove('error','needs-input');node.querySelector('.node-state').textContent=(live&&plan.current===role?(isReplay?'回放此阶段':'正在执行'):plan.visited.includes(role)||hasCalls?'已发生':'等待任务')+' · 模型请求 '+calls.length+' 次';
      $('action-'+role).textContent=plan.actions[role]?(!isDemo&&plan.actions[role].event==='worker_started'?'工作站处理任务':labels(plan.actions[role])):hasCalls?'实际请求已记录，结果可在费用明细核对':'本轮尚未执行';
    }
    document.querySelectorAll('.flow-lines path').forEach(p=>{const key=p.id.slice(5);p.classList.remove('on');p.classList.toggle('visited',plan.edges.includes(key));p.classList.toggle('flowing',plan.activeEdges.includes(key));});
    document.querySelectorAll('.flow-packet-channel').forEach(g=>g.classList.toggle('moving',plan.activeEdges.includes(g.dataset.edge)));
    $('flow-origin').textContent=(isDemo?'千问迭代链路':'研究链路')+(isReplay?' · 实际记录回放（非实时）':' · 真实执行状态');
    $('flow-state').textContent=isReplay?'回放中（非实时）':live?'实时执行中':events.length?'本次执行已结束':'等待任务';
    const last=events[events.length-1];$('flow-summary').textContent=(last?labels(last):'等待实际调用')+(isDemo?'；验证站为本地程序，不表示又调用了审核模型。':'');
    if($('flow-caption'))$('flow-caption').textContent=plan.activeEdges.length?'流动光点表示：'+plan.activeEdges.map(e=>({
      'source-classifier':'数据交给路由站','classifier-low':'派发给千问 / 低成本工作站','classifier-high':'派发至高级模型','low-high':'低站升级，交给高级模型','low-reviewer':'低站结果交给验证 / 审核','high-reviewer':'高级站结果交给验证 / 审核'})[e]).join('、'):'已发生的连接保持点亮；点击“回放实际调用”查看流转。';
    $('flow-live').hidden=!isReplay;$('flow-replay').disabled=!(cache[selected]?.events?.length||isDemo&&data.steps?.length);
    $('flow-replay-high').disabled=!events.some(e=>e.event==='route_recheck_started'||e.event==='worker_started'&&e.tier==='high');
    $('flow-research').classList.toggle('selected',!isDemo);$('flow-demo').classList.toggle('selected',isDemo);
  }
  function paint(){const data=cache[selected]||{events:[],status:'idle'};const events=selected==='demo'?demoEvents(data):data.events||[];draw(events,['queued','running'].includes(data.state||data.status),false);}
  function stop(){clearInterval(replayTimer);replayTimer=null;replaying=false;}
  function update(kind,data){const prior=cache[kind];cache[kind]=data;if(!selected)selected=kind;
    const running=['running','queued'].includes(data.state||data.status);
    if(running&&(!prior||prior.run_id!==data.run_id||!['running','queued'].includes(prior.state||prior.status))){stop();manual=false;selected=kind;}
    if(kind==='research'&&data.run_id&&!manual)selected='research';
    if(kind==='demo'&&!prior&&data.steps?.length&&!cache.research?.run_id&&!manual){selected=kind;}
    if(selected===kind&&!replaying)paint();
  }
  function replay(focusHigh=false){stop();const data=cache[selected];if(!data)return;const events=selected==='demo'?demoEvents(data):data.events||[];if(!events.length)return;
    // Skip repetitive phases, but keep every actual high request and low->high handoff.
    let sequence=events.filter(e=>['data_ready','route','worker_started','training_started','escalation','verification','iteration_completed','route_recheck_started','review_started','final_review'].includes(e.event));
    if(focusHigh){const high=sequence.find(e=>e.event==='worker_started'&&e.tier==='high'||e.event==='route_recheck_started');if(!high)return;
      sequence=high.step?sequence.filter(e=>e.step===high.step):sequence.slice(Math.max(0,sequence.indexOf(high)-2),sequence.indexOf(high)+4);}
    if(!sequence.length)return;replaying=true;let index=1;draw(sequence.slice(0,index),true,true);
    replayTimer=setInterval(()=>{index++;if(index>sequence.length){stop();paint();return;}draw(sequence.slice(0,index),true,true);},650);
  }
  if(typeof document!=='undefined')document.addEventListener('DOMContentLoaded',()=>{
    ensurePackets();$('flow-research').onclick=()=>{stop();manual=true;selected='research';paint();};$('flow-demo').onclick=()=>{stop();manual=true;selected='demo';paint();};$('flow-replay').onclick=()=>replay(false);$('flow-replay-high').onclick=()=>replay(true);$('flow-live').onclick=()=>{stop();paint();};
    document.addEventListener('visibilitychange',()=>document.documentElement.classList.toggle('flow-page-hidden',document.hidden));
  });
  return {derive,demoEvents,research:(state,events)=>update('research',{...state,events}),demo:data=>update('demo',data),configure:models=>{Object.assign(researchModels,models);if(selected==='research')paint();}};
});
