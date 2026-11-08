/* Explicit-state interleaving model for the lock-order incident in gdb/linux_incidents.c.
   Atomic lock/unlock actions; no OS scheduler, debugger or pthread execution is simulated. */
(function(root){
  "use strict";
  const choices={"A then B":["A","B"],"B then A":["B","A"]};
  function explore(options={}){
    const plans=(options.plans||[["A","B"],["B","A"]]).map(a=>a.slice());
    const barrier=options.barrier!==false;
    if(plans.length!==2||plans.some(p=>p.length!==2||p[0]===p[1]||!["A","B"].includes(p[0])||!["A","B"].includes(p[1])))
      throw Error("Each worker must acquire A and B exactly once.");
    const initial={pc:[0,0],locks:{A:-1,B:-1}}, visited=new Set();
    const pending=[{state:initial,trace:[]}];
    let deadlocks=0,terminations=0,firstFailure=null,firstSuccess=null;
    function clone(s){return {pc:s.pc.slice(),locks:{...s.locks}};}
    function next(s){
      if(s.pc[0]===5&&s.pc[1]===5)return [];
      if(barrier&&s.pc[0]===1&&s.pc[1]===1){
        const t=clone(s);t.pc=[2,2];return [{state:t,event:"Both workers pass the synchronization barrier"}];
      }
      const choices=[];
      for(let w=0;w<2;w++){
        const step=s.pc[w],t=clone(s),prefix="Worker "+(w+1)+": ";
        if(step===0||step===2){
          const lock=plans[w][step===0?0:1];
          if(t.locks[lock]!==-1)continue;
          t.locks[lock]=w;t.pc[w]++;
          choices.push({state:t,event:prefix+"acquires "+lock});
        }else if(step===1){
          if(barrier)continue;
          t.pc[w]++;choices.push({state:t,event:prefix+"continues to second lock"});
        }else if(step===3||step===4){
          const lock=plans[w][step===3?1:0];
          if(t.locks[lock]!==w)throw Error("Invalid lock ownership invariant.");
          t.locks[lock]=-1;t.pc[w]++;
          choices.push({state:t,event:prefix+"releases "+lock});
        }
      }
      return choices;
    }
    while(pending.length){
      const {state,trace}=pending.pop(),key=JSON.stringify(state);
      if(visited.has(key))continue;visited.add(key);
      if(visited.size>2000)throw Error("Exploration state bound exceeded.");
      if(state.pc.every(pc=>pc===5)){terminations++;if(!firstSuccess)firstSuccess=trace;continue;}
      const moves=next(state);
      if(!moves.length){deadlocks++;if(!firstFailure)firstFailure={trace,state};continue;}
      for(const m of moves)pending.push({state:m.state,trace:[...trace,m.event]});
    }
    return {safe:deadlocks===0&&terminations>0,deadlocks,terminations,states:visited.size,
      failure:firstFailure,success:firstSuccess,plans,barrier};
  }
  const API={explore,choices};
  if(typeof module!=="undefined"&&module.exports)module.exports=API;
  root.LockLab=API;
})(typeof self!=="undefined"?self:globalThis);