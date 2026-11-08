"use strict";
function fail(msg){self.postMessage({type:"result",ok:false,message:msg});}
function near(a,b,tol=1e-5){return Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=tol+tol*Math.abs(b);}
function compare(a,b){return Array.isArray(a)&&a.length===b.length&&a.every((x,i)=>near(x,b[i]));}
function softmax(logits){
  const mx=Math.max(...logits),nums=logits.map(z=>Math.exp(z-mx)),sum=nums.reduce((a,b)=>a+b,0);
  return nums.map(z=>z/sum);
}
function referenceAttention(q,keys,values){
 const raw=keys.map(k=>k.reduce((s,v,j)=>s+v*q[j],0)/Math.sqrt(q.length));
 const weights=softmax(raw);
 return {weights,output:values[0].map((_,j)=>values.reduce((s,v,i)=>s+v[j]*weights[i],0))};
}
function nll(logits,target){const m=Math.max(...logits);return m+Math.log(logits.reduce((s,v)=>s+Math.exp(v-m),0))-logits[target];}
function referenceGradient(z,t){
 const h=1e-5;return z.map((_,i)=>{
   const hi=z.slice(),lo=z.slice();hi[i]+=h;lo[i]-=h;return (nll(hi,t)-nll(lo,t))/(2*h);
 });
}
function referenceDistribution(z,temp,k){
 const keep=z.map((value,i)=>({value,i})).sort((a,b)=>b.value-a.value||a.i-b.i).slice(0,k);
 const idx=new Set(keep.map(x=>x.i)),m=Math.max(...keep.map(x=>x.value/temp));
 const values=z.map((v,i)=>idx.has(i)?Math.exp(v/temp-m):0),sum=values.reduce((a,b)=>a+b,0);
 return values.map(v=>v/sum);
}
function rng(seed=842){let x=seed>>>0;return()=>((x^=x<<13,x^=x>>>17,x^=x<<5)>>>0)/4294967296;}
function createTests(task,trace){
 const rand=rng(),cases=[];
 if(task==="attention"){
   const h=trace.head;
   cases.push({args:[h.q,h.keys,h.values],expected:referenceAttention(h.q,h.keys,h.values),actualWeights:h.weights});
   for(let j=0;j<20;j++){
     const n=1+(j%h.keys.length),q=h.q.map(v=>v*(0.6+0.8*rand()));
     const keys=h.keys.slice(0,n).map(row=>row.map(v=>v+(rand()-.5)*0.4));
     const values=h.values.slice(0,n).map(row=>row.map(v=>v+(rand()-.5)*0.5));
     cases.push({args:[q,keys,values],expected:referenceAttention(q,keys,values)});
   }
 }else if(task==="gradient"){
   const logits=trace.logits;
   for(let t=0;t<logits.length;t++)cases.push({args:[logits,t],expected:referenceGradient(logits,t)});
   for(let i=0;i<20;i++){const z=logits.map(x=>x+(rand()-.5)*6),t=Math.floor(rand()*z.length);cases.push({args:[z,t],expected:referenceGradient(z,t)});}
 }else if(task==="distribution"){
   const logits=trace.logits;
   for(const temp of [.5,1,1.5])for(const k of [1,Math.min(2,logits.length),logits.length])cases.push({args:[logits,temp,k],expected:referenceDistribution(logits,temp,k)});
   for(let i=0;i<20;i++){const z=logits.map(x=>x+(rand()-.5)*5),temp=0.5+1.5*rand(),k=1+Math.floor(rand()*z.length);cases.push({args:[z,temp,k],expected:referenceDistribution(z,temp,k)});}
 }else throw Error("Unknown problem.");
 return cases;
}
self.onmessage=({data})=>{
 try{
  const {task,source,trace}=data;
  const solution=Function('"use strict";\n'+source+'\nreturn solve;')();
  if(typeof solution!=="function")throw Error("Define a function named solve.");
  const cases=createTests(task,trace);
  let successes=0;
  for(const c of cases){
    const args=JSON.parse(JSON.stringify(c.args));
    let out;try{out=solution(...args);}catch(e){throw Error("Case "+(successes+1)+": "+e.message);}
    const ok=task==="attention"&&out&&compare(out.weights,c.expected.weights)&&compare(out.output,c.expected.output)
      ||task!=="attention"&&compare(out,c.expected);
    if(!ok)throw Error("Case "+(successes+1)+" failed.\nInput: "+JSON.stringify(c.args).slice(0,450)+"\nExpected: "+JSON.stringify(c.expected).slice(0,450)+"\nReceived: "+JSON.stringify(out).slice(0,450));
    if(c.actualWeights&&!compare(out.weights,c.actualWeights))throw Error("Mismatch with model's actual attention trace.");
    successes++;
  }
  self.postMessage({type:"result",ok:true,message:successes+" cases passed.",passed:successes});
 }catch(e){fail(String(e.message||e));}
};
