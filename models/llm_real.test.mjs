import { readFileSync } from "node:fs";
import { test } from "node:test";
import assert from "node:assert/strict";
import vm from "node:vm";
const modelSrc=readFileSync(new URL("./vendor/microgpt.js",import.meta.url),"utf8");
const mod=await import("data:text/javascript;base64,"+Buffer.from(modelSrc).toString("base64"));
const {MicroGPT,Value,softmax}=mod;
const corpus="ada\nava\nana\nida\nala\nava\nana\nada";
function create(seed=42){
 const m=new MicroGPT({nEmbd:8,nHead:2,nLayer:1,blockSize:8,learningRate:.025,numSteps:120,seed});
 m.loadData(corpus);m.initParams();return m;
}
const near=(a,b,tol=1e-5)=>Math.abs(a-b)<=tol+tol*Math.abs(b);
const refAttention=(q,keys,values)=>{
 const s=keys.map(k=>q.reduce((a,v,j)=>a+v*k[j],0)/Math.sqrt(q.length)),m=Math.max(...s),e=s.map(x=>Math.exp(x-m)),sum=e.reduce((a,b)=>a+b,0),w=e.map(x=>x/sum);
 return {weights:w,output:values[0].map((_,j)=>values.reduce((a,v,i)=>a+v[j]*w[i],0))};
};
const gradeSrc=readFileSync(new URL("./llm-grade-worker.js",import.meta.url),"utf8");
function grade(task,source,trace){
 const posted=[],context={self:{postMessage:x=>posted.push(x)},Function};
 vm.runInNewContext(gradeSrc,context);
 context.self.onmessage({data:{task,source,trace}});
 return posted.at(-1);
}
const solutions={
 attention:"function solve(q,keys,values){const s=keys.map(k=>q.reduce((a,v,j)=>a+v*k[j],0)/Math.sqrt(q.length)),m=Math.max(...s),e=s.map(x=>Math.exp(x-m)),z=e.reduce((a,b)=>a+b,0),weights=e.map(x=>x/z);return {weights,output:values[0].map((_,j)=>values.reduce((a,v,i)=>a+v[j]*weights[i],0))};}",
 gradient:"function solve(logits,target){const m=Math.max(...logits),ex=logits.map(x=>Math.exp(x-m)),z=ex.reduce((a,b)=>a+b,0);return ex.map((v,i)=>v/z-(i===target?1:0));}",
 distribution:"function solve(logits,temp,k){const keep=logits.map((z,i)=>({z,i})).sort((a,b)=>b.z-a.z||a.i-b.i).slice(0,k),selected=new Set(keep.map(x=>x.i)),m=Math.max(...keep.map(x=>x.z/temp)),p=logits.map((z,i)=>selected.has(i)?Math.exp(z/temp-m):0),sum=p.reduce((a,b)=>a+b,0);return p.map(x=>x/sum);}"
};
function traceFrom(model){
 const r=model.trainStep(),v=r.vizData[Math.min(2,r.vizData.length-1)];
 return {head:{...v.headVectors[0][0],weights:v.attentionWeights[0][0]},logits:v.logits};
}
test("Real transformer trains, generates and exposes finite activations",()=>{
 const m=create(),first=m.trainStep();
 assert.ok(m.params.length>800);
 for(let i=0;i<15;i++){const r=m.trainStep();assert.ok(Number.isFinite(r.loss));}
 const g=m.generate(.7);assert.ok(Array.isArray(g.tokens));assert.ok(Array.isArray(g.vizData));assert.ok(g.vizData.length>0);
 for(const v of g.vizData){assert.ok(v.logits.every(Number.isFinite));assert.ok(v.attentionWeights[0].every(head=>near(head.reduce((a,b)=>a+b,0),1)));}
});
test("Deterministic training and examples are reproducible from the seed",()=>{
 const a=create(),b=create();for(let i=0;i<6;i++){const x=a.trainStep(),y=b.trainStep();assert.equal(x.loss,y.loss);assert.equal(x.doc,y.doc);}
});
test("Attention trace reproduces model's actual multihead softmax calculations",()=>{
 const m=create();
 for(let j=0;j<5;j++){const r=m.trainStep();
  for(const pos of r.vizData)for(let l=0;l<pos.headVectors.length;l++)for(let h=0;h<pos.headVectors[l].length;h++){
   const v=pos.headVectors[l][h],exp=refAttention(v.q,v.keys,v.values);
   assert.equal(exp.weights.length,pos.attentionWeights[l][h].length);
   exp.weights.forEach((x,i)=>assert.ok(near(x,pos.attentionWeights[l][h][i],1e-10)));
  }
 }
});
test("Cross-entropy gradient agrees with model's scalar automatic differentiation and finite differences",()=>{
 const m=create(),r=m.trainStep(),logits=r.vizData[1].logits,target=2;
 const x=logits.map(v=>new Value(v)),loss=softmax(x)[target].log().neg();loss.backward();
 const nll=z=>{const m=Math.max(...z);return m+Math.log(z.reduce((s,v)=>s+Math.exp(v-m),0))-z[target];};
 const h=1e-5;for(let i=0;i<x.length;i++){const a=logits.slice(),b=logits.slice();a[i]+=h;b[i]-=h;assert.ok(near(x[i].grad,(nll(a)-nll(b))/(2*h))); }
});
test("All three candidate functions pass multiple actual model traces and parameterized variants",()=>{
 for(const seed of [42,139,2026]){
  const trace=traceFrom(create(seed));
  for(const [task,source]of Object.entries(solutions)){
   const r=grade(task,source,trace);assert.ok(r?.ok,task+" "+r?.message);assert.ok(r.passed>=20);
  }
 }
});
test("Faulty candidates are rejected by each problem grader",()=>{
 const tr=traceFrom(create());
 for(const task of Object.keys(solutions)){
  const r=grade(task,"function solve(){return []}",tr);assert.equal(r?.ok,false,task);
 }
});
test("Worker can reset, train, and sample from the full model",()=>{
 const engine=readFileSync(new URL("./llm-engine-worker.js",import.meta.url),"utf8").replace(/^import [^\n]+\n/,"");
 const outgoing=[],self={postMessage:x=>outgoing.push(x)};
 vm.runInNewContext("(function(MicroGPT,self){"+engine+"\nreturn self.onmessage;})",{}) (MicroGPT,self);
 self.onmessage({data:{type:"reset",dataset:corpus}});
 assert.equal(outgoing.at(-1).type,"state");assert.equal(outgoing.at(-1).step,1);
 self.onmessage({data:{type:"train",steps:10}});
 assert.equal(outgoing.at(-1).step,11);
 self.onmessage({data:{type:"generate"}});
 assert.equal(outgoing.at(-1).type,"generated");
});
