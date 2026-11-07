"use strict";
/* Small exact algorithms shared by the browser problems. */
const LLMCore=(()=>{
 const input=["A","B","C","D"];
 const E={A:[1,0],B:[0,1],C:[1,1],D:[2,1]};
 const vocab=["NEXT","STOP","MORE","DONE"];
 const outW=[[2,0],[0,2],[-1,-0.7],[0.25,0.8]];
 const dot=(a,b)=>a.reduce((v,x,i)=>v+x*b[i],0);
 const softmax=x=>{const finite=x.filter(Number.isFinite);if(!finite.length)throw Error("No valid logits");const m=Math.max(...finite),e=x.map(v=>Number.isFinite(v)?Math.exp(v-m):0),z=e.reduce((a,b)=>a+b,0);return e.map(v=>v/z);};
 const entropy=p=>-p.reduce((s,x)=>s+(x>0?x*Math.log(x):0),0);
 const value=v=>[v[0]-v[1],v[0]+v[1]];
 function attention(prefix=input,position=prefix.length-1,temp=1,causal=true){
  if(!Array.isArray(prefix)||!prefix.length||!prefix.every(t=>E[t])||!Number.isInteger(position)||position<0||position>=prefix.length||temp<=0)throw Error("Invalid attention inputs");
  const q=E[prefix[position]],keys=prefix.map(t=>E[t]),values=prefix.map(t=>value(E[t]));
  const scores=keys.map((k,i)=>causal&&i>position?-Infinity:dot(q,k)/Math.sqrt(2)/temp);
  const weights=softmax(scores),mixed=values[0].map((_,j)=>weights.reduce((s,w,i)=>s+w*values[i][j],0));
  const logits=outW.map(w=>dot(w,mixed));
  return{prefix,position,scores,weights,mixed,logits,prob:softmax(logits)};
 }
 function cached(prefix=input,temp=1){
  const keys=[],values=[],output=[];
  for(let i=0;i<prefix.length;i++){const t=prefix[i];if(!E[t])throw Error("Unknown token");keys.push(E[t]);values.push(value(E[t]));const scores=keys.map(k=>dot(E[t],k)/Math.sqrt(2)/temp),weights=softmax(scores),mixed=values[0].map((_,j)=>weights.reduce((s,w,k)=>s+w*values[k][j],0));output.push({weights,mixed,logits:outW.map(w=>dot(w,mixed))});}
  return output;
 }
 const rules=[["l","o"],["lo","w"],["e","r"],["low","er"]];
 function bpe(word="lower",count=3){
  if(!Number.isInteger(count)||count<0||count>rules.length)throw Error("Invalid merge count");
  const parts=[...word.toLowerCase()],history=[parts.join(" | ")];
  for(;;){let best=null;for(let i=0;i<parts.length-1;i++){const rank=rules.findIndex(([a,b],j)=>j<count&&a===parts[i]&&b===parts[i+1]);if(rank>=0&&(!best||rank<best.rank))best={i,rank};}if(!best)break;parts.splice(best.i,2,parts[best.i]+parts[best.i+1]);history.push(parts.join(" | "));}
  return{parts,history};
 }
 function context(limit=4){const all=["BOS","the","quick","brown","fox","jumps"];if(!Number.isInteger(limit)||limit<1||limit>all.length)throw Error("Invalid context limit");return{all,kept:all.slice(-limit)};}
 function sgd(w=0.5,x=1.5,y=1,lr=0.3){
  const sig=z=>1/(1+Math.exp(-z)),loss=z=>Math.log(1+Math.exp(-Math.abs(z)))-y*z+Math.max(z,0);
  const p=sig(w*x),gradient=(p-y)*x,nextW=w-lr*gradient;
  return{p,gradient,nextW,before:loss(w*x),after:loss(nextW*x)};
 }
 function decode(temp=1,k=3,u=0.5){
  if(!(temp>0)||!Number.isInteger(k)||k<1||k>vocab.length||u<0||u>=1)throw Error("Invalid decoding inputs");
  const logits=attention(["A","B","C"]).logits;
  const kept=logits.map((v,i)=>({v,i})).sort((a,b)=>b.v-a.v||a.i-b.i).slice(0,k).map(x=>x.i);
  const p=softmax(logits.map((v,i)=>kept.includes(i)?v/temp:-Infinity));
  let sum=0,chosen=vocab.at(-1);
  for(let i=0;i<p.length;i++){sum+=p[i];if(u<sum){chosen=vocab[i];break;}}
  return{logits,p,chosen,entropy:entropy(p),supported:p.filter(v=>v>0).length};
 }
 function work(n=5){if(!Number.isInteger(n)||n<1||n>1000)throw Error("Invalid length");let rerun=0,cache=0;for(let t=1;t<=n;t++){rerun+=t*(t+1)/2;cache+=t;}return{rerun,cache};}
 function kvBytes(batch=2,layers=4,heads=2,dim=8,seq=16,bits=16){return batch*layers*heads*dim*seq*2*bits/8;}
 function quant(bits=4){
  const source=[-0.95,-0.55,-0.15,0.2,0.48,0.7,1.15,1.5],scale=1.5/(2**(bits-1)-1);
  const q=source.map(v=>Math.max(-(2**(bits-1)),Math.min(2**(bits-1)-1,Math.round(v/scale)))*scale),mse=q.reduce((a,v,i)=>a+(v-source[i])**2,0)/q.length;
  return{source,q,mse,bytes:source.length*bits/8,scale};
 }
 function retrieve(x=1,y=0.4){
  const query=[x,y],docs={alpha:[1,0],beta:[1,1],gamma:[0,1]},len=v=>Math.hypot(...v);
  if(len(query)<1e-12)throw Error("Zero query");
  const ranked=Object.entries(docs).map(([name,v])=>({name,score:dot(query,v)/(len(query)*len(v))})).sort((a,b)=>b.score-a.score);
  return{ranked,winner:ranked[0].name};
 }
 return{input,vocab,attention,cached,bpe,context,sgd,decode,work,kvBytes,quant,retrieve,softmax,entropy};
})();
