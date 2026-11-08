import { MicroGPT } from "./vendor/microgpt.js";

const preset = "ada\nava\nana\nida\nala\nava\nana\nada";
let model=null, last=null;
function snapshot(result) {
  const i=Math.min(2,result.vizData.length-1);
  const v=result.vizData[i];
  return {
    step:result.step, loss:result.loss, doc:result.doc, tokens:result.tokens,
    position:i, head:{...v.headVectors[0][0],weights:v.attentionWeights[0][0]},
    logits:v.logits, vocab:[...model.uchars,"<BOS>"],
    parameters:model.params.length, history:model.getLossHistory().slice(-30)
  };
}
function start(text=preset,seed=42) {
  const docs=text.split("\n").map(v=>v.trim()).filter(Boolean);
  if(docs.length<3||docs.some(v=>v.length>7)||docs.join("").length>300)throw Error("Use 3–50 short training lines (up to seven characters each).");
  model=new MicroGPT({nEmbd:8,nHead:2,nLayer:1,blockSize:8,learningRate:.025,numSteps:120,seed});
  model.loadData(docs.join("\n"));model.initParams();
  last=model.trainStep();
  self.postMessage({type:"state",...snapshot(last),sample:null,documents:docs.length});
}
self.onmessage=e=>{
  try {
    const data=e.data||{};
    if(data.type==="reset")start(data.dataset||preset,data.seed??42);
    else if(data.type==="train"){
      if(!model)throw Error("Initialize the model first.");
      const n=data.steps===10?10:1;
      for(let i=0;i<n;i++)last=model.trainStep();
      self.postMessage({type:"state",...snapshot(last),sample:null});
    } else if(data.type==="generate"){
      if(!model)throw Error("Initialize the model first.");
      const sample=model.generate(data.temperature||0.8);
      self.postMessage({type:"generated",sample:sample.text,probabilities:sample.vizData[0]?.probs});
    }
  }catch(err){self.postMessage({type:"error",message:String(err.message||err)});}
};
