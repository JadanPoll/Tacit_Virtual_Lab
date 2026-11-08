"use strict";
importScripts("./gpu-virtual-core.js");
self.onmessage = ({data}) => {
  try {
    const seeds=[12345,0xdeadbeef,17,987654321];
    const results=seeds.map(s=>GPUVirtual.run(data.source,s));
    const first=results[0];
    const valid=results.every(r=>r.correct);
    self.postMessage({type:"result",...first,valid,tests:results.map((r,i)=>({seed:seeds[i],correct:r.correct,mismatches:r.mismatches,totalSectors:r.totalSectors})),
      achieved:valid&&results.every(r=>r.totalSectors<=320 && r.reads===1024 && r.writes===1024 && r.warpInstructions<=64)});
  }catch(e){self.postMessage({type:"error",message:String(e.message||e)});}
};