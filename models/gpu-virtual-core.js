/* Deterministic warp-memory laboratory. This is a scoped transaction model, not a GPU timing simulator. */
(function (root) {
  "use strict";
  const N=32, LANES=32, MAX_CALLS=256;
  function byteSectorSet(addresses) {
    const sectors=new Set();
    for(const address of addresses) {
      if(!Number.isSafeInteger(address)||address<0||address>Number.MAX_SAFE_INTEGER-3)throw Error("Invalid byte address: "+address);
      const first=Math.floor(address/32),last=Math.floor((address+3)/32);
      for(let s=first;s<=last;s++)sectors.add(s);
    }
    return [...sectors];
  }
  function sectorSet(indices) {
    for (const index of indices) {
      if (!Number.isInteger(index) || index<0 || index>=N*N)throw Error("Out-of-range or noninteger memory index: "+index);
    }
    return byteSectorSet(indices.map(index=>index*4));
  }
  function makeInput(seed) {
    let x=seed>>>0;
    const a=new Float32Array(N*N);
    for(let i=0;i<a.length;i++) {
      x^=x<<13;x^=x>>>17;x^=x<<5;
      a[i]=((x>>>0)%200001-100000)/2048;
    }
    return a;
  }
  function expected(input) {
    const out=new Float32Array(input.length);
    for(let r=0;r<N;r++)for(let c=0;c<N;c++)out[r*N+c]=input[c*N+r];
    return out;
  }
  function run(source, seed=12345) {
    if(typeof source!=="string" || source.length>14000)throw Error("Kernel must be source text of at most 14 KB.");
    const input=makeInput(seed), output=new Float32Array(input.length);
    const written=new Uint8Array(output.length),trace=[];
    let reads=0,writes=0,readSectors=0,writeSectors=0,calls=0;
    function check(indices,values) {
      if(!Array.isArray(indices) || indices.length!==LANES)throw Error("Every warp instruction must supply exactly 32 lane addresses.");
      if(++calls>MAX_CALLS)throw Error("Exceeded virtual instruction limit.");
      const sectors=sectorSet(indices);
      if(values && (!Array.isArray(values)||values.length!==LANES||values.some(v=>!Number.isFinite(v))))throw Error("Stores need 32 finite values.");
      return sectors;
    }
    function warpLoad(indices) {
      const sectors=check(indices);
      reads+=LANES;readSectors+=sectors.length;
      if(trace.length<12)trace.push({op:"load",addresses:indices.slice(),sectors:sectors.slice()});
      return indices.map(i=>input[i]);
    }
    function warpStore(indices,values) {
      const sectors=check(indices,values);
      writes+=LANES;writeSectors+=sectors.length;
      if(trace.length<12)trace.push({op:"store",addresses:indices.slice(),sectors:sectors.slice()});
      for(let t=0;t<LANES;t++){output[indices[t]]=values[t];written[indices[t]]++;}
    }
    const execute=Function("warpLoad","warpStore","N",'"use strict";\n'+source+'\nreturn kernel;')(warpLoad,warpStore,N);
    if(typeof execute!=="function")throw Error("Define function kernel().");
    execute();
    const gold=expected(input);
    let mismatches=0,firstMismatch=-1;
    for(let i=0;i<gold.length;i++)if(output[i]!==gold[i]||written[i]!==1){mismatches++;if(firstMismatch<0)firstMismatch=i;}
    return {correct:mismatches===0,mismatches,firstMismatch,readSectors,writeSectors,totalSectors:readSectors+writeSectors,
      reads,writes,warpInstructions:calls,trace};
  }
  const baseline=String.raw`function kernel() {
  // One warp writes each output row. The read is strided.
  for (let row = 0; row < N; row++) {
    const source = Array.from({length:32}, (_,lane) => lane*N + row);
    const destination = Array.from({length:32}, (_,lane) => row*N + lane);
    const values = warpLoad(source);
    warpStore(destination, values);
  }
}`;
  const tiled=String.raw`function kernel() {
  // Virtual shared tile: local JavaScript array, no global memory transactions.
  const tile = [];
  for (let row=0; row<N; row++) {
    tile[row] = warpLoad(Array.from({length:32}, (_,lane)=>row*N+lane));
  }
  for (let row=0; row<N; row++) {
    const values = Array.from({length:32}, (_,lane)=>tile[lane][row]);
    warpStore(Array.from({length:32}, (_,lane)=>row*N+lane), values);
  }
}`;
  const API={N,sectorSet,byteSectorSet,makeInput,expected,run,baseline,tiled};
  if(typeof module!=="undefined"&&module.exports)module.exports=API;
  root.GPUVirtual=API;
})(typeof self!=="undefined"?self:globalThis);
