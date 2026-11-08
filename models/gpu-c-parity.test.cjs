"use strict";
// Cross-check the exact primitive counted in the browser virtual kernel model
// against the native A40 C model's memory probe, including unaligned accesses.
const {execFileSync}=require("node:child_process");
const assert=require("node:assert/strict");
const path=require("node:path");
const gpu=require("./gpu-virtual-core.js");
const binary=path.resolve(process.argv[2]||"models/a40_model");
const cases=[[0,4],[4,4],[31,0],[31,64],[96,4],[0,128],[33,1],[127,4],[0,0]];
let x=0xa4012345;
for(let i=0;i<80;i++){
  x^=x<<13;x^=x>>>17;x^=x<<5;const base=(x>>>0)%4096;
  x^=x<<13;x^=x>>>17;x^=x<<5;const stride=(x>>>0)%129;
  cases.push([base,stride]);
}
for(const [base,stride] of cases){
  const addresses=Array.from({length:32},(_,lane)=>base+lane*stride);
  const js=gpu.byteSectorSet(addresses).length;
  const out=execFileSync(binary,["--probe","--base",String(base),"--stride",String(stride)],{encoding:"utf8"});
  const m=out.match(/sectors=(\d+);/);
  assert(m,"A40 C probe did not report sectors");
  assert.equal(Number(m[1]),js,"JS/C sector mismatch for base="+base+" stride="+stride);
}
assert.equal(gpu.byteSectorSet(Array(32).fill(31)).length,2);
assert.equal(gpu.byteSectorSet(Array.from({length:32},(_,lane)=>31+lane*64)).length,64);
console.log("GPU virtual/C parity:",cases.length,"warp access patterns passed.");
