"use strict";
const assert=require("node:assert/strict");
const gpu=require("./gpu-virtual-core.js");
for(const seed of [1,12345,0xdeadbeef,4294967295]){
  const base=gpu.run(gpu.baseline,seed),opt=gpu.run(gpu.tiled,seed);
  assert(base.correct && opt.correct,"transpose must be correct for every input");
  assert.equal(base.readSectors,1024);assert.equal(base.writeSectors,128);
  assert.equal(base.totalSectors,1152);
  assert.equal(opt.readSectors,128);assert.equal(opt.writeSectors,128);
  assert.equal(opt.totalSectors,256);
  assert.equal(opt.reads,1024);assert.equal(opt.writes,1024);
}
assert.deepEqual(gpu.sectorSet([7,8]),[0,1]);
assert.throws(()=>gpu.sectorSet([1024]),/Out-of-range/);
assert.equal(gpu.run("function kernel() {}",42).correct,false);
assert.throws(()=>gpu.run("function kernel(){warpLoad([0])}",42),/exactly 32/);

const wrong=gpu.baseline.replace("row*N + lane","lane*N + row"); // corruption: non-transposed output
assert.equal(gpu.run(wrong,98765).correct,false,"A plausible wrong indexing expression must fail correctness");
const duplicate=`function kernel() {
 for(let row=0;row<N;row++){
  const indices=Array.from({length:32},(_,i)=>row*N+i);
  const values=warpLoad(indices);
  warpStore(Array(32).fill(row*N),values);
 }
}`;
assert.equal(gpu.run(duplicate,12345).correct,false,"Duplicate output writes must not count as a correct run");

