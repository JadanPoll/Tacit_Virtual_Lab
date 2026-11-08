"use strict";
const assert=require("node:assert/strict");
const {explore}=require("./lock-lab.js");
const AB=["A","B"],BA=["B","A"];
let r=explore({plans:[AB,BA],barrier:true});
assert(!r.safe && r.deadlocks>0 && r.failure,"Original incident must deadlock");
r=explore({plans:[AB,BA],barrier:false});
assert(!r.safe && r.deadlocks>0 && r.terminations>0,"Removing only barrier leaves reachable deadlock");
r=explore({plans:[AB,AB],barrier:true});
assert(!r.safe && r.deadlocks>0,"Fixing only lock ordering leaves barrier deadlock");
for(const order of [AB,BA]){
 r=explore({plans:[order,order],barrier:false});
 assert(r.safe && r.terminations>0 && r.deadlocks===0,"Same global order, no barrier must terminate");
}
assert.throws(()=>explore({plans:[["A","A"],BA]}),/exactly once/);
console.log("Virtual Linux lock model: all regression cases passed.");