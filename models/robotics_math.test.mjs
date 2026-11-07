import { readFileSync } from "node:fs";
import { test } from "node:test";
import assert from "node:assert/strict";
const html=readFileSync(new URL("../robotics.html",import.meta.url),"utf8");
const src=html.match(/<script id="math-core">([\s\S]*?)<\/script>/)?.[1];
assert.ok(src,"Math model not found");
const M=new Function(src+";return RobotMath;")();
test("200 parameterized FK/Jacobian central finite differences",()=>{
for(let i=0;i<200;i++){const d={q1:-155+(i*53)%310,q2:-151+(i*67)%302,l1:0.5+(i%9)*0.11,l2:0.6+(i%8)*0.09,s1:2+i%10,s2:3+i%9,rho:(i%13-6)/10,noise:0.03+i*0.001,axis:i%2?"x":"y"};const p=M.calc(d);for(let j=0;j<2;j++){const key=j?"q2":"q1",dh=1e-6*180/Math.PI,hi=M.calc({...d,[key]:d[key]+dh}),lo=M.calc({...d,[key]:d[key]-dh});for(let k=0;k<2;k++)assert.ok(Math.abs((hi.tip[k]-lo.tip[k])/(2e-6)-p.J[k][j])<1e-6);}}});
test("200 PSD and covariance-reduction scenarios",()=>{for(let i=0;i<200;i++){const p=M.calc({q1:-120+i,q2:(i*17)%180-90,s1:2+i%12,s2:3+i%9,rho:(i%17-8)/10,noise:0.02+0.001*i,axis:i%2?"x":"y"});assert.ok(p.posterior.area<=p.prior.area+1e-12);for(let k=0;k<2;k++)assert.ok(p.U[k][k]>=-1e-12&&p.U[k][k]<=p.P[k][k]+1e-12);assert.ok(p.U[0][0]*p.U[1][1]-p.U[0][1]*p.U[1][0]>=-1e-12);}});
test("Ellipse area scales with joint standard deviation",()=>{for(let i=0;i<60;i++){const d={q1:i*5,q2:15+i*2,s1:6,s2:8,rho:0.3};const a=M.calc(d).prior.area,b=M.calc({...d,s2:16}).prior.area;assert.ok(Math.abs(b/a-2)<1e-6);}});
test("Fully extended arm is singular",()=>{for(const q1 of [-90,0,45,135])assert.ok(Math.abs(M.calc({q1,q2:0}).detJ)<1e-12);});
test("Reject physically invalid input",()=>{for(const v of [{l1:-1},{rho:2},{noise:0},{axis:"z"}])assert.throws(()=>M.calc(v));});
