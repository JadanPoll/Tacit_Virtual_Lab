"use strict";
const fs=require("node:fs"),path=require("node:path"),assert=require("node:assert/strict");
const pages=["index.html","gpu-investigation.html","robotics.html","locks.html","llm.html"];
for(const page of pages){
  const source=fs.readFileSync(page,"utf8");
  assert.match(source,/<main[\s>]/,"Missing main page content: "+page);
  const scripts=[...source.matchAll(/<script(?:\s+[^>]*)?>([\s\S]*?)<\/script>/g)];
  assert(scripts.length>=1,"Missing inline page behavior: "+page);
  for(const [n,m] of scripts.entries()){
    try{new Function(m[1]);}catch(e){throw Error(page+": inline script "+n+" syntax: "+e.message);}
  }
  for(const m of source.matchAll(/(?:href|src)="([^"]+)"/g)){
    const link=m[1];
    if(/^(https?:|#|data:)/.test(link))continue;
    const file=link.split("#")[0].split("?")[0];if(!file)continue;
    const target=path.join(path.dirname(page),file);
    assert(fs.existsSync(target),page+" references missing local asset "+target);
  }
}
console.log("All virtual lab HTML pages parse; local assets exist.");
