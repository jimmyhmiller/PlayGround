// End-to-end real-JS second Futamura projection demo.
//   node docs/p2real/run.js
const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");
const DIR = __dirname, ROOT = path.resolve(DIR, "../..");
const { parseProgram } = require(path.join(DIR, "parse.js"));

// 1. parse the interpreter written in plain JS
const INT = parseProgram(fs.readFileSync(path.join(DIR, "int.js"), "utf8"));
console.log("1. parsed real-JS interpreter int.js  ->  AST with funcs:",
            INT.map(f => f[1]).join(", "));

// 2. assemble  main(prog) = jsmix(INT, "interp", prog)  and specialize with RustPE
const jsmixSrc = fs.readFileSync(path.join(DIR, "jsmix.js"), "utf8");
fs.writeFileSync(path.join(DIR, "p2_jsmix.js"),
  jsmixSrc + `\nfunction main(prog){var INT=${JSON.stringify(INT)};return jsmix(INT,"interp",prog);}\n`);
console.log("2. specializing jsmix w.r.t. the interpreter (RustPE)...");
const bin = path.join(ROOT, "target/release/js-frontend");
const out = execSync(`SPEC_WEIGHT_BUDGET=12000000 ${bin} --js ${path.join(DIR,"p2_jsmix.js")}`,
                     { encoding: "utf8", maxBuffer: 1 << 28 });
const resid = out.split("\n").filter(l => !l.startsWith("---")).slice(
  out.split("\n").findIndex(l => l.includes("residual as JavaScript"))).join("\n");
fs.writeFileSync(path.join(DIR, "derived_compiler.js"), resid);
const blocks = (out.match(/(\d+) block/) || [])[1];
console.log(`   -> derived a COMPILER (${blocks} blocks) saved to docs/p2real/derived_compiler.js`);

// 3. use the derived compiler
eval(resid); const compile = main;
function ev(c,x){const t=c[0];if(t==="rlit")return c[1];if(t==="rin")return x;
  if(t==="radd")return ev(c[1],x)+ev(c[2],x);if(t==="rsub")return ev(c[1],x)-ev(c[2],x);
  if(t==="rmul")return ev(c[1],x)*ev(c[2],x);return 0;}
const prog = [["addin",0],["mullit",3],["addlit",1]];
console.log("\n3. run the derived compiler on a source program:");
console.log("   source  :", JSON.stringify(prog));
console.log("   compiled:", JSON.stringify(compile(prog)), " (== (0+x)*3+1)");
for (const x of [0,1,5]) console.log(`     run@${x} -> ${ev(compile(prog),x)}`);

// 4. verify against the interpreter
function interp(p,x){var a=0,i=0;while(i<p.length){var o=p[i][0];if(o==="addlit")a+=p[i][1];
  else if(o==="mullit")a*=p[i][1];else if(o==="addin")a+=x;else if(o==="subin")a-=x;i++;}return a;}
let s=7,rng=()=>{s=(s*1103515245+12345)&0x7fffffff;return s/0x7fffffff;},bad=0,n=0;
for(let t=0;t<3000;t++){const p=[];for(let k=0;k<1+Math.floor(rng()*7);k++){const r=Math.floor(rng()*4);
  if(r===0)p.push(["addlit",Math.floor(rng()*9)-4]);else if(r===1)p.push(["mullit",Math.floor(rng()*5)]);
  else if(r===2)p.push(["addin",0]);else p.push(["subin",0]);}
  const c=compile(p);for(let x=-3;x<=3;x++){n++;if(ev(c,x)!==interp(p,x))bad++;}}
console.log("\n4. verify  derivedCompiler(prog) ≡ interpreter:",
            bad===0?`ALL ${n} CHECKS PASS  ✓`:`FAILED (${bad})`);
