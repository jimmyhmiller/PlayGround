const fs = require("fs");

// the DERIVED source-to-source compiler (main : bytecode -> JS expression string)
eval(fs.readFileSync(__dirname + "/derived_compiler_str.js", "utf8"));
const compile = main;

// reference interpreter (the thing the compiler was derived from)
function interp(prog, x) {
  var s = [], i = 0;
  while (i < prog.length) { var op = prog[i][0];
    if (op==="pushlit") s.push(prog[i][1]);
    else if (op==="pushin") s.push(x);
    else if (op==="add"){var b=s.pop(),a=s.pop();s.push(a+b);}
    else if (op==="mul"){var b=s.pop(),a=s.pop();s.push(a*b);}
    else if (op==="sub"){var b=s.pop(),a=s.pop();s.push(a-b);}
    i++; }
  return s.pop();
}

// --- demo: compile a few programs to native JS and run them ---
const demos = [
  [["pushin",0],["pushlit",3],["mul",0],["pushlit",1],["add",0]],            // in*3+1
  [["pushin",0],["pushin",0],["mul",0]],                                      // in*in
  [["pushin",0],["pushlit",7],["add",0],["pushlit",2],["mul",0]],            // (in+7)*2
];
for (const prog of demos) {
  const text = compile(prog);                          // RUN THE COMPILER -> JS source text
  const fn = new Function("x", "return " + text + ";"); // -> native function, no interpreter
  console.log("compiled:", text);
  console.log("   run on x=4 ->", fn(4), " (interp says", interp(prog,4)+")");
}

// --- correctness sweep: compiled-native == interpreter over random programs ---
function rnd(rng,n){var p=[],d=0;while(p.length<n||d<1){var bin=d>=2,r=Math.floor(rng()*(bin?5:2));
  if(r===0){p.push(["pushlit",Math.floor(rng()*9)-4]);d++;}else if(r===1){p.push(["pushin",0]);d++;}
  else if(r===2){p.push(["add",0]);d--;}else if(r===3){p.push(["mul",0]);d--;}else{p.push(["sub",0]);d--;}
  if(p.length>n*3)break;}while(d>1){p.push(["add",0]);d--;}return p;}
let s=4242; const rng=()=>{s=(s*1103515245+12345)&0x7fffffff;return s/0x7fffffff;};
let bad=0,checks=0;
for(let t=0;t<3000;t++){const prog=rnd(rng,1+Math.floor(rng()*6));
  const fn=new Function("x","return "+compile(prog)+";");
  for(let x=-3;x<=3;x++){checks++;if(fn(x)!==interp(prog,x))bad++;}}
console.log(bad===0 ? ("\nDERIVED COMPILER ≡ INTERPRETER over "+checks+" checks  ✓") : ("\nBAD "+bad));
