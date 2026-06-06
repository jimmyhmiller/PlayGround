const fs = require("fs");
const { specializeGeneral } = require("../jspe.js");
const { parseProgram } = require("../parse.js");
const backend = require("../backend.js");

const INT_SRC = `
function interp(e, x) {
  if (e[0] === "num") { return e[1]; }
  if (e[0] === "x") { return x; }
  if (e[0] === "add") { return interp(e[1], x) + interp(e[2], x); }
  return 0;
}`;
const INT = parseProgram(INT_SRC);
let mixSrc = fs.readFileSync(require("path").join(__dirname, "..", "mix.subset.js"), "utf8");
mixSrc = mixSrc.slice(mixSrc.indexOf("function envNil"), mixSrc.indexOf("if (typeof module"));

// P2: build the COMPILER = jspe(mix, int)
const argVals = [ ["d", { __dyn: { tag: "Var", id: 0 } }], ["d", ["var", "x"]] ];
const compilerJs = specializeGeneral(mixSrc, "specializeProg", [{s: INT},{s:"interp"},{s: argVals}], "v0");
const compile = new Function(compilerJs + "; return main;")();  // compile(src) -> mix's ["prog",...]

// reference interpreter (oracle)
function refInterp(e, x){ if(e[0]==="num")return e[1]; if(e[0]==="x")return x; if(e[0]==="add")return refInterp(e[1],x)+refInterp(e[2],x); return 0; }

const SRCS = [
  ["add", ["x"], ["num", 5]],
  ["add", ["add", ["x"], ["x"]], ["num", 3]],
  ["x"],
  ["num", 42],
  ["add", ["num", 1], ["add", ["x"], ["x"]]],
];
let fail = 0, total = 0;
for (const src of SRCS) {
  const prog = compile(src);                    // run the P2 compiler on a concrete source
  const { entry, defs } = backend.compileResidual(prog);  // render the produced target
  const targetJs = defs + "function __t(x){ return " + entry + "; }";
  const target = new Function(targetJs + "; return __t;")();
  let ok = true;
  for (const x of [-4, 0, 1, 7, 100]) {
    const a = refInterp(src, x), b = target(x); total++;
    if (a !== b) { ok = false; fail++; console.log("  DIVERGE src="+JSON.stringify(src)+" x="+x+" int="+a+" target="+b); }
  }
  console.log((ok?"ok  ":"FAIL")+" "+JSON.stringify(src)+"  -> target: "+entry);
}
console.log("\nP2 verify: checks="+total+" fail="+fail);
process.exit(fail?1:0);
