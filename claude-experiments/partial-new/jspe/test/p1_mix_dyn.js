// test/p1_mix_dyn.js — mix must residualize DYNAMIC control flow.
// The interpreter branches on the dynamic input x (an object-language `ifpos` node:
// "if x>0 then A else B"). With x dynamic, mix can't pick a branch — it must emit a
// residual `cond`. Verified differentially: int(src,x) === residual(x) over many x.

const { parseProgram } = require("../parse.js");
const mix = require("../mix.subset.js");
const { toSingleExpr } = require("../backend.js");

const INT_SRC = `
function interp(e, x) {
  if (e[0] === "num") { return e[1]; }
  if (e[0] === "x") { return x; }
  if (e[0] === "add") { return interp(e[1], x) + interp(e[2], x); }
  if (e[0] === "mul") { return interp(e[1], x) * interp(e[2], x); }
  if (e[0] === "sub") { return interp(e[1], x) - interp(e[2], x); }
  if (e[0] === "ifpos") { if (interp(e[1], x) > 0) { return interp(e[2], x); } return interp(e[3], x); }
  return 0;
}`;

function render(ast) {
  const t = ast[0];
  if (t === "lit") return typeof ast[1] === "string" ? JSON.stringify(ast[1]) : String(ast[1]);
  if (t === "var") return ast[1];
  if (t === "bin") return "(" + render(ast[2]) + " " + ast[1] + " " + render(ast[3]) + ")";
  if (t === "cond") return "(" + render(ast[1]) + " ? " + render(ast[2]) + " : " + render(ast[3]) + ")";
  if (t === "idx") return render(ast[1]) + "[" + render(ast[2]) + "]";
  if (t === "dot") return render(ast[1]) + "." + ast[2];
  if (t === "arr") return "[" + ast[1].map(render).join(", ") + "]";
  throw new Error("render: unhandled " + t + " in " + JSON.stringify(ast));
}
function refInterp(e, x) {
  if (e[0] === "num") return e[1];
  if (e[0] === "x") return x;
  if (e[0] === "add") return refInterp(e[1], x) + refInterp(e[2], x);
  if (e[0] === "mul") return refInterp(e[1], x) * refInterp(e[2], x);
  if (e[0] === "sub") return refInterp(e[1], x) - refInterp(e[2], x);
  if (e[0] === "ifpos") { if (refInterp(e[1], x) > 0) return refInterp(e[2], x); return refInterp(e[3], x); }
  return 0;
}

const prog = parseProgram(INT_SRC);
const SRCS = {
  // if x>0 then x*2 else 0-x   (== abs(x)*sign-ish)
  "ifpos x: x*2 else -x": ["ifpos", ["x"], ["mul", ["x"], ["num", 2]], ["sub", ["num", 0], ["x"]]],
  // nested: if (x-3)>0 then x else 3
  "ifpos x-3: x else 3": ["ifpos", ["sub", ["x"], ["num", 3]], ["x"], ["num", 3]],
};

let fail = 0, total = 0;
for (const [label, srcExpr] of Object.entries(SRCS)) {
  const result = mix.specializeProg(prog, "interp", [["s", srcExpr], ["d", ["var", "x"]]]);
  const js = toSingleExpr(result);
  const fn = new Function("x", "return " + js + ";");
  let ok = true;
  for (const x of [-9, -3, -1, 0, 1, 2, 4, 8]) {
    const a = refInterp(srcExpr, x), b = fn(x);
    total++;
    if (a !== b) { ok = false; fail++; console.log(`  DIVERGE ${label} x=${x}: int=${a} residual=${b}`); }
  }
  console.log((ok ? "ok  " : "FAIL") + "  " + label.padEnd(22) + " -> " + js);
}
console.log(`\np1_mix_dyn: checks=${total} fail=${fail}`);
process.exit(fail ? 1 : 0);
