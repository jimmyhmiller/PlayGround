// test/p1_mix_rfg.js — RESIDUAL FUNCTION GENERATION (the online-PE core).
// The interpreter's recursion is driven by the DYNAMIC input x: object node ["sumto"]
// means "sum 1..x", evaluated by a recursive helper sumto(n) with n derived from x.
// Since the trip count depends on dynamic x, mix CANNOT unroll — it must emit a
// residual recursive function. Without memoization mix inlines forever (stack overflow).
//
// Expected residual (shape): a recursive fn, e.g.
//   function f(n){ return ((n <= 0) ? 0 : (n + f((n - 1)))); }   ; entry: f(x)
//
// This test currently FAILS (overflow / unsupported) — it's the red target for RFG.

const { parseProgram } = require("../parse.js");
const mix = require("../mix.subset.js");
const { compileResidual } = require("../backend.js");

const INT_SRC = `
function interp(e, x) {
  if (e[0] === "num") { return e[1]; }
  if (e[0] === "x") { return x; }
  if (e[0] === "add") { return interp(e[1], x) + interp(e[2], x); }
  if (e[0] === "sumto") { return sumto(x); }
  return 0;
}
function sumto(n) {
  if (n <= 0) { return 0; }
  return n + sumto(n - 1);
}`;

function refInterp(e, x) {
  if (e[0] === "num") return e[1];
  if (e[0] === "x") return x;
  if (e[0] === "add") return refInterp(e[1], x) + refInterp(e[2], x);
  if (e[0] === "sumto") { let s = 0; for (let n = x; n > 0; n--) s += n; return s; }
  return 0;
}

const prog = parseProgram(INT_SRC);
let fail = 0, total = 0;
try {
  const resid = mix.specializeProg(prog, "interp", [["s", ["sumto"]], ["d", ["var", "x"]]]);
  const { entry, defs } = compileResidual(resid);
  const js = defs + `function __entry(x) { return ${entry}; }\n`;
  console.log(js);
  const fn = new Function(js + "; return __entry;")();
  for (const x of [0, 1, 2, 3, 5, 10, 50]) {
    const a = refInterp(["sumto"], x), b = fn(x);
    total++;
    if (a !== b) { fail++; console.log(`  DIVERGE x=${x}: int=${a} residual=${b}`); }
  }
} catch (e) {
  fail++; console.log("RFG not ready: " + e.message);
}
console.log(`\np1_mix_rfg: checks=${total} fail=${fail}`);
process.exit(fail ? 1 : 0);
