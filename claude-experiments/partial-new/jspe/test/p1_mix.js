// test/p1_mix.js — Futamura PROJECTION 1 using mix.subset.js (the subset-written PE),
// DRIVEN BY NODE (everything static except the interpreted program's input).
//
//   target = mix(int, src)
//
// `int` is an arithmetic-expression interpreter (in the subset). `src` is a static
// expression AST. mix specializes int on src, threading the dynamic input `x`, and
// must produce a residual that equals `src` compiled. We verify it differentially:
// running int(src, x) directly === running the rendered residual at x, for many x.

const { parseProgram } = require("../parse.js");
const mix = require("../mix.subset.js");

// ---- the interpreter, written in the subset (this is `int`) ----
const INT_SRC = `
function interp(e, x) {
  if (e[0] === "num") { return e[1]; }
  if (e[0] === "x") { return x; }
  if (e[0] === "add") { return interp(e[1], x) + interp(e[2], x); }
  if (e[0] === "mul") { return interp(e[1], x) * interp(e[2], x); }
  if (e[0] === "sub") { return interp(e[1], x) - interp(e[2], x); }
  return 0;
}`;

// ---- residual-AST printer (back-end; renders mix's output to JS source) ----
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

// a concrete reference interpreter (Node) to act as the oracle for int(src, x)
function refInterp(e, x) {
  if (e[0] === "num") return e[1];
  if (e[0] === "x") return x;
  if (e[0] === "add") return refInterp(e[1], x) + refInterp(e[2], x);
  if (e[0] === "mul") return refInterp(e[1], x) * refInterp(e[2], x);
  if (e[0] === "sub") return refInterp(e[1], x) - refInterp(e[2], x);
  return 0;
}

const prog = parseProgram(INT_SRC);

// a few static source expressions to compile
const SRCS = {
  "x*x + 5": ["add", ["mul", ["x"], ["x"]], ["num", 5]],
  "(x - 2) * (x + 3)": ["mul", ["sub", ["x"], ["num", 2]], ["add", ["x"], ["num", 3]]],
  "x*x*x": ["mul", ["x"], ["mul", ["x"], ["x"]]],
  "((x+1)*(x+1)) - x": ["sub", ["mul", ["add", ["x"], ["num", 1]], ["add", ["x"], ["num", 1]]], ["x"]],
};

let fail = 0, total = 0;
for (const [label, srcExpr] of Object.entries(SRCS)) {
  // mix(int, src): src static, x dynamic
  const residAst = mix.specialize(prog, "interp", [["s", srcExpr], ["d", ["var", "x"]]]);
  const js = render(residAst);
  const fn = new Function("x", "return " + js + ";");
  let ok = true;
  for (const x of [-7, -2, 0, 1, 3, 5, 11, 42]) {
    const a = refInterp(srcExpr, x), b = fn(x);
    total++;
    if (a !== b) { ok = false; fail++; console.log(`  DIVERGE ${label} x=${x}: int=${a} residual=${b}`); }
  }
  console.log((ok ? "ok  " : "FAIL") + "  " + label.padEnd(22) + " -> " + js);
}
console.log(`\np1_mix: checks=${total} fail=${fail}`);
process.exit(fail ? 1 : 0);
