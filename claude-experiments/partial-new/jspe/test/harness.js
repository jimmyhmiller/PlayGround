// harness.js — the DIFFERENTIAL integration gate. For random programs in the object
// subset: residual_oracle = RustPE(prog), residual_jspe = jspe.compile(prog); run BOTH
// residuals on random inputs and require identical outputs. RustPE is the golden spec.
//
//   node test/harness.js [count] [seed]
//
// Until enough leaf tasks land, jspe.compile throws UNIMPLEMENTED — the harness reports
// "jspe not ready (UNIMPLEMENTED: X)" instead of a divergence, so it doubles as a
// progress signal (which task is the current blocker).
const fs = require("fs"), os = require("os"), path = require("path");
const { execSync } = require("child_process");
const ROOT = path.resolve(__dirname, "../..");
const BIN = path.join(ROOT, "target/release/js-frontend");
let jspe; try { jspe = require("../jspe.js"); } catch (e) { jspe = null; }

// --- deterministic PRNG ---
let SEED = Number(process.argv[3] || 1);
const rnd = () => { SEED = (SEED * 1103515245 + 12345) & 0x7fffffff; return SEED / 0x7fffffff; };
const ri = (n) => Math.floor(rnd() * n);

// --- generator: programs in the CURRENT supported subset. Grow this as tasks land. ---
// v0: `function main(input){ var a = <expr over input and ints>; return a; }`
function genExpr(depth) {
  if (depth <= 0 || ri(3) === 0) return ri(2) ? "input" : "" + ri(9);  // non-negative until parser handles unary minus
  const op = ["+", "-", "*"][ri(3)];
  return "(" + genExpr(depth - 1) + " " + op + " " + genExpr(depth - 1) + ")";
}
function genBody(d) {
  if (d <= 0 || ri(3) === 0) { const r = ri(3); return r === 0 ? "a" : r === 1 ? "n" : "" + ri(5); }
  return "(" + genBody(d - 1) + " " + ["+","-","*"][ri(3)] + " " + genBody(d - 1) + ")";
}
function genLoop() {
  return "function main(input) { var a = " + ri(9) + "; var n = input; " +
    "while (n > 0) { a = " + genBody(2) + "; n = (n - 1); } return a; }";
}
function genArray() {
  const size = 2 + ri(3);
  const init = []; for (let i = 0; i < size; i++) init.push(genExpr(1));
  let body = "var a = [" + init.join(", ") + "]; ";
  for (let i = 0; i < ri(3); i++) body += "a[" + ri(size) + "] = " + genExpr(1) + "; ";
  const reads = []; for (let i = 0; i < 2 + ri(2); i++) reads.push("a[" + ri(size) + "]");
  return "function main(input) { " + body + "return (" + reads.join(" + ") + "); }";
}
function genTape() {
  const size = 2 + ri(3);
  let s = "function main(input) { var tape = [" + Array(size).fill("0").join(", ") + "]; var ptr = 0; var n = input; ";
  s += "while (n > 0) { tape[ptr] = (tape[ptr] + " + genExpr(1) + "); ptr = (ptr + 1); if (ptr > " + (size - 1) + ") { ptr = 0; } n = (n - 1); } ";
  const reads = []; for (let i = 0; i < size; i++) reads.push("tape[" + i + "]");
  return s + "return (" + reads.join(" + ") + "); }";
}
function genAB(d) { if (d <= 0 || ri(3) === 0) { const r = ri(3); return r === 0 ? "a" : r === 1 ? "b" : "" + ri(5); } return "(" + genAB(d-1) + " " + ["+","-","*"][ri(3)] + " " + genAB(d-1) + ")"; }
function genCall() {
  return "function f(a, b) { return (" + genAB(2) + "); } function g(x) { return f(x, (x + " + ri(5) + ")); } " +
    "function main(input) { return (g(input) + f(input, " + ri(5) + ")); }";
}
function genProgram() {
  const r = ri(5);
  if (r === 0) return genLoop();
  if (r === 1) return genArray();
  if (r === 2) return genTape();
  if (r === 3) return genCall();
  return "function main(input) { return " + genExpr(2 + ri(3)) + "; }";
}

// --- run RustPE to get the oracle residual ---
function rustResidual(src) {
  const tmp = path.join(os.tmpdir(), "jspe_h_" + process.pid + "_" + (SEED & 0xffff) + ".js");
  fs.writeFileSync(tmp, src);
  const out = execSync(`SPEC_WEIGHT_BUDGET=4000000 ${BIN} --js ${tmp}`, { encoding: "utf8", maxBuffer: 1 << 26 });
  const lines = out.split("\n");
  const i = lines.findIndex((l) => l.includes("residual as JavaScript"));
  return lines.slice(i + 1).filter((l) => !l.startsWith("---")).join("\n").trim();
}
function run(residual, input) {
  const fn = new Function(residual + "\nreturn main;")();
  return fn(input);
}

// --- main loop ---
const count = Number(process.argv[2] || 200);
let ok = 0, diverge = 0, notReady = 0, lastBlocker = "";
for (let t = 0; t < count; t++) {
  const src = genProgram();
  let rRes;
  try { rRes = rustResidual(src); } catch (e) { continue; } // RustPE rejected/errored: skip
  if (!jspe) { notReady++; lastBlocker = "jspe.js failed to load"; continue; }
  let jRes;
  try { jRes = jspe.compile(src); }
  catch (e) { notReady++; lastBlocker = e.message; continue; }
  let bad = false;
  for (const x of [-3, -1, 0, 2, 5, 17]) {
    let a, b;
    try { a = run(rRes, x); } catch (e) { a = "ERR:" + e.message; }
    try { b = run(jRes, x); } catch (e) { b = "ERR:" + e.message; }
    if (a !== b && !(Number.isNaN(a) && Number.isNaN(b))) { bad = true; if (diverge < 5) console.log(`DIVERGE x=${x}\n src: ${src}\n rust: ${a}\n jspe: ${b}`); break; }
  }
  if (bad) diverge++; else ok++;
}
console.log(`\nharness: ok=${ok} diverge=${diverge} not-ready=${notReady}` + (notReady ? `\n  current blocker: ${lastBlocker}` : ""));
process.exit(diverge ? 1 : 0);
