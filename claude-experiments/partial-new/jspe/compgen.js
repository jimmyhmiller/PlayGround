#!/usr/bin/env node
// compgen.js — turn an INTERPRETER into a COMPILER, automatically.
//
// This is the second Futamura projection:   compiler = jspe(mix, interpreter)
// where `mix` is a partial evaluator written in jspe's own subset, and `jspe` is the
// partial evaluator. We specialize `mix` with respect to your interpreter; the residual
// is a standalone compiler for your interpreter's source language.
//
// Usage:
//   node compgen.js <interpreter.js> [entryFn] > compiler.js
//
// Then run the generated compiler directly with Node:
//   node compiler.js '<source-AST-json>'          # prints the compiled target (JS source)
//   node compiler.js '<source-AST-json>' <x>       # compiles AND runs the target at x
//
// Example:
//   node compgen.js examples/calc.interp.js > calc-compiler.js
//   node calc-compiler.js '["add",["mul",["x"],["x"]],["num",5]]'        #  (x * x) + 5
//   node calc-compiler.js '["add",["mul",["x"],["x"]],["num",5]]' 7      #  54
const fs = require("fs");
const path = require("path");
const { specializeGeneral } = require("./jspe.js");
const { parseProgram } = require("./parse.js");

const interpFile = process.argv[2];
const entryFn = process.argv[3] || "interp";
if (!interpFile) {
  console.error("usage: node compgen.js <interpreter.js> [entryFn] > compiler.js");
  process.exit(2);
}
const interpSrc = fs.readFileSync(interpFile, "utf8");
const INT = parseProgram(interpSrc);

// `mix` is a FIRST-ORDER FUNCTIONAL partial evaluator: it specializes interpreters built
// from var/if/return/block + pure expressions (incl. recursion). It does NOT residualize
// imperative constructs — `while` loops, assignment/mutation, or expression statements.
// Reject those up front with a clear error instead of silently dropping them.
function checkSupported(funcs) {
  const STMT_OK = new Set(["return", "var", "if", "block"]);
  const EXPR_OK = new Set(["lit", "var", "bin", "unary", "cond", "idx", "dot", "call", "arr"]);
  const bad = [];
  const expr = (n) => {
    if (!Array.isArray(n)) return;
    const t = n[0];
    if (t === "assign") { bad.push("assignment / mutation (`a = ...` or `a[i] = ...`)"); return; }
    if (t === "fun") { bad.push("function/closure expression"); return; }
    if (!EXPR_OK.has(t)) { bad.push("expression `" + t + "`"); return; }
    if (t === "bin") { expr(n[2]); expr(n[3]); }
    else if (t === "unary") expr(n[2]);
    else if (t === "cond") { expr(n[1]); expr(n[2]); expr(n[3]); }
    else if (t === "idx") { expr(n[1]); expr(n[2]); }
    else if (t === "dot") expr(n[1]);
    else if (t === "call") { expr(n[1]); n[2].forEach(expr); }
    else if (t === "arr") n[1].forEach(expr);
    // lit, var: leaves
  };
  const stmt = (s) => {
    if (s[0] === "while") bad.push("`while` loop");
    else if (s[0] === "expr") bad.push("expression statement (a statement that isn't var/if/return)");
    else if (!STMT_OK.has(s[0])) bad.push("statement `" + s[0] + "`");
    if (s[0] === "return") expr(s[1]);
    else if (s[0] === "var") expr(s[2]);
    else if (s[0] === "if") { expr(s[1]); stmt(s[2]); if (s[3]) stmt(s[3]); }
    else if (s[0] === "block") s[1].forEach(stmt);
  };
  for (const f of funcs) f[3].forEach(stmt);
  return [...new Set(bad)];
}
const unsupported = checkSupported(INT);
if (unsupported.length) {
  console.error("compgen: this interpreter uses constructs `mix` cannot compile yet:\n  - " +
    unsupported.join("\n  - ") +
    "\n\n`mix` is a first-order FUNCTIONAL partial evaluator. Write the interpreter with only:" +
    "\n  statements: var, if/else, return, { blocks }   (NO while, NO assignment)" +
    "\n  expressions: literals, vars, + - * / === < ..., ?:, a[i], a.b, typeof, named calls (recursion OK)" +
    "\n\nImperative interpreters (mutable tape, loops — e.g. Brainfuck) need mix's imperative" +
    "\nresidualization, which isn't built yet.");
  process.exit(1);
}

// load mix (the partial evaluator, written in jspe's subset)
let mixSrc = fs.readFileSync(path.join(__dirname, "mix.subset.js"), "utf8");
mixSrc = mixSrc.slice(mixSrc.indexOf("function envNil"), mixSrc.indexOf("if (typeof module"));

// GENERATE: jspe(mix, interpreter) with the source program DYNAMIC -> the compiler body.
const argVals = [["d", { __dyn: { tag: "Var", id: 0 } }], ["d", ["var", "x"]]];
const mainSrc = specializeGeneral(mixSrc, "specializeProg", [{ s: INT }, { s: entryFn }, { s: argVals }], "v0");

// inline the back-end (simplify + render) so the emitted compiler is fully self-contained.
let backendSrc = fs.readFileSync(path.join(__dirname, "backend.js"), "utf8");
backendSrc = backendSrc.replace(/module\.exports[\s\S]*$/, "");

const banner = `// ============================================================================
// AUTO-GENERATED COMPILER  —  do not edit by hand.
// Produced by the 2nd Futamura projection:  compiler = jspe(mix, ${path.basename(interpFile)})
// Entry interpreter function: ${entryFn}().  Standalone — run it with Node:
//   node <this-file> '<source-AST-json>'        prints the compiled target (JS)
//   node <this-file> '<source-AST-json>' <x>    compiles AND runs the target at x
// ============================================================================`;

const cli = `
// ---- command line ----------------------------------------------------------
const __args = process.argv.slice(2);
if (__args.length === 0) {
  console.error("usage: node " + require("path").basename(process.argv[1]) + " '<source-AST-json>' [x]");
  process.exit(2);
}
let __src;
try { __src = JSON.parse(__args[0]); }
catch (e) { console.error("source must be JSON, e.g. '[\\"add\\",[\\"x\\"],[\\"num\\",5]]' — " + e.message); process.exit(2); }
const __prog = main(__src);                       // <- the generated compiler runs here
const __clean = simplify(__prog);                 // constant-fold the (now-static) source dispatch
const { entry: __e, defs: __d } = renderProg(__clean);
const __targetSrc = __d + "function target(x) { return " + __e + "; }";
if (__args.length >= 2) {
  const __x = Number(__args[1]);
  const __target = new Function(__targetSrc + "; return target;")();
  console.log(__target(__x));                      // compile AND run
} else {
  console.log("// target for " + JSON.stringify(__src));
  console.log(__targetSrc);                         // just print the compiled target
}
`;

process.stdout.write(banner + "\n" + mainSrc + "\n" + backendSrc + "\n" + cli);
