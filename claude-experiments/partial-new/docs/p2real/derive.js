#!/usr/bin/env node
// Derive a compiler from an interpreter written in the JS subset.
//
//   node docs/p2real/derive.js <interpreter.js> [-o compiler.js] [--entry interp]
//
// Reads the interpreter, specializes jsmix w.r.t. it with RustPE (the second
// Futamura projection), and writes a SELF-CONTAINED, runnable compiler.
const fs = require("fs");
const path = require("path");
const os = require("os");
const { execSync } = require("child_process");
const DIR = __dirname, ROOT = path.resolve(DIR, "../..");
const { parseProgram } = require(path.join(DIR, "parse.js"));

const args = process.argv.slice(2);
const intFile = args.find((a) => !a.startsWith("-") && args[args.indexOf(a) - 1] !== "-o" && args[args.indexOf(a) - 1] !== "--entry");
const outFile = args.includes("-o") ? args[args.indexOf("-o") + 1] : null;
const entry = args.includes("--entry") ? args[args.indexOf("--entry") + 1] : "interp";
if (!intFile) {
  console.error("usage: node derive.js <interpreter.js> [-o compiler.js] [--entry interp]");
  process.exit(2);
}

const INT = parseProgram(fs.readFileSync(intFile, "utf8"));
const jsmixSrc = fs.readFileSync(path.join(DIR, "jsmix.js"), "utf8");
const tmp = path.join(os.tmpdir(), "p2_jsmix_" + process.pid + ".js");
fs.writeFileSync(tmp, jsmixSrc +
  `\nfunction main(prog){var INT=${JSON.stringify(INT)};return jsmix(INT,${JSON.stringify(entry)},prog);}\n`);

const bin = path.join(ROOT, "target/release/js-frontend");
const out = execSync(`SPEC_WEIGHT_BUDGET=12000000 ${bin} --js ${tmp}`, { encoding: "utf8", maxBuffer: 1 << 28 });
const lines = out.split("\n");
const start = lines.findIndex((l) => l.includes("residual as JavaScript"));
if (start < 0) { console.error("RustPE produced no residual:\n" + out); process.exit(1); }
const resid = lines.slice(start + 1).filter((l) => !l.startsWith("---")).join("\n").trim();
const compileBody = resid.replace(/^function main\(/, "function compile(");
const blocks = (out.match(/(\d+) block/) || [])[1];

const RUNTIME_AND_CLI = `
// compile(prog) returns a JS SOURCE STRING. To run it, make a native function.
module.exports = { compile: compile,
                   run: function(p,x){ return Function("x","return "+compile(p)+";")(x); } };
if (require.main === module) {
  var prog = JSON.parse(process.argv[2] || "[]");
  var src = compile(prog);
  console.log("compiled JS source:  function (x) { return " + src + "; }");
  if (process.argv[3] !== undefined)
    console.log("run(" + process.argv[3] + ") = " + Function("x","return "+src+";")(Number(process.argv[3])));
}
`;
const full = `// Compiler derived from ${path.basename(intFile)} by specializing jsmix (RustPE, ${blocks} blocks).\n` +
  `// Use:  node ${outFile || "compiler.js"} '<program-json>' [input]\n` +
  compileBody + "\n" + RUNTIME_AND_CLI;

if (outFile) { fs.writeFileSync(outFile, full); console.error(`derived a ${blocks}-block compiler -> ${outFile}`); }
else process.stdout.write(full);
