#!/usr/bin/env node
// bf.js — run a real Brainfuck program with the interpreter (examples/bf_real.interp.js).
//   node bf.js '<bf-source>' [input]
//   node bf.js program.bf [input]
const fs = require("fs");
const path = require("path");

// parse BF into NESTED blocks: an op is a char code (number), a loop is a nested array.
function parseBF(src) {
  let i = 0;
  function block() {
    const items = [];
    while (i < src.length) {
      const c = src[i]; i++;
      if (c === "]") return items;
      if ("+-<>.,".indexOf(c) >= 0) items.push(c.charCodeAt(0));
      else if (c === "[") items.push(block());
    }
    return items;
  }
  return block();
}
function loadInterp(tapeSize) {
  let s = fs.readFileSync(path.join(__dirname, "examples", "bf_real.interp.js"), "utf8");
  s = s.replace(/TAPE_SIZE/g, String(tapeSize));
  return new Function(s + "; return interp;")();
}

const arg = process.argv[2];
if (!arg) { console.error("usage: node bf.js '<bf-source>' [input]"); process.exit(2); }
const src = fs.existsSync(arg) ? fs.readFileSync(arg, "utf8") : arg;
const inputStr = process.argv[3] || "";
const interp = loadInterp(30000);
const out = interp(parseBF(src), [...inputStr].map((c) => c.charCodeAt(0)));
process.stdout.write(out.map((b) => String.fromCharCode(b)).join(""));
