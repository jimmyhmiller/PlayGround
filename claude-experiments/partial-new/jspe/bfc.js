#!/usr/bin/env node
// bfc.js — COMPILE a Brainfuck program to JavaScript, by partially-evaluating the real
// BF interpreter (examples/bf_real.interp.js) with the program STATIC and input DYNAMIC.
// This is the first Futamura projection done by jspe (a full imperative partial evaluator).
//
//   node bfc.js '<bf-source>' [tapeSize]   > target.js     # emit compiled JS
// The emitted target is `function main(v1)` taking an input byte-array, returning out bytes.
const fs = require("fs");
const path = require("path");
const { specializeGeneral } = require("./jspe.js");

function parseBF(src) {                 // BF -> nested blocks (op = char code, loop = array)
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
const arg = process.argv[2];
if (!arg) { console.error("usage: node bfc.js '<bf-source>' [tapeSize] > target.js"); process.exit(2); }
const src = fs.existsSync(arg) ? fs.readFileSync(arg, "utf8") : arg;
const tapeSize = Number(process.argv[3] || 16);
let interpSrc = fs.readFileSync(path.join(__dirname, "examples", "bf_real.interp.js"), "utf8").replace(/TAPE_SIZE/g, String(tapeSize));
const prog = parseBF(src);
const targetJs = specializeGeneral(interpSrc, "interp", [{ s: prog }, { d: true }], "v1");
process.stdout.write(targetJs);
