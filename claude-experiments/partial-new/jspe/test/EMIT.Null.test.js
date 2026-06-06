module.exports = [
  {name: "null literal", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const e = { tag: "Null" };
    const result = EMIT.Null(e, emit);
    if (result !== 'null') throw new Error(`Expected 'null', got ${result}`);
  }},
  {name: "null in cond", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    // Ensure EMIT.Cond is implemented (stub for test)
    EMIT.Cond = (e, rec) => "(" + rec(e.c) + " ? " + rec(e.t) + " : " + rec(e.e) + ")";
    const e = { tag: "Cond", c: { tag: "Null" }, t: { tag: "Num", n: 1 }, e: { tag: "Num", n: 0 } };
    const result = emit(e);
    if (result !== '(null ? 1 : 0)') throw new Error(`Expected '(null ? 1 : 0)', got ${result}`);
  }},
  {name: "null in bin", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const e = { tag: "Bin", op: "===", a: { tag: "Null" }, b: { tag: "Null" } };
    const result = emit(e);
    if (result !== '(null === null)') throw new Error(`Expected '(null === null)', got ${result}`);
  }}
];