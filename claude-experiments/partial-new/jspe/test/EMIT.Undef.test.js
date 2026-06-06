module.exports = [
  {name: "undef", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const e = { tag: "Undef" };
    const result = EMIT.Undef(e, emit);
    if (result !== 'undefined') throw new Error("Expected 'undefined', got " + result);
  }},
  {name: "undef via emit", fn: () => {
    const { emit } = require("../ir.js");
    const e = { tag: "Undef" };
    const result = emit(e);
    if (result !== 'undefined') throw new Error("Expected 'undefined', got " + result);
  }},
  {name: "undef in bin", fn: () => {
    const { emit } = require("../ir.js");
    const e = { tag: "Bin", op: "===", a: { tag: "Undef" }, b: { tag: "Undef" } };
    const result = emit(e);
    if (result !== '(undefined === undefined)') throw new Error("Expected '(undefined === undefined)', got " + result);
  }}
];