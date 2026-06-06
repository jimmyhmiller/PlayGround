module.exports = [
  {name: "pusharr_basic", fn: () => {
    const { EMITOP, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const o = { tag: "PushArr", a: RE.Var(0), v: RE.Num(42) };
    const result = EMITOP.PushArr(o, emit);
    if (result !== "v0.push(42);") throw new Error("Expected v0.push(42); got " + result);
  }},
  {name: "pusharr_nested", fn: () => {
    const { EMITOP, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const o = { tag: "PushArr", a: RE.Index(RE.Var(1), RE.Num(0)), v: RE.Str("hello") };
    const result = EMITOP.PushArr(o, emit);
    if (result !== "v1[0].push(\"hello\");") throw new Error("Expected v1[0].push(\"hello\"); got " + result);
  }},
  {name: "pusharr_complex_expr", fn: () => {
    const { EMITOP, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const o = { tag: "PushArr", a: RE.Bin("+", RE.Var(2), RE.Num(1)), v: RE.Bool(true) };
    const result = EMITOP.PushArr(o, emit);
    if (result !== "(v2 + 1).push(true);") throw new Error("Expected (v2 + 1).push(true); got " + result);
  }}
];