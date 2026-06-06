module.exports = [
  {name: "setindex basic", fn: () => {
    const { EMITOP, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const o = { tag: "SetIndex", a: RE.Var(0), i: RE.Num(1), v: RE.Num(42) };
    const result = EMITOP.SetIndex(o, emit);
    if (result !== "v0[1] = 42;") throw new Error("Expected v0[1] = 42;, got " + result);
  }},
  {name: "setindex with string index", fn: () => {
    const { EMITOP, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const o = { tag: "SetIndex", a: RE.Var(1), i: RE.Str("key"), v: RE.Bool(true) };
    const result = EMITOP.SetIndex(o, emit);
    if (result !== 'v1["key"] = true;') throw new Error("Expected v1[\"key\"] = true;, got " + result);
  }},
  {name: "setindex nested expression", fn: () => {
    const { EMITOP, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const o = { tag: "SetIndex", a: RE.Index(RE.Var(2), RE.Num(0)), i: RE.Num(3), v: RE.Num(7) };
    const result = EMITOP.SetIndex(o, emit);
    if (result !== "v2[0][3] = 7;") throw new Error("Expected v2[0][3] = 7;, got " + result);
  }}
];