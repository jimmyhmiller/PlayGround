module.exports = [
  {name: "call no args", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const e = RE.Call(RE.Var(0), []);
    const result = EMIT.Call(e, emit);
    if (result !== "v0()") throw new Error("Expected v0(), got " + result);
  }},
  {name: "call one arg", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const e = RE.Call(RE.Var(1), [RE.Num(42)]);
    const result = EMIT.Call(e, emit);
    if (result !== "v1(42)") throw new Error("Expected v1(42), got " + result);
  }},
  {name: "call multiple args", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const e = RE.Call(RE.Var(2), [RE.Str("a"), RE.Bool(true), RE.Num(3)]);
    const result = EMIT.Call(e, emit);
    if (result !== 'v2("a", true, 3)') throw new Error("Expected v2(\"a\", true, 3), got " + result);
  }}
];