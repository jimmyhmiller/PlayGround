module.exports = [
  {name: "index basic", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const e = RE.Index(RE.Var(0), RE.Num(1));
    const result = emit(e);
    if (result !== "v0[1]") throw new Error("Expected v0[1], got " + result);
  }},
  {name: "index nested", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const e = RE.Index(RE.Index(RE.Var(0), RE.Num(0)), RE.Num(1));
    const result = emit(e);
    if (result !== "v0[0][1]") throw new Error("Expected v0[0][1], got " + result);
  }},
  {name: "index with string key", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const e = RE.Index(RE.Var(0), RE.Str("key"));
    const result = emit(e);
    if (result !== 'v0["key"]') throw new Error('Expected v0["key"], got ' + result);
  }}
];