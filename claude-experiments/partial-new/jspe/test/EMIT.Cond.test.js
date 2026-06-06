module.exports = [
  {name: "Cond true branch", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const e = RE.Cond(RE.Bool(true), RE.Num(1), RE.Num(2));
    const result = emit(e);
    if (result !== "(true ? 1 : 2)") throw new Error("Expected (true ? 1 : 2), got " + result);
  }},
  {name: "Cond false branch", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const e = RE.Cond(RE.Bool(false), RE.Str("a"), RE.Str("b"));
    const result = emit(e);
    if (result !== "(false ? \"a\" : \"b\")") throw new Error("Expected (false ? \"a\" : \"b\"), got " + result);
  }},
  {name: "Cond nested", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const inner = RE.Cond(RE.Bool(true), RE.Num(1), RE.Num(2));
    const outer = RE.Cond(RE.Bool(false), inner, RE.Num(3));
    const result = emit(outer);
    if (result !== "(false ? (true ? 1 : 2) : 3)") throw new Error("Expected (false ? (true ? 1 : 2) : 3), got " + result);
  }}
];