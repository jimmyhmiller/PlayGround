module.exports = [
  {name: "Eval basic", fn: () => {
    const { EMITOP, emit, varName } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const o = { tag: "Eval", dst: 0, expr: RE.Num(42) };
    const result = EMITOP.Eval(o, emit);
    if (result !== "let v0 = 42;") throw new Error("Expected 'let v0 = 42;', got: " + result);
  }},
  {name: "Eval with var expr", fn: () => {
    const { EMITOP, emit, varName } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const o = { tag: "Eval", dst: 1, expr: RE.Var(0) };
    const result = EMITOP.Eval(o, emit);
    if (result !== "let v1 = v0;") throw new Error("Expected 'let v1 = v0;', got: " + result);
  }},
  {name: "Eval with bin expr", fn: () => {
    const { EMITOP, emit, varName } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const o = { tag: "Eval", dst: 2, expr: RE.Bin("+", RE.Num(1), RE.Num(2)) };
    const result = EMITOP.Eval(o, emit);
    if (result !== "let v2 = (1 + 2);") throw new Error("Expected 'let v2 = (1 + 2);', got: " + result);
  }}
];