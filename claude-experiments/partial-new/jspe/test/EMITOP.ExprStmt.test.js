module.exports = [
  {name: "ExprStmt basic", fn: () => {
    const { EMITOP, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const expr = RE.Num(42);
    const op = { tag: "ExprStmt", expr };
    const result = EMITOP.ExprStmt(op, emit);
    if (result !== "42;") throw new Error("Expected '42;', got " + result);
  }},
  {name: "ExprStmt with binary expression", fn: () => {
    const { EMITOP, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const expr = RE.Bin("+", RE.Num(1), RE.Num(2));
    const op = { tag: "ExprStmt", expr };
    const result = EMITOP.ExprStmt(op, emit);
    if (result !== "(1 + 2);") throw new Error("Expected '(1 + 2);', got " + result);
  }},
  {name: "ExprStmt with string literal", fn: () => {
    const { EMITOP, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const expr = RE.Str("hello");
    const op = { tag: "ExprStmt", expr };
    const result = EMITOP.ExprStmt(op, emit);
    if (result !== '"hello";') throw new Error('Expected "hello";, got ' + result);
  }}
];