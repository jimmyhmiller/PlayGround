module.exports = [
  {name: "store_var", fn: () => {
    const { EMITOP, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const o = { tag: "Store", name: 0, expr: RE.Num(42) };
    const result = EMITOP.Store(o, emit);
    if (result !== "v0 = 42;") throw new Error("Expected 'v0 = 42;', got " + result);
  }},
  {name: "store_string_name", fn: () => {
    const { EMITOP, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const o = { tag: "Store", name: "x", expr: RE.Str("hello") };
    const result = EMITOP.Store(o, emit);
    if (result !== "x = \"hello\";") throw new Error("Expected 'x = \"hello\";', got " + result);
  }},
  {name: "store_bool", fn: () => {
    const { EMITOP, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const o = { tag: "Store", name: 1, expr: RE.Bool(false) };
    const result = EMITOP.Store(o, emit);
    if (result !== "v1 = false;") throw new Error("Expected 'v1 = false;', got " + result);
  }}
];