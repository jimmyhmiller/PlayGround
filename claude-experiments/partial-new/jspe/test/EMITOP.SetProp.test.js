module.exports = [
  {name: "setprop basic", fn: () => {
    const { EMITOP, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const o = { tag: "SetProp", a: RE.Var(0), k: "x", v: RE.Num(42) };
    const result = EMITOP.SetProp(o, emit);
    if (result !== "v0.x = 42;") throw new Error("Expected 'v0.x = 42;', got: " + result);
  }},
  {name: "setprop with string key", fn: () => {
    const { EMITOP, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const o = { tag: "SetProp", a: RE.Var(1), k: "foo", v: RE.Str("bar") };
    const result = EMITOP.SetProp(o, emit);
    if (result !== 'v1.foo = "bar";') throw new Error("Expected 'v1.foo = \"bar\";', got: " + result);
  }},
  {name: "setprop with nested expression", fn: () => {
    const { EMITOP, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const o = { tag: "SetProp", a: RE.Bin("+", RE.Var(2), RE.Num(1)), k: "y", v: RE.Bool(true) };
    const result = EMITOP.SetProp(o, emit);
    if (result !== "(v2 + 1).y = true;") throw new Error("Expected '(v2 + 1).y = true;', got: " + result);
  }}
];