module.exports = [
  {name: "get basic", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const e = RE.Get(RE.Var(0), "foo");
    const result = emit(e);
    if (result !== "v0.foo") throw new Error("Expected v0.foo, got " + result);
  }},
  {name: "get nested", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const inner = RE.Get(RE.Var(1), "bar");
    const e = RE.Get(inner, "baz");
    const result = emit(e);
    if (result !== "v1.bar.baz") throw new Error("Expected v1.bar.baz, got " + result);
  }},
  {name: "get with number", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const e = RE.Get(RE.Num(42), "prop");
    const result = emit(e);
    if (result !== "42.prop") throw new Error("Expected 42.prop, got " + result);
  }}
];