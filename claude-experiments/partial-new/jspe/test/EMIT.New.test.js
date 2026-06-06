module.exports = [
  {name: "new with no args", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const e = { tag: "New", f: { tag: "Var", id: 0 }, args: [] };
    const result = EMIT.New(e, emit);
    if (result !== "new v0()") throw new Error("Expected 'new v0()', got: " + result);
  }},
  {name: "new with one arg", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const e = { tag: "New", f: { tag: "Var", id: 1 }, args: [{ tag: "Num", n: 42 }] };
    const result = EMIT.New(e, emit);
    if (result !== "new v1(42)") throw new Error("Expected 'new v1(42)', got: " + result);
  }},
  {name: "new with multiple args", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const e = { tag: "New", f: { tag: "Var", id: 2 }, args: [{ tag: "Num", n: 1 }, { tag: "Str", s: "hello" }] };
    const result = EMIT.New(e, emit);
    if (result !== 'new v2(1, "hello")') throw new Error("Expected 'new v2(1, \"hello\")', got: " + result);
  }}
];