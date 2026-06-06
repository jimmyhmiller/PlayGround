module.exports = [
  {name: "opaque basic", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const e = RE.Opaque("foo", [RE.Num(1), RE.Str("bar")]);
    const result = emit(e);
    if (result !== "foo(1, \"bar\")") throw new Error("Expected foo(1, \"bar\") got " + result);
  }},
  {name: "opaque no args", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const e = RE.Opaque("baz", []);
    const result = emit(e);
    if (result !== "baz()") throw new Error("Expected baz() got " + result);
  }},
  {name: "opaque nested", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const { RE } = require("../contracts.js");
    const inner = RE.Opaque("inner", [RE.Num(2)]);
    const e = RE.Opaque("outer", [inner, RE.Bool(true)]);
    const result = emit(e);
    if (result !== "outer(inner(2), true)") throw new Error("Expected outer(inner(2), true) got " + result);
  }}
];