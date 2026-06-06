module.exports = [
  {name: "emit bool true", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const e = { tag: "Bool", b: true };
    const result = emit(e);
    if (result !== "true") throw new Error("Expected 'true', got " + result);
  }},
  {name: "emit bool false", fn: () => {
    const { EMIT, emit } = require("../ir.js");
    const e = { tag: "Bool", b: false };
    const result = emit(e);
    if (result !== "false") throw new Error("Expected 'false', got " + result);
  }},
  {name: "emit bool via EMIT directly", fn: () => {
    const { EMIT } = require("../ir.js");
    const e = { tag: "Bool", b: true };
    const result = EMIT.Bool(e, (x) => { throw new Error("should not recurse"); });
    if (result !== "true") throw new Error("Expected 'true', got " + result);
  }}
];