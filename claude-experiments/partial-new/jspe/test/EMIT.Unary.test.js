module.exports = [
  {
    name: "unary_not",
    fn: () => {
      const { EMIT, emit } = require("../ir.js");
      const e = { tag: "Unary", op: "!", a: { tag: "Bool", b: false } };
      const result = emit(e);
      if (result !== "(!false)") throw new Error("Expected (!false), got " + result);
    }
  },
  {
    name: "unary_typeof",
    fn: () => {
      const { EMIT, emit } = require("../ir.js");
      const e = { tag: "Unary", op: "typeof", a: { tag: "Var", id: 0 } };
      const result = emit(e);
      if (result !== "(typeof v0)") throw new Error("Expected (typeof v0), got " + result);
    }
  },
  {
    name: "unary_void",
    fn: () => {
      const { EMIT, emit } = require("../ir.js");
      const e = { tag: "Unary", op: "void", a: { tag: "Num", n: 42 } };
      const result = emit(e);
      if (result !== "(void 42)") throw new Error("Expected (void 42), got " + result);
    }
  },
  {
    name: "unary_neg",
    fn: () => {
      const { EMIT, emit } = require("../ir.js");
      const e = { tag: "Unary", op: "-", a: { tag: "Num", n: 5 } };
      const result = emit(e);
      if (result !== "(-5)") throw new Error("Expected (-5), got " + result);
    }
  },
  {
    name: "unary_tilde",
    fn: () => {
      const { EMIT, emit } = require("../ir.js");
      const e = { tag: "Unary", op: "~", a: { tag: "Num", n: 1 } };
      const result = emit(e);
      if (result !== "(~1)") throw new Error("Expected (~1), got " + result);
    }
  }
];