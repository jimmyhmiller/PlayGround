// step/arith.js — push/load/store/bin. Each export is one task (tasks.json step.*).
// Handler: (s, instr, out) -> Step.  Ports the matching arm of src/js.rs `step`.
const { UNIMPLEMENTED } = require("../contracts.js");

const H = {};
// [task step.push] one handler shared by the Push* instrs (PushNum/Str/Bool/Undef/Null)
H.PushNum = (s, i, out) => {
  const val = { tag: "Num", n: i.n };
  s.frames[s.frames.length - 1].ostack.push(val);
  return { tag: "Continue" };
};
H.PushStr = (s, i, out) => {
  const val = { tag: "Str", s: i.s };
  s.frames[s.frames.length - 1].ostack.push(val);
  return { tag: "Continue" };
};
H.PushBool = (s, i, out) => {
  const val = { tag: "Bool", b: i.b };
  s.frames[s.frames.length - 1].ostack.push(val);
  return { tag: "Continue" };
};
H.PushUndef = (s, i, out) => {
  const val = { tag: "Undef" };
  s.frames[s.frames.length - 1].ostack.push(val);
  return { tag: "Continue" };
};
H.PushNull = (s, i, out) => {
  const val = { tag: "Null" };
  s.frames[s.frames.length - 1].ostack.push(val);
  return { tag: "Continue" };
};
// [task step.load]
H.Load = (s, i, out) => {
  const slot = i.slot;
  const val = s.frames[s.frames.length - 1].locals[slot];
  s.frames[s.frames.length - 1].ostack.push(val);
  return { tag: "Continue" };
};
// [task step.store]
H.Store = (s, i, out) => {
  const slot = i.slot;
  const val = s.frames[s.frames.length - 1].ostack.pop();
  s.frames[s.frames.length - 1].locals[slot] = val;
  return { tag: "Continue" };
};
// [task step.bin]   uses BINOP (task step.evalbin)
const { absToRExpr } = require("../state.js");
const { AB, RE } = require("../contracts.js");
H.Bin = (s, i, out) => {
  const top = s.frames[s.frames.length - 1];
  const b = top.ostack.pop(), a = top.ostack.pop();
  top.ostack.push(evalBin(i.op, a, b));
  return { tag: "Continue" };
};
// typeof (and !) folded on static values; Refs are heap arrays/objects -> "object".
H.Unary = (s, i, out) => {
  const top = s.frames[s.frames.length - 1];
  const a = top.ostack.pop();
  top.ostack.push(evalUnary(i.op, a));
  return { tag: "Continue" };
};
function evalUnary(op, a) {
  if (op === "typeof") {
    switch (a.tag) {
      case "Num": return AB.Str("number");
      case "Str": return AB.Str("string");
      case "Bool": return AB.Str("boolean");
      case "Undef": return AB.Str("undefined");
      case "Null": return AB.Str("object");
      case "Ref": return AB.Str("object"); // arrays/objects
      default: return AB.Dyn(RE.Unary("typeof", absToRExpr(a)));
    }
  }
  if (op === "!") {
    const t = a.tag === "Num" ? a.n !== 0 : a.tag === "Str" ? a.s !== "" : a.tag === "Bool" ? a.b : a.tag === "Undef" || a.tag === "Null" ? false : a.tag === "Ref" ? true : null;
    if (t !== null) return AB.Bool(!t);
    return AB.Dyn(RE.Unary("!", absToRExpr(a)));
  }
  return AB.Dyn(RE.Unary(op, absToRExpr(a)));
}
const num = (v) => v.tag === "Num";
// A statically-known PRIMITIVE Abs -> {ok, v} with its concrete JS value. Refs/Dyn are not static.
function staticVal(v) {
  switch (v.tag) {
    case "Num": return { ok: true, v: v.n };
    case "Str": return { ok: true, v: v.s };
    case "Bool": return { ok: true, v: v.b };
    case "Undef": return { ok: true, v: undefined };
    case "Null": return { ok: true, v: null };
    default: return { ok: false };
  }
}
// lift a concrete JS value back to an Abs primitive (only the types JS comparisons/arith yield)
function liftVal(r) {
  if (typeof r === "number") return AB.Num(r);
  if (typeof r === "string") return AB.Str(r);
  if (typeof r === "boolean") return AB.Bool(r);
  if (r === undefined) return AB.Undef();
  if (r === null) return AB.Null();
  throw new Error("liftVal: non-primitive result " + typeof r);
}
// Pure JS semantics for every op we fold. Both operands already known static.
const JSOP = {
  "+": (a, b) => a + b, "-": (a, b) => a - b, "*": (a, b) => a * b, "/": (a, b) => a / b,
  "%": (a, b) => a % b, "===": (a, b) => a === b, "!==": (a, b) => a !== b,
  "==": (a, b) => a == b, "!=": (a, b) => a != b, "<": (a, b) => a < b, "<=": (a, b) => a <= b,
  ">": (a, b) => a > b, ">=": (a, b) => a >= b, "&": (a, b) => a & b, "|": (a, b) => a | b,
  "^": (a, b) => a ^ b, "<<": (a, b) => a << b, ">>": (a, b) => a >> b, ">>>": (a, b) => a >>> b,
};
// One folding rule for ALL ops: if both operands are static primitives, compute with real JS
// semantics; otherwise residualize. This is the static-comparison fold that lets an
// interpreter's opcode dispatch (op === "...") collapse — the prerequisite for P1/P2.
const BINOP = {};
function evalBin(op, a, b) {
  const f = JSOP[op];
  if (f) {
    const sa = staticVal(a), sb = staticVal(b);
    if (sa.ok && sb.ok) return liftVal(f(sa.v, sb.v));
  }
  return AB.Dyn(RE.Bin(op, absToRExpr(a), absToRExpr(b))); // residualize when any operand is dynamic
}
H.BINOP = BINOP;

module.exports = H;
