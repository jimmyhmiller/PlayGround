// whistle.js — loop termination/materialization (ported from src/js.rs). Owns the
// PE's "brain": detect a dynamically-controlled loop, materialize its carried slots
// to STABLE runtime vars (keyed by (head,slot)) so the canonical-key memo ties it.
const meta = require("./step/meta.js");
const { absTruthy, absToRExpr } = require("./state.js");
const { RE } = require("./contracts.js");

// Abs -> RExpr for materialization: a heap Ref becomes an array/object LITERAL
// (so a loop-carried array materializes to a stable runtime array variable).
function refToExpr(s, abs) {
  if (abs.tag !== "Ref") return absToRExpr(abs);
  const obj = s.heap.get(abs.addr);
  if (obj.tag === "Array") return RE.Arr(obj.elems.map((e) => refToExpr(s, e)));
  if (obj.tag === "Object") return RE.Opaque("Object.assign", [RE.Opaque("{}", []), ...obj.fields.map(([k, v]) => refToExpr(s, v))]);
  throw new Error("cannot materialize heap " + obj.tag);
}

const jsOp = { "+": (a, b) => a + b, "-": (a, b) => a - b, "*": (a, b) => a * b,
  "/": (a, b) => a / b, "%": (a, b) => a % b, "<": (a, b) => a < b, ">": (a, b) => a > b,
  "<=": (a, b) => a <= b, ">=": (a, b) => a >= b, "===": (a, b) => a === b, "!==": (a, b) => a !== b,
  "==": (a, b) => a == b, "!=": (a, b) => a != b, "&": (a, b) => a & b, "|": (a, b) => a | b,
  "^": (a, b) => a ^ b, "<<": (a, b) => a << b, ">>": (a, b) => a >> b, ">>>": (a, b) => a >>> b };
// concrete value of a static primitive Abs (mirror of arith.js staticVal)
function sv(v) {
  switch (v.tag) {
    case "Num": return { ok: true, v: v.n }; case "Str": return { ok: true, v: v.s };
    case "Bool": return { ok: true, v: v.b }; case "Undef": return { ok: true, v: undefined };
    case "Null": return { ok: true, v: null }; default: return { ok: false };
  }
}
function absBin(op, a, b) {
  const fa = sv(a), fb = sv(b);
  if (fa.ok && fb.ok && jsOp[op]) {
    const r = jsOp[op](fa.v, fb.v);
    if (typeof r === "boolean") return { tag: "Bool", b: r };
    if (typeof r === "string") return { tag: "Str", s: r };
    return { tag: "Num", n: r };
  }
  return { tag: "Dyn", expr: { tag: "Var", id: -1 } }; // placeholder: just needs to read as dynamic
}

// Dry-run the loop-head condition over current locals; true iff it's dynamic.
function dynamicallyControlled(s) {
  const code = meta.get().code;
  const top = s.frames[s.frames.length - 1];
  let pc = top.pc;
  const st = [];
  for (let g = 0; g < 256; g++) {
    const ins = code[pc];
    if (!ins) return true;
    switch (ins.tag) {
      case "PushNum": st.push({ tag: "Num", n: ins.n }); break;
      case "PushStr": st.push({ tag: "Str", s: ins.s }); break;
      case "PushBool": st.push({ tag: "Bool", b: ins.b }); break;
      case "PushUndef": st.push({ tag: "Undef" }); break;
      case "PushNull": st.push({ tag: "Null" }); break;
      case "Load": st.push(top.locals[ins.slot]); break;
      case "Bin": { const b = st.pop(), a = st.pop(); st.push(absBin(ins.op, a, b)); break; }
      case "Unary": { const a = st.pop(); st.push(a.tag === "Dyn" ? a : a); break; }
      case "GetProp": {  // a STATIC array .length keeps a counting loop static (it unrolls)
        const o = st.pop();
        if (o && o.tag === "Ref") { const obj = s.heap.get(o.addr);
          if (obj.tag === "Array" && ins.k === "length") { st.push({ tag: "Num", n: obj.elems.length }); break; }
          if (obj.tag === "Object") { const f = obj.fields.find((x) => x[0] === ins.k); if (f) { st.push(f[1]); break; } } }
        return true;
      }
      case "GetIndex": {  // a STATIC array at a STATIC index stays static; dynamic index -> dynamic
        const idx = st.pop(), arr = st.pop();
        if (arr && arr.tag === "Ref" && idx && idx.tag === "Num") { const obj = s.heap.get(arr.addr);
          if (obj.tag === "Array" && idx.n >= 0 && idx.n < obj.elems.length) { st.push(obj.elems[idx.n]); break; } }
        return true;
      }
      case "JmpIfFalsy": return absTruthy(st[st.length - 1]) === null;
      default: return true; // anything else in a condition -> conservatively dynamic
    }
    pc++;
  }
  return true;
}

// Bind each slot's current value to a STABLE var v<id> (id keyed by head+slot) and
// replace the slot with Dyn(Var id). Emits `v<id> = <expr>;`.
function materialize(s, slots, out, headPc) {
  const top = s.frames[s.frames.length - 1];
  for (const slot of slots) {
    const v = top.locals[slot];
    const id = 1000000 + (headPc | 0) * 256 + slot;
    if (v && v.tag === "Dyn" && v.expr && v.expr.tag === "Var" && v.expr.id === id) continue; // already THIS stable var
    out.push({ tag: "Store", name: "v" + id, expr: refToExpr(s, v) });   // Ref -> array literal; else value
    top.locals[slot] = { tag: "Dyn", expr: { tag: "Var", id } };
  }
}

// if-join merge marker consumption (no markers generated for plain while/if yet -> no-op).
function consumePendingJoins(s, target, out) { /* extended when nested joins land */ }

function loopDiverges(seen, cand) { return false; } // growth whistle (static-pc loops) — later

function whistle(seen, cand /*, ctx */) {
  // Only a LOOP HEAD can force generalization. A non-head point revisited with a
  // different state is normal straight-line re-specialization (unrolling), bounded by
  // the loop's own termination — generalizing there would wrongly abort a statically
  // bounded unroll the moment a carried dynamic value (e.g. acc) changes.
  const top = cand.frames[cand.frames.length - 1];
  if (!meta.get().loopHeads.has(top.pc)) return false;
  return JSON.stringify(seen) !== JSON.stringify(cand) && (dynamicallyControlled(cand) || loopDiverges(seen, cand));
}

function generalize(seen, from, out) {
  const g = structuredClone(from);
  const fi = g.frames.length - 1;
  const slots = [];
  for (let i = 0; i < g.frames[fi].locals.length; i++) {
    if (JSON.stringify(seen.frames[fi].locals[i]) !== JSON.stringify(from.frames[fi].locals[i])) slots.push(i);
  }
  materialize(g, slots, out, g.frames[fi].pc);
  return g;
}

module.exports = { dynamicallyControlled, materialize, consumePendingJoins, loopDiverges, whistle, generalize };
