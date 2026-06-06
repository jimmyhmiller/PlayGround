// step/mem.js — heap ops with partial-static (scalar-replaceable) arrays/objects.
// Static array + static index folds (the array can vanish from the residual); a
// dynamic access escapes via materializeValue (emits construction Ops to `out`).
const { RE, OP, AB } = require("../contracts.js");
const { absToRExpr } = require("../state.js");
const H = {};
const ost = (s) => s.frames[s.frames.length - 1].ostack;
const escVar = (addr) => 2000000 + addr;

// Abs -> RExpr, emitting any heap construction to `out`. Stable var per address.
H.materializeValue = (s, abs, out) => {
  if (abs.tag === "Ref") {
    const obj = s.heap.get(abs.addr), id = escVar(abs.addr);
    if (obj.tag === "Array") {
      out.push(OP.Store("v" + id, RE.Arr(obj.elems.map((e) => H.materializeValue(s, e, out)))));
      return RE.Var(id);
    }
    if (obj.tag === "Object") {
      out.push(OP.Store("v" + id, RE.Opaque("{}", [])));
      for (const [k, v] of obj.fields) out.push(OP.SetProp(RE.Var(id), k, H.materializeValue(s, v, out)));
      return RE.Var(id);
    }
    throw new Error("cannot materialize heap " + obj.tag);
  }
  return absToRExpr(abs); // Num/Str/Bool/Undef/Null/Dyn
};

H.NewArray = (s, i, out) => {
  const elems = [];
  for (let j = 0; j < i.n; j++) elems.unshift(ost(s).pop());
  const addr = s.nextAddr++;
  s.heap.set(addr, { tag: "Array", elems });
  ost(s).push(AB.Ref(addr));
  return { tag: "Continue" };
};
H.NewObject = (s, i, out) => {
  const fields = [];
  for (let j = i.keys.length - 1; j >= 0; j--) fields[j] = [i.keys[j], ost(s).pop()];
  const addr = s.nextAddr++;
  s.heap.set(addr, { tag: "Object", fields });
  ost(s).push(AB.Ref(addr));
  return { tag: "Continue" };
};
H.GetIndex = (s, i, out) => {
  const idx = ost(s).pop(), arr = ost(s).pop();
  if (arr.tag === "Ref" && idx.tag === "Num") {
    const obj = s.heap.get(arr.addr);
    if (obj && obj.tag === "Array" && idx.n >= 0 && idx.n < obj.elems.length) { ost(s).push(obj.elems[idx.n]); return { tag: "Continue" }; }
  }
  ost(s).push(AB.Dyn(RE.Index(H.materializeValue(s, arr, out), H.materializeValue(s, idx, out))));
  return { tag: "Continue" };
};
H.SetIndexOp = (s, i, out) => {
  const val = ost(s).pop(), idx = ost(s).pop(), arr = ost(s).pop();
  if (arr.tag === "Ref" && idx.tag === "Num") {
    const obj = s.heap.get(arr.addr);
    if (obj && obj.tag === "Array" && idx.n >= 0 && idx.n < obj.elems.length) { obj.elems[idx.n] = val; return { tag: "Continue" }; }
  }
  out.push(OP.SetIndex(H.materializeValue(s, arr, out), H.materializeValue(s, idx, out), H.materializeValue(s, val, out)));
  return { tag: "Continue" };
};
H.GetProp = (s, i, out) => {
  const o = ost(s).pop();
  if (o.tag === "Ref") {
    const obj = s.heap.get(o.addr);
    if (obj.tag === "Array" && i.k === "length") { ost(s).push(AB.Num(obj.elems.length)); return { tag: "Continue" }; }
    if (obj.tag === "Object") { const f = obj.fields.find((x) => x[0] === i.k); if (f) { ost(s).push(f[1]); return { tag: "Continue" }; } }
  }
  ost(s).push(AB.Dyn(RE.Get(H.materializeValue(s, o, out), i.k)));
  return { tag: "Continue" };
};
H.SetPropOp = (s, i, out) => {
  const val = ost(s).pop(), o = ost(s).pop();
  if (o.tag === "Ref") { const obj = s.heap.get(o.addr); if (obj.tag === "Object") { const f = obj.fields.find((x) => x[0] === i.k); if (f) f[1] = val; else obj.fields.push([i.k, val]); return { tag: "Continue" }; } }
  out.push(OP.SetProp(H.materializeValue(s, o, out), i.k, H.materializeValue(s, val, out)));
  return { tag: "Continue" };
};
module.exports = H;
