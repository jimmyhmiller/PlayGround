// mix_imp.js — an IMPERATIVE online partial evaluator (Node prototype; later ported to
// the subset for P2 self-application). Specializes interpreters built from var/assign/
// if/while/return/block + pure expressions + mutable arrays + named calls.
//
// Abstract values:  {k:"s", v}            a STATIC value (number/string/bool/null)
//                   {k:"d", e}            a DYNAMIC value, described by a residual expr `e`
//                   {k:"r", a}            a reference into the abstract heap (an array)
// Heap:  addr -> {elems:[absval...], dyn:bool}   (partial-static array; dyn=escaped)
// Residual statements (emitted to `out`):  ["assign",lval,e] ["while",c,body] ["if",c,t,e]
//                   ["setidx",a,i,v] ["ret",e]    (lval = ["var",n] | ["idx",a,i])
// Residual exprs reuse the tagged-array form: ["lit",v] ["var",n] ["bin",op,a,b]
//                   ["un",op,a] ["idx",a,i] ["dot",a,k] ["call",f,args] ["arr",els]
const PRIM = (a) => a.k === "s";
const DYN = (e) => ({ k: "d", e });
const STAT = (v) => ({ k: "s", v });

function freshCtr(st) { return st.ctr++; }

// reify an abstract value to a residual expression (materializing heap arrays to literals)
function reify(st, a, out) {
  if (a.k === "s") return ["lit", a.v];
  if (a.k === "d") return a.e;
  if (a.k === "r") return reifyHeap(st, a.a, out);
  throw new Error("reify: bad " + JSON.stringify(a));
}
function reifyHeap(st, addr, out) {
  const o = st.heap.get(addr);
  if (o.residVar) return ["var", o.residVar];        // already materialized to a runtime var
  return ["arr", o.elems.map((e) => reify(st, e, out))];
}

const BINOP = { "+": (a, b) => a + b, "-": (a, b) => a - b, "*": (a, b) => a * b, "/": (a, b) => a / b,
  "%": (a, b) => a % b, "===": (a, b) => a === b, "!==": (a, b) => a !== b, "==": (a, b) => a == b,
  "!=": (a, b) => a != b, "<": (a, b) => a < b, "<=": (a, b) => a <= b, ">": (a, b) => a > b,
  ">=": (a, b) => a >= b, "&": (a, b) => a & b, "|": (a, b) => a | b, "^": (a, b) => a ^ b,
  "<<": (a, b) => a << b, ">>": (a, b) => a >> b };
const UNOP = { "typeof": (a) => typeof a, "!": (a) => !a, "-": (a) => -a };

// ---- expression specializer: returns an abstract value (may emit nothing for pure exprs) ----
function specExpr(n, env, st, out) {
  const t = n[0];
  if (t === "lit") return STAT(n[1]);
  if (t === "var") { if (!(n[1] in env)) throw new Error("unbound " + n[1]); return env[n[1]]; }
  if (t === "bin") {
    const a = specExpr(n[2], env, st, out), b = specExpr(n[3], env, st, out);
    if (PRIM(a) && PRIM(b) && BINOP[n[1]]) return STAT(BINOP[n[1]](a.v, b.v));
    // equality where both operands are KNOWN (incl. heap refs): refs equal iff same addr;
    // a ref is never === a primitive. This lets `typeof it === "number"` style dispatch fold.
    const EQ = { "===": (e) => e, "==": (e) => e, "!==": (e) => !e, "!=": (e) => !e };
    if (EQ[n[1]] && (a.k === "s" || a.k === "r") && (b.k === "s" || b.k === "r")) {
      const eq = a.k === "r" || b.k === "r" ? (a.k === "r" && b.k === "r" && a.a === b.a) : (n[1][0] === "=" ? a.v === b.v : a.v == b.v);
      return STAT(EQ[n[1]](eq));
    }
    return DYN(["bin", n[1], reify(st, a, out), reify(st, b, out)]);
  }
  if (t === "unary" || t === "un") {
    const a = specExpr(n[2], env, st, out);
    if (a.k === "r" && n[1] === "typeof") return STAT("object");
    if (PRIM(a) && UNOP[n[1]]) return STAT(UNOP[n[1]](a.v));
    return DYN([t === "unary" ? "un" : t, n[1], reify(st, a, out)]);
  }
  if (t === "cond") {
    const c = specExpr(n[1], env, st, out);
    if (PRIM(c)) return specExpr(c.v ? n[2] : n[3], env, st, out);
    return DYN(["cond", reify(st, c, out), reify(st, specExpr(n[2], env, st, out), out), reify(st, specExpr(n[3], env, st, out), out)]);
  }
  if (t === "arr") {
    const elems = n[1].map((e) => specExpr(e, env, st, out));
    const addr = st.nextAddr++;
    st.heap.set(addr, { elems, dyn: false, residVar: null });
    return { k: "r", a: addr };
  }
  if (t === "idx") {
    const base = specExpr(n[1], env, st, out), ix = specExpr(n[2], env, st, out);
    if (base.k === "r" && PRIM(ix)) {
      const o = st.heap.get(base.a);
      if (!o.dyn && ix.v >= 0 && ix.v < o.elems.length) return o.elems[ix.v];
    }
    return DYN(["idx", reify(st, base, out), reify(st, ix, out)]);
  }
  if (t === "dot") {
    const base = specExpr(n[1], env, st, out);
    if (base.k === "r" && n[2] === "length") { const o = st.heap.get(base.a); if (!o.dyn) return STAT(o.elems.length); }
    return DYN(["dot", reify(st, base, out), n[2]]);
  }
  if (t === "call") return specCall(n[1][1], n[2], env, st, out);
  throw new Error("specExpr: unhandled " + t);
}

// ---- calls: inline (unfold). The static program drives recursion to termination. ----
function specCall(name, argNodes, env, st, out) {
  const fn = st.funcs[name];
  if (!fn) throw new Error("unknown fn " + name);
  const args = argNodes.map((a) => specExpr(a, env, st, out));
  const nenv = Object.create(null);
  fn[2].forEach((p, i) => (nenv[p] = args[i] !== undefined ? args[i] : STAT(undefined)));
  const r = specStmts(fn[3], nenv, st, out);
  return r.ret !== undefined ? r.ret : STAT(undefined);
}

// assign an abstract value to an lvalue (var or array index), emitting residual on dynamic
function assignLval(lval, val, env, st, out) {
  if (lval[0] === "var") { env[lval[1]] = val; return; }
  if (lval[0] === "idx") {
    const base = specExpr(lval[1], env, st, out), ix = specExpr(lval[2], env, st, out);
    if (base.k === "r" && PRIM(ix)) {
      const o = st.heap.get(base.a);
      if (!o.dyn) { while (o.elems.length <= ix.v) o.elems.push(STAT(0)); o.elems[ix.v] = val; return; }
    }
    out.push(["setidx", reify(st, base, out), reify(st, ix, out), reify(st, val, out)]);
    return;
  }
  throw new Error("assignLval: bad lval " + lval[0]);
}

// ---- statement specializer: mutates env/heap, emits to out, returns {ret?} ----
function specStmts(stmts, env, st, out) {
  for (const s of stmts) {
    const r = specStmt(s, env, st, out);
    if (r && r.ret !== undefined) return r;
  }
  return {};
}
function specStmt(s, env, st, out) {
  const t = s[0];
  if (t === "var") { env[s[1]] = specExpr(s[2], env, st, out); return {}; }
  if (t === "expr") { if (s[1][0] === "assign") assignStmt(s[1], env, st, out); else specExpr(s[1], env, st, out); return {}; }
  if (t === "assign") { assignStmt(s, env, st, out); return {}; }
  if (t === "return") { return { ret: specExpr(s[1], env, st, out) }; }
  if (t === "block") return specStmts(s[1], env, st, out);
  if (t === "if") {
    const c = specExpr(s[1], env, st, out);
    if (PRIM(c)) { if (c.v) return specStmt(s[2], env, st, out); if (s[3]) return specStmt(s[3], env, st, out); return {}; }
    return specIfDyn(c, s[2], s[3], env, st, out);
  }
  if (t === "while") return specWhile(s[1], s[2], env, st, out);
  throw new Error("specStmt: unhandled " + t);
}
function assignStmt(s, env, st, out) {       // ["assign", lval, rhs]  (s could be expr-statement form)
  const lval = s[1], rhs = s[2];
  const val = specExpr(rhs, env, st, out);
  assignLval(lval, val, env, st, out);
}

// vars syntactically ASSIGNED in a node (loop-carried / branch-divergent candidates)
function assignedVars(node, acc) {
  acc = acc || new Set();
  if (!Array.isArray(node)) return acc;
  if (node[0] === "assign" && node[1] && node[1][0] === "var") acc.add(node[1][1]);
  for (const c of node) if (Array.isArray(c)) assignedVars(c, acc);
  return acc;
}

function snapHeap(h) { const m = new Map(); for (const [a, o] of h) m.set(a, { elems: o.elems.slice(), dyn: o.dyn, residVar: o.residVar }); return m; }
function setEnv(env, snap) { for (const k of Object.keys(env)) delete env[k]; Object.assign(env, snap); }
function reifyShallow(a) { if (a.k === "s") return ["lit", a.v]; if (a.k === "d") return a.e; throw new Error("heap-ref divergence in dynamic if (needs deeper join)"); }

// DYNAMIC if. Two cases:
//  (A) neither branch emits residual statements -> the only effects are abstract env/heap
//      changes; MERGE divergent vars/cells via cond(condE, then, else) — a precise phi.
//  (B) a branch emits residual statements (state already dynamic) -> emit `if(c){..}else{..}`;
//      divergent state then lives in runtime arrays (merged by the in-place setidx writes).
function specIfDyn(c, thenS, elseS, env, st, out) {
  const condE = reify(st, c, out);
  const env0 = { ...env }, heap0 = snapHeap(st.heap);
  const thenOut = []; const tr = specStmt(thenS, env, st, thenOut);
  const thenEnv = { ...env }, thenHeap = snapHeap(st.heap);
  setEnv(env, env0); st.heap = snapHeap(heap0);
  const elseOut = []; let er = {}; if (elseS) er = specStmt(elseS, env, st, elseOut);
  const elseEnv = { ...env }, elseHeap = snapHeap(st.heap);
  if ((tr && tr.ret !== undefined) || (er && er.ret !== undefined)) throw new Error("return inside dynamic if not supported");

  if (thenOut.length === 0 && elseOut.length === 0) {          // (A) merge via cond
    setEnv(env, env0);
    for (const k of new Set([...Object.keys(thenEnv), ...Object.keys(elseEnv)])) {
      const ta = thenEnv[k], ea = elseEnv[k];
      env[k] = JSON.stringify(ta) === JSON.stringify(ea) ? ta : DYN(["cond", condE, reifyShallow(ta), reifyShallow(ea)]);
    }
    const merged = new Map();
    for (const [addr, to] of thenHeap) {
      const eo = elseHeap.get(addr) || to;
      const elems = to.elems.map((tc, i) => {
        const ec = eo.elems[i] !== undefined ? eo.elems[i] : tc;
        return JSON.stringify(tc) === JSON.stringify(ec) ? tc : DYN(["cond", condE, reifyShallow(tc), reifyShallow(ec)]);
      });
      merged.set(addr, { elems, dyn: to.dyn, residVar: to.residVar });
    }
    st.heap = merged;
    return {};
  }
  // (B) residual statements: reconcile env (refs same addr ok; diverged scalars unsupported)
  setEnv(env, env0);
  for (const k of Object.keys(thenEnv)) {
    const a = thenEnv[k], b = elseEnv[k];
    if (a && b && a.k === "r" && b.k === "r" && a.a === b.a) { env[k] = a; }
    else if (JSON.stringify(a) === JSON.stringify(b)) { env[k] = a; }
    else throw new Error("dynamic-if (B) diverges scalar `" + k + "`");
  }
  out.push(["if", condE, thenOut, elseOut]);
  return {};
}

// ---- while: static condition unrolls; dynamic condition residualizes a loop ----
const UNROLL_CAP = 2000000;
function specWhile(condN, bodyN, env, st, out) {
  // peek: is the condition static right now?
  let c = specExpr(condN, env, st, out);
  if (PRIM(c)) {                              // STATIC loop -> unroll
    let guard = 0;
    while (c.v) {
      const r = specStmt(bodyN, env, st, out);
      if (r && r.ret !== undefined) return r;
      c = specExpr(condN, env, st, out);
      if (PRIM(c) === false) break;           // became dynamic mid-loop -> fall through to residualize
      if (++guard > UNROLL_CAP) throw new Error("static while exceeded unroll cap");
    }
    if (PRIM(c)) return {};
  }
  return specWhileDyn(condN, bodyN, env, st, out);  // DYNAMIC loop
}
// DYNAMIC while: generalize the loop-carried (assigned) vars to fresh dynamic runtime
// vars, residualize the body once over them, and emit a residual `while`.
function specWhileDyn(condN, bodyN, env, st, out) {
  const mvars = [...assignedVars(bodyN)].filter((v) => v in env);
  const loopVar = {};
  for (const mv of mvars) {                       // wX = <current value>;  env[mv] := dynamic wX
    const w = "w" + (st.ctr++);
    loopVar[mv] = w;
    out.push(["assign", ["var", w], reify(st, env[mv], out)]);
    env[mv] = DYN(["var", w]);
  }
  const condE = reify(st, specExpr(condN, env, st, out), out);   // condition over the loop vars
  const bodyOut = [];
  const r = specStmt(bodyN, env, st, bodyOut);
  if (r && r.ret !== undefined) throw new Error("return inside dynamic while not supported");
  for (const mv of mvars) {                       // loop-carried update: wX = <new value>
    const cur = env[mv];
    if (!(cur.k === "d" && cur.e[0] === "var" && cur.e[1] === loopVar[mv])) {
      bodyOut.push(["assign", ["var", loopVar[mv]], reify(st, cur, bodyOut)]);
    }
    env[mv] = DYN(["var", loopVar[mv]]);          // dynamic after the loop too
  }
  out.push(["while", condE, bodyOut]);
  return {};
}

// deep-load a concrete JS value into the abstract heap as STATIC structure
function loadStatic(st, v) {
  if (Array.isArray(v)) {
    const elems = v.map((x) => loadStatic(st, x));
    const addr = st.nextAddr++;
    st.heap.set(addr, { elems, dyn: false, residVar: null });
    return { k: "r", a: addr };
  }
  return STAT(v);
}

// ---- top-level. argSpecs[i] = {s: jsValue} (static) | {d: name} (dynamic -> Var name) ----
function specialize(funcs, entry, argSpecs) {
  const st = { funcs: {}, heap: new Map(), nextAddr: 0, ctr: 0 };
  for (const f of funcs) st.funcs[f[1]] = f;
  const fn = st.funcs[entry];
  const env = Object.create(null);
  fn[2].forEach((p, i) => {
    const spec = argSpecs[i] || { s: undefined };
    env[p] = "d" in spec ? DYN(["var", spec.d]) : loadStatic(st, spec.s);
  });
  const out = [];
  const r = specStmts(fn[3], env, st, out);
  return { stmts: out, ret: r.ret ? reify(st, r.ret, out) : ["lit", undefined] };
}

module.exports = { specialize, loadStatic, STAT, DYN };
