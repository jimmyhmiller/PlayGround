// backend.js — Node-side back-end for mix's residual programs (["prog", entry, fns]).
// Renders to JS source and inlines acyclic residual functions. The inliner is what
// collapses static-recursion (an acyclic residual call graph) back to a single
// expression, while leaving genuine dynamic recursion (self-cyclic fns = RFG) intact.

// ---- which residual fns are part of a call cycle (must stay as functions) ----
function recursiveSet(fns) {
  const byName = new Map(fns.map((f) => [f[0], f]));
  const callees = (body) => {
    const out = new Set();
    const walk = (n) => {
      if (!Array.isArray(n)) return;
      if (n[0] === "call") { out.add(n[1]); n[2].forEach(walk); return; }
      n.forEach(walk);
    };
    walk(body);
    return out;
  };
  const adj = new Map(fns.map((f) => [f[0], callees(f[2])]));
  const inCycle = new Set();
  for (const start of adj.keys()) {
    // can `start` reach itself?
    const seen = new Set();
    const stack = [...(adj.get(start) || [])];
    while (stack.length) {
      const n = stack.pop();
      if (n === start) { inCycle.add(start); break; }
      if (seen.has(n)) continue;
      seen.add(n);
      for (const m of adj.get(n) || []) stack.push(m);
    }
  }
  return inCycle;
}

// substitute params->arg ASTs in a (closed-except-params) residual body
function subst(ast, map) {
  if (!Array.isArray(ast)) return ast;
  if (ast[0] === "var" && map.has(ast[1])) return map.get(ast[1]);
  return ast.map((x) => (Array.isArray(x) ? subst(x, map) : x));
}

// inline every call to a NON-recursive fn, everywhere, until none remain.
function collapse(result) {
  let [, entry, fns] = result;
  const rec = recursiveSet(fns);
  const def = new Map(fns.map((f) => [f[0], f]));
  const inlineNonRec = (ast, guard) => {
    if (!Array.isArray(ast)) return ast;
    if (ast[0] === "call") {
      const name = ast[1];
      const args = ast[2].map((a) => inlineNonRec(a, guard));
      if (!rec.has(name) && def.has(name)) {
        const [, params, body] = def.get(name);
        const map = new Map(params.map((p, i) => [p, args[i]]));
        return inlineNonRec(subst(body, map), guard); // body of a non-rec fn -> inline through
      }
      return ["call", name, args];
    }
    return ast.map((x) => inlineNonRec(x, guard));
  };
  entry = inlineNonRec(entry, null);
  const keptFns = fns.filter((f) => rec.has(f[0])).map((f) => [f[0], f[1], inlineNonRec(f[2], null)]);
  return ["prog", entry, keptFns];
}

// ---- simplify: constant-fold + inline-on-constant-data ----
// The generated (trivial-projection) compiler emits a target that re-dispatches on the
// now-CONSTANT source baked in (`src[0] === "num" ? ...`) and calls the residual
// interpreter `interp_0` on constant sub-sources. Constant-folding that dispatch and
// inlining those calls (the source structure is constant, so recursion bottoms out)
// recovers the clean, OPTIMIZED target — post-specialization simplification.
const RTAGS = new Set(["lit", "var", "bin", "cond", "idx", "dot", "get", "call", "arr", "unary", "new", "opaque", "index"]);
const liftData = (v) => (Array.isArray(v) ? v : ["lit", v]); // data array stays raw; primitive -> lit
function constOf(n) {
  if (Array.isArray(n)) return n[0] === "lit" ? { ok: true, v: n[1] } : { ok: false };
  return { ok: true, v: n }; // raw primitive
}
const BINOP = {
  "+": (a, b) => a + b, "-": (a, b) => a - b, "*": (a, b) => a * b, "/": (a, b) => a / b,
  "%": (a, b) => a % b, "===": (a, b) => a === b, "!==": (a, b) => a !== b, "==": (a, b) => a == b,
  "!=": (a, b) => a != b, "<": (a, b) => a < b, "<=": (a, b) => a <= b, ">": (a, b) => a > b,
  ">=": (a, b) => a >= b, "&": (a, b) => a & b, "|": (a, b) => a | b, "^": (a, b) => a ^ b,
  "<<": (a, b) => a << b, ">>": (a, b) => a >> b, ">>>": (a, b) => a >>> b,
};
const UNOP = { "typeof": (a) => typeof a, "!": (a) => !a, "-": (a) => -a };
const isData = (a) => Array.isArray(a) && !RTAGS.has(a[0]);         // a baked source-data node
const isConstArg = (a) => isData(a) || (Array.isArray(a) && a[0] === "lit");
// simplify = a memoizing online specializer over the residual IR. Each call with a
// constant argument is residualized into a function keyed by its constant-arg SKELETON
// (dynamic args become fresh params). A SHRINKING skeleton (structural recursion over a
// now-constant source) produces finitely many functions that collapse() inlines away; a
// REPEATING skeleton (a data-dependent loop, e.g. Brainfuck `[...]`) maps to one
// self-recursive function that stays — a clean residual loop. This is mix's own strategy,
// re-applied to clean up the trivial-projection compiler's output.
function simplify(result) {
  const defs = {};
  for (const f of result[2]) defs[f[0]] = f;
  const origRec = recursiveSet(result[2]);  // which original fns are recursive (loops)
  const memo = new Map();   // skeletonKey -> residual fn name
  const gen = [];           // generated residual fns [name, params, body]
  let ctr = 0;
  const skel = (name, args) => {
    let k = name;
    for (const a of args) k += isData(a) ? "|D" + JSON.stringify(a)
      : (Array.isArray(a) && a[0] === "lit") ? "|L" + JSON.stringify(a[1]) : "|d";
    return k;
  };
  function residualize(name, args) {
    const fn = defs[name];
    const key = skel(name, args);
    let rn = memo.get(key);
    if (rn === undefined) {
      rn = "r" + (ctr++);
      memo.set(key, rn);                                   // register BEFORE body (ties loops)
      const map = new Map(), params = [];
      fn[1].forEach((p, k) => {
        const a = args[k];
        if (isConstArg(a)) map.set(p, a);
        else { const fp = "q" + k + "_" + rn; map.set(p, ["var", fp]); params.push(fp); }
      });
      gen.push([rn, params, spec(subst(fn[2], map))]);
    }
    return ["call", rn, args.filter((a) => !isConstArg(a))]; // residual call passes only dynamic args
  }
  function spec(n) {
    if (!Array.isArray(n)) return n;
    const t = n[0];
    if (!RTAGS.has(t)) return n;                            // raw data
    if (t === "lit" || t === "var") return n;
    if (t === "bin") { const a = spec(n[2]), b = spec(n[3]), ca = constOf(a), cb = constOf(b);
      if (ca.ok && cb.ok && BINOP[n[1]]) return ["lit", BINOP[n[1]](ca.v, cb.v)]; return ["bin", n[1], a, b]; }
    if (t === "unary") { const a = spec(n[2]), ca = constOf(a);
      if (ca.ok && UNOP[n[1]]) return ["lit", UNOP[n[1]](ca.v)]; return ["unary", n[1], a]; }
    if (t === "cond") { const c = spec(n[1]), cc = constOf(c);
      if (cc.ok) return spec(cc.v ? n[2] : n[3]); return ["cond", c, spec(n[2]), spec(n[3])]; }
    if (t === "idx") { const a = spec(n[1]), i = spec(n[2]), ci = constOf(i);
      if (ci.ok) { if (isData(a)) return liftData(a[ci.v]); if (a[0] === "arr") return spec(a[1][ci.v]); }
      return ["idx", a, i]; }
    if (t === "dot" || t === "get") { const a = spec(n[1]);
      if (isData(a) && n[2] === "length") return ["lit", a.length]; return [t, a, n[2]]; }
    if (t === "arr") return ["arr", n[1].map(spec)];
    if (t === "call") {
      const name = n[1], args = n[2].map(spec);
      if (!defs[name]) return ["call", name, args];
      if (args.some(isConstArg)) return residualize(name, args);     // a constant arg -> specialize/loop-tie
      if (!origRec.has(name)) {                                       // non-recursive, all-dynamic -> inline + fold
        const map = new Map(defs[name][1].map((p, k) => [p, args[k]]));
        return spec(subst(defs[name][2], map));
      }
      return ["call", name, args];                                   // recursive + all-dynamic (rare): keep
    }
    return n;
  }
  const entry = spec(result[1]);
  return collapse(["prog", entry, gen]);                   // inline the acyclic residual fns
}

// ---- rendering ----
function renderExpr(ast) {
  const t = ast[0];
  if (t === "lit") return typeof ast[1] === "string" ? JSON.stringify(ast[1]) : String(ast[1]);
  if (t === "var") return ast[1];
  if (t === "bin") return "(" + renderExpr(ast[2]) + " " + ast[1] + " " + renderExpr(ast[3]) + ")";
  if (t === "cond") return "(" + renderExpr(ast[1]) + " ? " + renderExpr(ast[2]) + " : " + renderExpr(ast[3]) + ")";
  if (t === "idx") return renderExpr(ast[1]) + "[" + renderExpr(ast[2]) + "]";
  if (t === "dot") return renderExpr(ast[1]) + "." + ast[2];
  if (t === "arr") return "[" + ast[1].map(renderExpr).join(", ") + "]";
  if (t === "call") return ast[1] + "(" + ast[2].map(renderExpr).join(", ") + ")";
  // fallback: a raw data value baked in from the (now-constant) source -> a JS literal
  return JSON.stringify(ast);
}
function renderProg(result) {
  const [, entry, fns] = result;
  let out = "";
  for (const [name, params, body] of fns) {
    out += `function ${name}(${params.join(", ")}) { return ${renderExpr(body)}; }\n`;
  }
  return { entry: renderExpr(entry), defs: out };
}

// collapse + render; for a fully-acyclic program `defs` is empty and entry is one expr.
function compileResidual(result) {
  return renderProg(collapse(result));
}
// collapse and demand a single expression (no residual fns left); else throw.
function toSingleExpr(result) {
  const c = collapse(result);
  if (c[2].length) throw new Error("residual functions remain: " + c[2].map((f) => f[0]).join(", "));
  return renderExpr(c[1]);
}

module.exports = { recursiveSet, collapse, simplify, renderExpr, renderProg, compileResidual, toSingleExpr };
