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
  throw new Error("renderExpr: unhandled " + t + " in " + JSON.stringify(ast));
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

module.exports = { recursiveSet, collapse, renderExpr, renderProg, compileResidual, toSingleExpr };
