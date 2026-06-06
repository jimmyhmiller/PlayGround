// mix_imp.subset.js — the imperative online partial evaluator, IN jspe's subset (so jspe
// can self-apply it: P2). Port of mix_imp.js. No Maps/spread/JSON/closures: env is an
// array of [name,val] pairs; heap is an array of ["cell",elems,dyn,residVar]; st carries
// [funcs, heap, nextAddr, ctr] and is mutated by index.
//
// value: ["s",v] static | ["d",e] dynamic(expr) | ["r",a] heap ref
// st:    ["st", funcs, heap, nextAddr, ctr]

// ---- structural equality on abstract values / residual exprs (replaces JSON.stringify) ----
function eqv(a, b) {
  if (typeof a === "number") { return a === b; }
  if (typeof a === "string") { return a === b; }
  if (typeof a === "boolean") { return a === b; }
  if (a === null) { return b === null; }
  // both arrays
  if (b === null) { return false; }
  if (typeof b !== "object") { return false; }
  if (a.length !== b.length) { return false; }
  var i = 0;
  while (i < a.length) { if (eqv(a[i], b[i]) === false) { return false; } i = i + 1; }
  return true;
}
function arrCopy(a) { var o = []; var i = 0; while (i < a.length) { o[i] = a[i]; i = i + 1; } return o; }

// ---- env: a BOX ["e", pairs] (so it can be restored by swapping the inner array, no
// in-place shrink). Snapshots (envCopy) are plain pairs arrays. ----
function envNew() { return ["e", []]; }
function envHas(env, name) { var p = env[1]; var i = 0; while (i < p.length) { if (p[i][0] === name) { return true; } i = i + 1; } return false; }
function envGet(env, name) { var p = env[1]; var i = 0; while (i < p.length) { if (p[i][0] === name) { return p[i][1]; } i = i + 1; } return ["d", ["__unbound__", name]]; }
function envSet(env, name, val) { var p = env[1]; var i = 0; while (i < p.length) { if (p[i][0] === name) { p[i][1] = val; return env; } i = i + 1; } p[p.length] = [name, val]; return env; }
function envCopy(env) { var p = env[1]; var o = []; var i = 0; while (i < p.length) { o[i] = [p[i][0], p[i][1]]; i = i + 1; } return o; }
function envRestore(env, snap) { var o = []; var i = 0; while (i < snap.length) { o[i] = [snap[i][0], snap[i][1]]; i = i + 1; } env[1] = o; }

// ---- st accessors ----
function stFuncs(st) { return st[1]; }
function stHeap(st) { return st[2]; }
function stSetHeap(st, h) { st[2] = h; }
function stNextAddr(st) { var a = st[3]; st[3] = a + 1; return a; }
function stFresh(st) { var c = st[4]; st[4] = c + 1; return "w" + c; }
function findFun(funcs, name) { var i = 0; while (i < funcs.length) { if (funcs[i][1] === name) { return funcs[i]; } i = i + 1; } return ["__missing__", name]; }

// heap cell = ["cell", elems, dyn, residVar]
function heapCopy(h) {
  var o = []; var i = 0;
  while (i < h.length) { var c = h[i]; o[i] = ["cell", arrCopy(c[1]), c[2], c[3]]; i = i + 1; }
  return o;
}

// ---- reify abstract value -> residual expr ----
function reify(st, a) {
  if (a[0] === "s") { return ["lit", a[1]]; }
  if (a[0] === "d") { return a[1]; }
  return reifyHeap(st, a[1]);
}
function reifyHeap(st, addr) {
  var c = stHeap(st)[addr];
  if (c[3] !== null) { return ["var", c[3]]; }
  var els = []; var elems = c[1]; var i = 0;
  while (i < elems.length) { els[i] = reify(st, elems[i]); i = i + 1; }
  return ["arr", els];
}
function reifyShallow(a) {
  if (a[0] === "s") { return ["lit", a[1]]; }
  if (a[0] === "d") { return a[1]; }
  return ["__ref_divergence__"];
}

// ---- operators ----
function binop(op, a, b) {
  if (op === "+") { return a + b; } if (op === "-") { return a - b; } if (op === "*") { return a * b; }
  if (op === "/") { return a / b; } if (op === "%") { return a % b; }
  if (op === "<") { return a < b; } if (op === ">") { return a > b; }
  if (op === "<=") { return a <= b; } if (op === ">=") { return a >= b; }
  if (op === "===") { return a === b; } if (op === "!==") { return a !== b; }
  if (op === "==") { return a == b; } if (op === "!=") { return a != b; }
  return ["__binop__", op];
}
function isBinop(op) {
  var ops = ["+", "-", "*", "/", "%", "<", ">", "<=", ">=", "===", "!==", "==", "!="];
  var i = 0; while (i < ops.length) { if (ops[i] === op) { return true; } i = i + 1; } return false;
}
function isEq(op) {
  if (op === "===") { return true; } if (op === "!==") { return true; }
  if (op === "==") { return true; } if (op === "!=") { return true; } return false;
}
function eqResult(op, e) { if (op === "===") { return e; } if (op === "==") { return e; } return !e; }

// ---- expression specializer ----
function specExpr(n, env, st, out) {
  var t = n[0];
  if (t === "lit") { return ["s", n[1]]; }
  if (t === "var") { return envGet(env, n[1]); }
  if (t === "bin") {
    var a = specExpr(n[2], env, st, out);
    var b = specExpr(n[3], env, st, out);
    if (a[0] === "s") { if (b[0] === "s") { if (isBinop(n[1])) { return ["s", binop(n[1], a[1], b[1])]; } } }
    if (isEq(n[1])) {
      var ak = a[0]; var bk = b[0];
      if (ak !== "d") { if (bk !== "d") {
        var eq = false;
        if (ak === "r") { if (bk === "r") { eq = a[1] === b[1]; } else { eq = false; } }
        else { if (bk === "r") { eq = false; } else { eq = a[1] === b[1]; } }
        return ["s", eqResult(n[1], eq)];
      } }
    }
    return ["d", ["bin", n[1], reify(st, a), reify(st, b)]];
  }
  if (t === "unary") {
    var ua = specExpr(n[2], env, st, out);
    if (ua[0] === "r") { if (n[1] === "typeof") { return ["s", "object"]; } }
    if (ua[0] === "s") {
      if (n[1] === "typeof") { return ["s", typeof ua[1]]; }
      if (n[1] === "!") { return ["s", !ua[1]]; }
      if (n[1] === "-") { return ["s", 0 - ua[1]]; }
    }
    return ["d", ["un", n[1], reify(st, ua)]];
  }
  if (t === "cond") {
    var c = specExpr(n[1], env, st, out);
    if (c[0] === "s") { if (c[1]) { return specExpr(n[2], env, st, out); } return specExpr(n[3], env, st, out); }
    return ["d", ["cond", reify(st, c), reify(st, specExpr(n[2], env, st, out)), reify(st, specExpr(n[3], env, st, out))]];
  }
  if (t === "arr") {
    var elems = []; var els = n[1]; var i = 0;
    while (i < els.length) { elems[i] = specExpr(els[i], env, st, out); i = i + 1; }
    var addr = stNextAddr(st);
    stHeap(st)[addr] = ["cell", elems, false, null];
    return ["r", addr];
  }
  if (t === "idx") {
    var base = specExpr(n[1], env, st, out);
    var ix = specExpr(n[2], env, st, out);
    if (base[0] === "r") { if (ix[0] === "s") {
      var oc = stHeap(st)[base[1]];
      if (oc[2] === false) { if (ix[1] >= 0) { if (ix[1] < oc[1].length) { return oc[1][ix[1]]; } } }
    } }
    return ["d", ["idx", reify(st, base), reify(st, ix)]];
  }
  if (t === "dot") {
    var db = specExpr(n[1], env, st, out);
    if (db[0] === "r") { if (n[2] === "length") { var dc = stHeap(st)[db[1]]; if (dc[2] === false) { return ["s", dc[1].length]; } } }
    return ["d", ["dot", reify(st, db), n[2]]];
  }
  if (t === "call") { return specCall(n[1][1], n[2], env, st, out); }
  return ["d", ["__unhandled_expr__", t]];
}

function specCall(name, argNodes, env, st, out) {
  var fn = findFun(stFuncs(st), name);
  var args = []; var i = 0;
  while (i < argNodes.length) { args[i] = specExpr(argNodes[i], env, st, out); i = i + 1; }
  var nenv = envNew(); var params = fn[2]; var j = 0;
  while (j < params.length) { nenv = envSet(nenv, params[j], args[j]); j = j + 1; }
  var r = specStmts(fn[3], nenv, st, out);
  if (r[0] === "ret") { return r[1]; }
  return ["s", 0];
}

// ---- assignment ----
function assignLval(lval, val, env, st, out) {
  if (lval[0] === "var") { envSet(env, lval[1], val); return val; }
  var base = specExpr(lval[1], env, st, out);
  var ix = specExpr(lval[2], env, st, out);
  var done = false;
  if (base[0] === "r") { if (ix[0] === "s") {
    var oc = stHeap(st)[base[1]];
    if (oc[2] === false) { while (oc[1].length <= ix[1]) { oc[1][oc[1].length] = ["s", 0]; } oc[1][ix[1]] = val; done = true; }
  } }
  if (done === false) { out[out.length] = ["setidx", reify(st, base), reify(st, ix), reify(st, val)]; }
  return val;
}

// ---- statements: returns ["ret", val] or ["next"] ----
function specStmts(stmts, env, st, out) {
  var i = 0;
  while (i < stmts.length) {
    var r = specStmt(stmts[i], env, st, out);
    if (r[0] === "ret") { return r; }
    i = i + 1;
  }
  return ["next"];
}
function specStmt(s, env, st, out) {
  var t = s[0];
  if (t === "var") { envSet(env, s[1], specExpr(s[2], env, st, out)); return ["next"]; }
  if (t === "expr") { if (s[1][0] === "assign") { assignLval(s[1][1], specExpr(s[1][2], env, st, out), env, st, out); } else { specExpr(s[1], env, st, out); } return ["next"]; }
  if (t === "assign") { assignLval(s[1], specExpr(s[2], env, st, out), env, st, out); return ["next"]; }
  if (t === "return") { return ["ret", specExpr(s[1], env, st, out)]; }
  if (t === "block") { return specStmts(s[1], env, st, out); }
  if (t === "if") {
    var c = specExpr(s[1], env, st, out);
    if (c[0] === "s") { if (c[1]) { return specStmt(s[2], env, st, out); } if (s[3] !== null) { return specStmt(s[3], env, st, out); } return ["next"]; }
    return specIfDyn(c, s[2], s[3], env, st, out);
  }
  if (t === "while") { return specWhile(s[1], s[2], env, st, out); }
  return ["next"];
}

// ---- assigned-var collection ----
function assignedVars(node, acc) {
  if (typeof node !== "object") { return acc; }
  if (node === null) { return acc; }
  if (node[0] === "assign") { if (node[1] !== null) { if (node[1][0] === "var") { acc[acc.length] = node[1][1]; } } }
  var i = 0;
  while (i < node.length) { if (typeof node[i] === "object") { if (node[i] !== null) { assignedVars(node[i], acc); } } i = i + 1; }
  return acc;
}

// ---- dynamic if (cond-phi merge if no residual stmts; else emit residual if) ----
function specIfDyn(c, thenS, elseS, env, st, out) {
  var condE = reify(st, c);
  var env0 = envCopy(env);
  var heap0 = heapCopy(stHeap(st));
  var thenOut = []; var tr = specStmt(thenS, env, st, thenOut);
  var thenEnv = envCopy(env); var thenHeap = heapCopy(stHeap(st));
  envRestore(env, env0); stSetHeap(st, heapCopy(heap0));
  var elseOut = []; var er = ["next"]; if (elseS !== null) { er = specStmt(elseS, env, st, elseOut); }
  var elseEnv = envCopy(env); var elseHeap = heapCopy(stHeap(st));
  if (tr[0] === "ret") { return ["ret", ["d", ["__ret_in_dyn_if__"]]]; }
  if (er[0] === "ret") { return ["ret", ["d", ["__ret_in_dyn_if__"]]]; }

  if (thenOut.length === 0) { if (elseOut.length === 0) {
    // (A) merge via cond
    envRestore(env, env0);
    var k = 0;
    while (k < thenEnv.length) {
      var nm = thenEnv[k][0];
      var ta = thenEnv[k][1];
      var ea = lookupPair(elseEnv, nm, ta);
      if (eqv(ta, ea)) { envSet(env, nm, ta); } else { envSet(env, nm, ["d", ["cond", condE, reifyShallow(ta), reifyShallow(ea)]]); }
      k = k + 1;
    }
    var merged = []; var hi = 0;
    while (hi < thenHeap.length) {
      var tc = thenHeap[hi];
      var ec = hi < elseHeap.length ? elseHeap[hi] : tc;
      var mel = []; var ci = 0;
      while (ci < tc[1].length) {
        var tcell = tc[1][ci];
        var ecell = ci < ec[1].length ? ec[1][ci] : tcell;
        if (eqv(tcell, ecell)) { mel[ci] = tcell; } else { mel[ci] = ["d", ["cond", condE, reifyShallow(tcell), reifyShallow(ecell)]]; }
        ci = ci + 1;
      }
      merged[hi] = ["cell", mel, tc[2], tc[3]];
      hi = hi + 1;
    }
    stSetHeap(st, merged);
    return ["next"];
  } }
  // (B) residual statements: env refs same -> ok; emit if
  envRestore(env, env0);
  out[out.length] = ["if", condE, thenOut, elseOut];
  return ["next"];
}
function lookupPair(env, name, dflt) { var i = 0; while (i < env.length) { if (env[i][0] === name) { return env[i][1]; } i = i + 1; } return dflt; }

// ---- while ----
function specWhile(condN, bodyN, env, st, out) {
  var c = specExpr(condN, env, st, out);
  if (c[0] === "s") {
    var guard = 0;
    while (c[1]) {
      var r = specStmt(bodyN, env, st, out);
      if (r[0] === "ret") { return r; }
      c = specExpr(condN, env, st, out);
      if (c[0] !== "s") { return specWhileDynResume(condN, bodyN, env, st, out); }
      guard = guard + 1;
      if (guard > 2000000) { return ["ret", ["d", ["__unroll_overflow__"]]]; }
    }
    return ["next"];
  }
  return specWhileDynResume(condN, bodyN, env, st, out);
}
function specWhileDynResume(condN, bodyN, env, st, out) {
  var mvars = [];
  assignedVars(bodyN, mvars);
  var loopVars = [];   // [name, freshvar]
  var i = 0;
  while (i < mvars.length) {
    var mv = mvars[i];
    if (envHas(env, mv)) {
      var fresh = stFresh(st);
      loopVars[loopVars.length] = [mv, fresh];
      out[out.length] = ["assign", ["var", fresh], reify(st, envGet(env, mv))];
      envSet(env, mv, ["d", ["var", fresh]]);
    }
    i = i + 1;
  }
  var condE = reify(st, specExpr(condN, env, st, out));
  var bodyOut = [];
  var r = specStmt(bodyN, env, st, bodyOut);
  if (r[0] === "ret") { return ["ret", ["d", ["__ret_in_dyn_while__"]]]; }
  var j = 0;
  while (j < loopVars.length) {
    var name = loopVars[j][0];
    var lv = loopVars[j][1];
    var cur = envGet(env, name);
    if (eqv(cur, ["d", ["var", lv]]) === false) { bodyOut[bodyOut.length] = ["assign", ["var", lv], reify(st, cur)]; }
    envSet(env, name, ["d", ["var", lv]]);
    j = j + 1;
  }
  out[out.length] = ["while", condE, bodyOut];
  return ["next"];
}

// ---- deep-load a JS value into the heap as static structure ----
function loadStatic(st, v) {
  if (typeof v === "object") { if (v !== null) {
    var elems = []; var i = 0;
    while (i < v.length) { elems[i] = loadStatic(st, v[i]); i = i + 1; }
    var addr = stNextAddr(st);
    stHeap(st)[addr] = ["cell", elems, false, null];
    return ["r", addr];
  } }
  return ["s", v];
}

// ---- top-level. argSpecs[i] = ["s", jsValue] | ["d", name] ----
function specialize(funcs, entryName, argSpecs) {
  var st = ["st", funcs, [], 0, 0];
  var fn = findFun(funcs, entryName);
  var env = envNew();
  var params = fn[2]; var i = 0;
  while (i < params.length) {
    var spec = argSpecs[i];
    if (spec[0] === "d") { envSet(env, params[i], ["d", ["var", spec[1]]]); }
    else { envSet(env, params[i], loadStatic(st, spec[1])); }
    i = i + 1;
  }
  var out = [];
  var r = specStmts(fn[3], env, st, out);
  var ret = ["lit", 0];
  if (r[0] === "ret") { ret = reify(st, r[1]); }
  return ["prog", out, ret];
}

if (typeof module !== "undefined") {
  module.exports = { specialize: specialize, specExpr: specExpr, specStmts: specStmts };
}
