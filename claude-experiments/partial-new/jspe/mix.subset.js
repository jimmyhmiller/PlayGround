// mix.subset.js — an ONLINE partial evaluator WRITTEN IN the jspe subset.
//
// This is the self-application artifact. It must stay inside the exact language jspe
// accepts (arith, comparison, arrays, index, .length, if/while, var/assign, named
// non-closure functions, array literals — NO closures, NO Map, NO .push, NO throw),
// so that jspe can later partially-evaluate THIS file: jspe(mix, int) = compiler (P2).
//
// It is an AST-walking specializer (the classic self-applicable shape), not a port of
// the bytecode engine. The AST it consumes is exactly parse.js's tagged-array AST.
//
//   value  ::= ["s", v]        a STATIC value (number/string/bool/array)
//            | ["d", ast]      a DYNAMIC value, described by a residual AST node
//   env    ::= ["nil"] | ["cons", name, value, env]      (linked list — no .push needed)
//   prog   ::= [ ["fundecl", name, params, body], ... ]  (the program being specialized)
//
// Residual ASTs reuse parse.js node shapes: ["lit",v] ["var",n] ["bin",op,a,b]
// ["cond",c,t,e] ["idx",a,b]. A printer (Node side) renders them to JS source.

// ---- environment (linked list) ----
function envNil() { return ["nil"]; }
function envExtend(env, name, val) { return ["cons", name, val, env]; }
function envLookup(env, name) {
  var e = env;
  while (e[0] === "cons") {
    if (e[1] === name) { return e[2]; }
    e = e[3];
  }
  // unbound: not expected for well-formed programs; surface it as a poison dynamic var
  return ["d", ["var", name]];
}

// ---- value helpers ----
function isStatic(v) { return v[0] === "s"; }
// reify: turn a value back into a residual AST node (static -> a literal).
function reify(v) {
  if (v[0] === "s") { return ["lit", v[1]]; }
  return v[1];
}

// ---- static primitive operators (the dispatch that FOLDS when op is static) ----
function applyOp(op, a, b) {
  if (op === "+") { return a + b; }
  if (op === "-") { return a - b; }
  if (op === "*") { return a * b; }
  if (op === "===") { return a === b; }
  if (op === "!==") { return a !== b; }
  if (op === "==") { return a == b; }
  if (op === "!=") { return a != b; }
  if (op === "<") { return a < b; }
  if (op === ">") { return a > b; }
  if (op === "<=") { return a <= b; }
  if (op === ">=") { return a >= b; }
  // unknown operator: keep it visible rather than silently wrong
  return ["__unknown_op__", op];
}

// ---- find a function definition by name ----
function findFun(prog, name) {
  var i = 0;
  while (i < prog.length) {
    if (prog[i][1] === name) { return prog[i]; }
    i = i + 1;
  }
  return ["__missing_fun__", name];
}

// ---- bind a parameter list to evaluated arguments ----
function bindArgs(params, args, callerEnv, prog) {
  var env = envNil();
  var i = 0;
  while (i < params.length) {
    env = envExtend(env, params[i], peval(args[i], callerEnv, prog));
    i = i + 1;
  }
  return env;
}

// ---- expression specializer ----
function peval(expr, env, prog) {
  var tag = expr[0];
  if (tag === "lit") { return ["s", expr[1]]; }
  if (tag === "var") { return envLookup(env, expr[1]); }
  if (tag === "arr") {
    // array literal: static iff every element is static, else residual array node
    var els = expr[1];
    var allStatic = true;
    var i = 0;
    while (i < els.length) {
      var ve = peval(els[i], env, prog);
      if (ve[0] !== "s") { allStatic = false; }
      i = i + 1;
    }
    // rebuild (second pass keeps it simple in-subset)
    var vals = [];
    var resid = [];
    i = 0;
    while (i < els.length) {
      var v2 = peval(els[i], env, prog);
      vals = arrPush(vals, reifyStaticOrVal(v2));
      resid = arrPush(resid, reify(v2));
      i = i + 1;
    }
    if (allStatic) { return ["s", vals]; }
    return ["d", ["arr", resid]];
  }
  if (tag === "bin") {
    var a = peval(expr[2], env, prog);
    var b = peval(expr[3], env, prog);
    if (a[0] === "s") {
      if (b[0] === "s") { return ["s", applyOp(expr[1], a[1], b[1])]; }
    }
    return ["d", ["bin", expr[1], reify(a), reify(b)]];
  }
  if (tag === "cond") {
    var c = peval(expr[1], env, prog);
    if (c[0] === "s") {
      if (c[1]) { return peval(expr[2], env, prog); }
      return peval(expr[3], env, prog);
    }
    return ["d", ["cond", reify(c), reify(peval(expr[2], env, prog)), reify(peval(expr[3], env, prog))]];
  }
  if (tag === "idx") {
    var base = peval(expr[1], env, prog);
    var ix = peval(expr[2], env, prog);
    if (base[0] === "s") {
      if (ix[0] === "s") { return liftElem(base[1][ix[1]]); }
    }
    return ["d", ["idx", reify(base), reify(ix)]];
  }
  if (tag === "dot") {
    var d = peval(expr[1], env, prog);
    if (d[0] === "s") {
      if (expr[2] === "length") { return ["s", d[1].length]; }
    }
    return ["d", ["dot", reify(d), expr[2]]];
  }
  if (tag === "call") {
    var fname = expr[1][1];           // callee is ["var", name]
    var fn = findFun(prog, fname);
    var nenv = bindArgs(fn[2], expr[2], env, prog);
    var r = pevalStmts(fn[3], nenv, prog);
    if (r[0] === "ret") { return r[1]; }
    return ["s", 0];                  // fell off the end -> undefined-ish (subset has no undefined literal)
  }
  return ["d", ["__unhandled_expr__", tag]];
}

// a raw heap element of a static array is already a concrete value; wrap it static.
function liftElem(x) { return ["s", x]; }
// when (re)building a static array we need the concrete value back out of a value-cell.
function reifyStaticOrVal(v) { if (v[0] === "s") { return v[1]; } return v; }
// array append without .push (subset has no push): build a fresh array via index copy.
function arrPush(a, x) {
  var out = [];
  var i = 0;
  while (i < a.length) { out[i] = a[i]; i = i + 1; }
  out[a.length] = x;
  return out;
}

// ---- statement specializer: returns ["ret", value] or ["next"] ----
function pevalStmts(stmts, env, prog) {
  var i = 0;
  var cur = env;
  while (i < stmts.length) {
    var s = stmts[i];
    var st = s[0];
    if (st === "return") { return ["ret", peval(s[1], cur, prog)]; }
    if (st === "var") { cur = envExtend(cur, s[1], peval(s[2], cur, prog)); }
    else if (st === "if") {
      var c = peval(s[1], cur, prog);
      if (c[0] === "s") {
        if (c[1]) {
          var r1 = pevalStmt(s[2], cur, prog);
          if (r1[0] === "ret") { return r1; }
          cur = r1[1];
        } else {
          if (s[3] !== null) {
            var r2 = pevalStmt(s[3], cur, prog);
            if (r2[0] === "ret") { return r2; }
            cur = r2[1];
          }
        }
      } else {
        // dynamic condition: residualization of statement-if is a later milestone (needs
        // join handling). Make it a HARD, visible stop rather than silently wrong.
        return ["ret", ["d", ["__dynamic_if_unsupported__", reify(c)]]];
      }
    }
    else if (st === "block") {
      var rb = pevalStmts(s[1], cur, prog);
      if (rb[0] === "ret") { return rb; }
    }
    i = i + 1;
  }
  return ["next"];
}

// evaluate a single statement (used by if-branches). Returns ["ret",v] or ["env", env].
function pevalStmt(s, env, prog) {
  var st = s[0];
  if (st === "return") { return ["ret", peval(s[1], env, prog)]; }
  if (st === "block") {
    var r = pevalStmts(s[1], env, prog);
    if (r[0] === "ret") { return r; }
    return ["env", env];
  }
  if (st === "var") { return ["env", envExtend(env, s[1], peval(s[2], env, prog))]; }
  // expression statement etc.: no binding change
  return ["env", env];
}

// ---- top-level: specialize fn `entry` of `prog`, binding params to given values ----
// staticArgs: array of ["s", value]; dynArgs: array of [name, ["d", ["var", name]]] already as values.
// argVals: array of value-cells (["s",..]/["d",..]) in param order. Returns a residual AST.
function specialize(prog, entryName, argVals) {
  var fn = findFun(prog, entryName);
  var env = envNil();
  var i = 0;
  while (i < fn[2].length) {
    env = envExtend(env, fn[2][i], argVals[i]);
    i = i + 1;
  }
  var r = pevalStmts(fn[3], env, prog);
  if (r[0] === "ret") { return reify(r[1]); }
  return ["lit", 0];
}

if (typeof module !== "undefined") {
  module.exports = {
    specialize: specialize, peval: peval, pevalStmts: pevalStmts,
    envNil: envNil, envExtend: envExtend, reify: reify,
  };
}
