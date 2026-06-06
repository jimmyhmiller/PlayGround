// mix.subset.js — an ONLINE partial evaluator WRITTEN IN the jspe subset.
//
// Self-application artifact: stays inside the exact language jspe accepts (arith,
// comparison, arrays, index, .length, if/while, var/assign, named non-closure
// functions, array literals — NO closures, NO Map, NO .push, NO throw, NO &&/||,
// NO object literals), so jspe can partially-evaluate THIS file: jspe(mix,int)=compiler.
//
// AST-walking specializer over parse.js's tagged-array AST.
//   value ::= ["s", v]    STATIC (number/string/bool/array)
//           | ["d", ast]  DYNAMIC, described by a residual AST node
//   env   ::= ["nil"] | ["cons", name, value, env]
//   ctx   ::= ["ctx", prog, memoKeys, memoNames, fns, counter]   (mutable arrays)
//
// STRATEGY (no explicit binding-time analysis): a call with ALL-static args is
// UNFOLDED (run at PE time). A call with ANY dynamic arg is RESIDUALIZED into a memoized
// residual function keyed by its STATIC SKELETON. Structural recursion over a static
// program yields an ACYCLIC residual call graph (each call shrinks the static arg ->
// distinct skeleton) which a back-end pass inlines away; genuine DYNAMIC recursion
// (same skeleton re-entered) yields a SELF-CYCLE that stays a residual function = RFG.
//
// specializeProg returns ["prog", entryAst, fns] where fns = [[name, params, body],...].

// ---- environment ----
function envNil() { return ["nil"]; }
function envExtend(env, name, val) { return ["cons", name, val, env]; }
function envLookup(env, name) {
  var e = env;
  while (e[0] === "cons") {
    if (e[1] === name) { return e[2]; }
    e = e[3];
  }
  return ["d", ["var", name]]; // unbound -> poison dynamic var (visible if it ever fires)
}

// ---- value helpers ----
function reify(v) {
  if (v[0] === "s") { return ["lit", v[1]]; }
  return v[1];
}
function liftElem(x) { return ["s", x]; }
function reifyStaticOrVal(v) { if (v[0] === "s") { return v[1]; } return v; }
function arrPush(a, x) {
  var out = [];
  var i = 0;
  while (i < a.length) { out[i] = a[i]; i = i + 1; }
  out[a.length] = x;
  return out;
}
function anyDynamic(vals) {
  var i = 0;
  while (i < vals.length) { if (vals[i][0] !== "s") { return true; } i = i + 1; }
  return false;
}

// ---- static primitive operators (folds when op is static) ----
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
  return ["__unknown_op__", op];
}

function findFun(prog, name) {
  var i = 0;
  while (i < prog.length) {
    if (prog[i][1] === name) { return prog[i]; }
    i = i + 1;
  }
  return ["__missing_fun__", name];
}

// ---- static-skeleton serialization for memo keys (no typeof: probe .substring/.length) ----
function isStrV(v) { return v.substring !== undefined; }
function serial(v) {
  if (isStrV(v)) { return "'" + v; }
  if (v.length !== undefined) {            // array (numbers/bools have undefined .length)
    var s = "[";
    var i = 0;
    while (i < v.length) { s = s + serial(v[i]) + ","; i = i + 1; }
    return s + "]";
  }
  return "#" + v;                          // number/bool via coercion
}
function skeletonKey(fname, argvals) {
  var k = fname;
  var i = 0;
  while (i < argvals.length) {
    if (argvals[i][0] === "s") { k = k + "|s" + serial(argvals[i][1]); }
    else { k = k + "|d"; }
    i = i + 1;
  }
  return k;
}

// ---- ctx (mutable) ----
function mkCtx(prog) { return ["ctx", prog, [], [], [], 0]; }
function cprog(ctx) { return ctx[1]; }
function memoFind(ctx, key) {
  var keys = ctx[2];
  var i = 0;
  while (i < keys.length) { if (keys[i] === key) { return ctx[3][i]; } i = i + 1; }
  return "";
}
function ctxFresh(ctx, base) { var n = ctx[5]; ctx[5] = n + 1; return base + "_" + n; }

// ---- evaluate argument expressions to value cells ----
function evalArgs(args, env, ctx) {
  var out = [];
  var i = 0;
  while (i < args.length) { out = arrPush(out, peval(args[i], env, ctx)); i = i + 1; }
  return out;
}
function bindValues(params, vals) {
  var env = envNil();
  var i = 0;
  while (i < params.length) { env = envExtend(env, params[i], vals[i]); i = i + 1; }
  return env;
}

// ---- residualize a dynamic-arg call into a memoized residual function ----
function residualizeCall(ctx, fn, argvals) {
  var key = skeletonKey(fn[1], argvals);
  var name = memoFind(ctx, key);
  if (name === "") {
    name = ctxFresh(ctx, fn[1]);
    ctx[2] = arrPush(ctx[2], key);          // register BEFORE specializing body so the
    ctx[3] = arrPush(ctx[3], name);         // recursive call finds this same residual fn
    var nenv = envNil();
    var rparams = [];
    var i = 0;
    while (i < fn[2].length) {
      var pn = fn[2][i];
      if (argvals[i][0] === "s") { nenv = envExtend(nenv, pn, argvals[i]); }
      else { nenv = envExtend(nenv, pn, ["d", ["var", pn]]); rparams = arrPush(rparams, pn); }
      i = i + 1;
    }
    var r = pevalStmts(fn[3], nenv, ctx);
    var body = ["lit", 0];
    if (r[0] === "ret") { body = reify(r[1]); }
    ctx[4] = arrPush(ctx[4], [name, rparams, body]);
  }
  var callArgs = [];
  var j = 0;
  while (j < argvals.length) {
    if (argvals[j][0] !== "s") { callArgs = arrPush(callArgs, reify(argvals[j])); }
    j = j + 1;
  }
  return ["d", ["call", name, callArgs]];
}

// ---- expression specializer ----
function peval(expr, env, ctx) {
  var tag = expr[0];
  if (tag === "lit") { return ["s", expr[1]]; }
  if (tag === "var") { return envLookup(env, expr[1]); }
  if (tag === "arr") {
    var els = expr[1];
    var allStatic = true;
    var vals = [];
    var resid = [];
    var i = 0;
    while (i < els.length) {
      var ve = peval(els[i], env, ctx);
      if (ve[0] !== "s") { allStatic = false; }
      vals = arrPush(vals, reifyStaticOrVal(ve));
      resid = arrPush(resid, reify(ve));
      i = i + 1;
    }
    if (allStatic) { return ["s", vals]; }
    return ["d", ["arr", resid]];
  }
  if (tag === "bin") {
    var a = peval(expr[2], env, ctx);
    var b = peval(expr[3], env, ctx);
    if (a[0] === "s") {
      if (b[0] === "s") { return ["s", applyOp(expr[1], a[1], b[1])]; }
    }
    return ["d", ["bin", expr[1], reify(a), reify(b)]];
  }
  if (tag === "cond") {
    var c = peval(expr[1], env, ctx);
    if (c[0] === "s") {
      if (c[1]) { return peval(expr[2], env, ctx); }
      return peval(expr[3], env, ctx);
    }
    return ["d", ["cond", reify(c), reify(peval(expr[2], env, ctx)), reify(peval(expr[3], env, ctx))]];
  }
  if (tag === "idx") {
    var base = peval(expr[1], env, ctx);
    var ix = peval(expr[2], env, ctx);
    if (base[0] === "s") {
      if (ix[0] === "s") { return liftElem(base[1][ix[1]]); }
    }
    return ["d", ["idx", reify(base), reify(ix)]];
  }
  if (tag === "dot") {
    var d = peval(expr[1], env, ctx);
    if (d[0] === "s") {
      if (expr[2] === "length") { return ["s", d[1].length]; }
    }
    return ["d", ["dot", reify(d), expr[2]]];
  }
  if (tag === "call") {
    var fname = expr[1][1];
    var fn = findFun(cprog(ctx), fname);
    var argvals = evalArgs(expr[2], env, ctx);
    if (anyDynamic(argvals)) { return residualizeCall(ctx, fn, argvals); }
    // fully-static call: UNFOLD (run at PE time; result may stay static)
    var senv = bindValues(fn[2], argvals);
    var rs = pevalStmts(fn[3], senv, ctx);
    if (rs[0] === "ret") { return rs[1]; }
    return ["s", 0];
  }
  return ["d", ["__unhandled_expr__", tag]];
}

// ---- statement specializer: returns ["ret", value] or ["next"] ----
function pevalStmts(stmts, env, ctx) {
  var i = 0;
  var cur = env;
  while (i < stmts.length) {
    var s = stmts[i];
    var st = s[0];
    if (st === "return") { return ["ret", peval(s[1], cur, ctx)]; }
    if (st === "var") { cur = envExtend(cur, s[1], peval(s[2], cur, ctx)); }
    else if (st === "if") {
      var c = peval(s[1], cur, ctx);
      if (c[0] === "s") {
        if (c[1]) {
          var r1 = pevalStmt(s[2], cur, ctx);
          if (r1[0] === "ret") { return r1; }
          cur = r1[1];
        } else {
          if (s[3] !== null) {
            var r2 = pevalStmt(s[3], cur, ctx);
            if (r2[0] === "ret") { return r2; }
            cur = r2[1];
          }
        }
      } else {
        // dynamic condition -> cond(C, then, else); implicit else = the continuation.
        var thenVal = branchValue(s[2], cur, ctx);
        var elseVal = ["d", ["__dyn_if_continuation_no_return__"]];
        if (s[3] !== null) {
          elseVal = branchValue(s[3], cur, ctx);
        } else {
          var rest = tailFrom(stmts, i + 1);
          var rr = pevalStmts(rest, cur, ctx);
          if (rr[0] !== "ret") { return ["ret", ["d", ["__dyn_if_continuation_no_return__"]]]; }
          elseVal = rr[1];
        }
        return ["ret", ["d", ["cond", reify(c), reify(thenVal), reify(elseVal)]]];
      }
    }
    else if (st === "block") {
      var rb = pevalStmts(s[1], cur, ctx);
      if (rb[0] === "ret") { return rb; }
    }
    i = i + 1;
  }
  return ["next"];
}

function branchValue(stmt, env, ctx) {
  var r = pevalStmt(stmt, env, ctx);
  if (r[0] === "ret") { return r[1]; }
  return ["d", ["__dyn_branch_no_return__"]];
}
function tailFrom(stmts, start) {
  var out = [];
  var i = start;
  while (i < stmts.length) { out[out.length] = stmts[i]; i = i + 1; }
  return out;
}
function pevalStmt(s, env, ctx) {
  var st = s[0];
  if (st === "return") { return ["ret", peval(s[1], env, ctx)]; }
  if (st === "block") {
    var r = pevalStmts(s[1], env, ctx);
    if (r[0] === "ret") { return r; }
    return ["env", env];
  }
  if (st === "var") { return ["env", envExtend(env, s[1], peval(s[2], env, ctx))]; }
  return ["env", env];
}

// ---- top-level: specialize fn `entryName`, binding params to argVals (value cells) ----
function specializeProg(prog, entryName, argVals) {
  var ctx = mkCtx(prog);
  var fn = findFun(prog, entryName);
  var env = envNil();
  var i = 0;
  while (i < fn[2].length) { env = envExtend(env, fn[2][i], argVals[i]); i = i + 1; }
  var r = pevalStmts(fn[3], env, ctx);
  var entry = ["lit", 0];
  if (r[0] === "ret") { entry = reify(r[1]); }
  return ["prog", entry, ctx[4]];
}

if (typeof module !== "undefined") {
  module.exports = {
    specializeProg: specializeProg, peval: peval, pevalStmts: pevalStmts,
    envNil: envNil, envExtend: envExtend, reify: reify, skeletonKey: skeletonKey,
  };
}
