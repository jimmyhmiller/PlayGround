// ============================================================================
// A GENUINE partial evaluator `peval` for a tiny first-order object language L,
// written in the JS subset. It is GENERIC: it knows nothing about the source
// language being interpreted. Its only codegen rule is "a primitive op with a
// dynamic operand residualizes to that same op" (pevalPrim) -- so when it
// specializes an INTERPRETER expressed in L, the interpreter's `+`/`*`/`-` turn
// into compiled `+`/`*`/`-` automatically. Nothing here mentions the source
// language's opcodes.
//
// Binding-time value:  ["S", v]  static (v: number | string | array)
//                      ["D", c]  dynamic, c = residual-code AST
// Residual code AST:   ["rlit",n] | ["rin"] | ["radd",a,b] | ["rsub",a,b] | ["rmul",a,b]
// L expression AST:    ["num",n] | ["str",s] | ["var",name]
//                      | ["prim",op,[args]] | ["if",c,t,e] | ["call",fname,[args]]
// L program:           list of  [name, [params], body]
// ============================================================================

function lookupEnv(env, name) {       // env: list of [name, btval]
  var i = 0;
  while (i < env.length) { if (env[i][0] === name) { return env[i][1]; } i = i + 1; }
  return ["S", 0];                    // unbound -> 0 (shouldn't happen for well-formed L)
}
function lookupFn(funcs, name) {
  var i = 0;
  while (i < funcs.length) { if (funcs[i][0] === name) { return funcs[i]; } i = i + 1; }
  return funcs[0];
}
function bindParams(params, argvals) {
  var env = [];
  var i = 0;
  while (i < params.length) { env.push([params[i], argvals[i]]); i = i + 1; }
  return env;
}

// lift a binding-time value to residual code (a static number becomes a literal)
function toCode(v) {
  if (v[0] === "S") { return ["rlit", v[1]]; }
  return v[1];
}

// the ONLY codegen: a primitive whose operands are all static folds; otherwise
// it residualizes to the SAME primitive over residual operands.
function pevalPrim(op, vals) {
  if (op === "+" || op === "-" || op === "*") {
    var a = vals[0]; var b = vals[1];
    if (a[0] === "S" && b[0] === "S") {
      if (op === "+") { return ["S", a[1] + b[1]]; }
      if (op === "-") { return ["S", a[1] - b[1]]; }
      return ["S", a[1] * b[1]];
    }
    var ca = toCode(a); var cb = toCode(b);
    if (op === "+") { return ["D", ["radd", ca, cb]]; }
    if (op === "-") { return ["D", ["rsub", ca, cb]]; }
    return ["D", ["rmul", ca, cb]];
  }
  // structural / comparison prims: in P1 these run on the static program, so they
  // fold. (If an operand were dynamic they'd need to residualize too; not needed
  // for this source language, whose control flow is all program-structure.)
  var x = vals[0];
  if (op === "len") { return ["S", x[1].length]; }
  if (op === "idx") { return ["S", x[1][vals[1][1]]]; }
  if (op === "nth0") { return ["S", x[1][0]]; }
  if (op === "nth1") { return ["S", x[1][1]]; }
  if (op === "<") { var lt = 0; if (x[1] < vals[1][1]) { lt = 1; } return ["S", lt]; }
  if (op === "==") { var eq = 0; if (x[1] === vals[1][1]) { eq = 1; } return ["S", eq]; }
  return ["S", 0];
}

function pevalExpr(funcs, expr, env) {
  var tag = expr[0];
  if (tag === "num") { return ["S", expr[1]]; }
  if (tag === "str") { return ["S", expr[1]]; }
  if (tag === "var") { return lookupEnv(env, expr[1]); }
  if (tag === "prim") {
    var op = expr[1];
    var argExprs = expr[2];
    var vals = [];
    var i = 0;
    while (i < argExprs.length) { vals.push(pevalExpr(funcs, argExprs[i], env)); i = i + 1; }
    return pevalPrim(op, vals);
  }
  if (tag === "if") {
    var cv = pevalExpr(funcs, expr[1], env);
    if (cv[0] === "S") {
      if (cv[1]) { return pevalExpr(funcs, expr[2], env); }
      return pevalExpr(funcs, expr[3], env);
    }
    // dynamic condition -> residual if (not needed for this source lang; stub-safe)
    return ["D", ["rif", cv[1], toCode(pevalExpr(funcs, expr[2], env)), toCode(pevalExpr(funcs, expr[3], env))]];
  }
  // call: inline the callee (static-depth recursion over the static program)
  var fname = expr[1];
  var argExprs2 = expr[2];
  var argvals = [];
  var k = 0;
  while (k < argExprs2.length) { argvals.push(pevalExpr(funcs, argExprs2[k], env)); k = k + 1; }
  var fn = lookupFn(funcs, fname);
  var env2 = bindParams(fn[1], argvals);
  return pevalExpr(funcs, fn[2], env2);
}

// peval entry: specialize function `int` given binding-time arguments.
function peval(funcs, argvals) {
  var fn = lookupFn(funcs, "int");
  var env = bindParams(fn[1], argvals);
  return toCode(pevalExpr(funcs, fn[2], env));   // returns residual-code AST
}
