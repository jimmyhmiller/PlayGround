// A GENUINE partial evaluator for L, structured so RustPE can specialize IT w.r.t.
// an interpreter (P2). Design notes that make P2 tractable:
//   * PROGRAM STRUCTURE (prog, instr, opcode) is PLAIN data peval walks with normal
//     ops. When RustPE makes `prog` dynamic, those ops residualize automatically --
//     peval needs no manual residual-building for them.
//   * SOURCE VALUES (acc, input) are binding-time tagged ["S",n] | ["D",codeAST];
//     pevalPrim residualizes arithmetic GENERICALLY (the codegen for +,*,- falls out
//     of the interpreter's prim ops -- nothing opcode-specific here).
//   * `lift` makes a program literal into a (always-dynamic) source value, so the
//     accumulator has a UNIFORM representation every iteration -> the loop ties.
//   * The interpreter uses STRUCTURAL recursion over the program list (empty/first/
//     rest), so the loop-carried `prog` shrinks via residual `rest`; RustPE's growth
//     whistle ties peval's recursion into a residual loop.
//
// codeAST (the compiled OUTPUT): ["rlit",n] | ["rin"] | ["radd",a,b] | ["rsub",a,b] | ["rmul",a,b]
// L expr: ["num",n]|["str",s]|["var",x]|["lift",e]|["prim",op,[args]]|["if",c,t,e]|["call",f,[args]]

function lookupEnv(env, name) {
  var i = 0;
  while (i < env.length) { if (env[i][0] === name) { return env[i][1]; } i = i + 1; }
  return 0;
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
function toCode(v) { if (v[0] === "S") { return ["rlit", v[1]]; } return v[1]; }

// GENERIC arithmetic codegen: static operands fold, else residualize the SAME op.
function pevalArith(op, a, b) {
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

function pevalExpr(funcs, expr, env) {
  var tag = expr[0];
  if (tag === "num") { return expr[1]; }                 // plain program datum
  if (tag === "str") { return expr[1]; }                 // plain program datum
  if (tag === "var") { return lookupEnv(env, expr[1]); }
  if (tag === "lift") {                                   // program literal -> dynamic source value
    var d = pevalExpr(funcs, expr[1], env);
    return ["D", ["rlit", d]];
  }
  if (tag === "prim") {
    var op = expr[1];
    var as = expr[2];
    if (op === "+" || op === "-" || op === "*") {
      return pevalArith(op, pevalExpr(funcs, as[0], env), pevalExpr(funcs, as[1], env));
    }
    // structural prims on PLAIN program data (residualized by RustPE when dynamic)
    if (op === "empty") { var p = pevalExpr(funcs, as[0], env); return p.length === 0; }
    if (op === "first") { return pevalExpr(funcs, as[0], env)[0]; }
    if (op === "rest")  { var q = pevalExpr(funcs, as[0], env); return q.slice(1); }
    if (op === "nth0")  { return pevalExpr(funcs, as[0], env)[0]; }
    if (op === "nth1")  { return pevalExpr(funcs, as[0], env)[1]; }
    if (op === "==")    { return pevalExpr(funcs, as[0], env) === pevalExpr(funcs, as[1], env); }
    return 0;
  }
  if (tag === "if") {
    var cv = pevalExpr(funcs, expr[1], env);              // plain boolean (RustPE residualizes if dynamic)
    if (cv) { return pevalExpr(funcs, expr[2], env); }
    return pevalExpr(funcs, expr[3], env);
  }
  if (tag === "fold") {
    // ["fold", listExpr, initExpr, stepFnName, [extraArgExprs]]
    // An EXPLICIT loop (not inlined recursion) so the loop-carried list & acc are
    // peval-level LOCALS: when `list` is dynamic, RustPE residualizes this while
    // and its growth whistle ties it. `list` shrinks STRUCTURALLY (slice(1)) so it
    // has no static counter that would unroll forever.
    var list = pevalExpr(funcs, expr[1], env);
    var acc = pevalExpr(funcs, expr[2], env);
    var stepFn = lookupFn(funcs, expr[3]);
    var extraExprs = expr[4];
    var extra = [];
    var e = 0;
    while (e < extraExprs.length) { extra.push(pevalExpr(funcs, extraExprs[e], env)); e = e + 1; }
    while (list.length > 0) {
      var item = list[0];
      var args = [item, acc];
      var j = 0;
      while (j < extra.length) { args.push(extra[j]); j = j + 1; }
      acc = pevalExpr(funcs, stepFn[2], bindParams(stepFn[1], args));
      list = list.slice(1);
    }
    return acc;
  }
  // call: inline the callee. structural recursion over the (shrinking) program list.
  var fname = expr[1];
  var as2 = expr[2];
  var argvals = [];
  var k = 0;
  while (k < as2.length) { argvals.push(pevalExpr(funcs, as2[k], env)); k = k + 1; }
  var fn = lookupFn(funcs, fname);
  return pevalExpr(funcs, fn[2], bindParams(fn[1], argvals));
}

// peval entry: specialize `int` applied to (prog, input-as-dynamic-source-value).
function peval(funcs, prog) {
  var fn = lookupFn(funcs, "int");
  var env = bindParams(fn[1], [prog, ["D", ["rin"]]]);
  return toCode(pevalExpr(funcs, fn[2], env));
}
