// ---- object language L: stack bytecode over one input. prog = list of [op, arg] ----
// ops: pushlit n | pushin | add | mul | sub
function interp(prog, input) {
  var stack = [];
  var i = 0;
  while (i < prog.length) {
    var instr = prog[i];
    var op = instr[0];
    if (op === "pushlit") { stack.push(instr[1]); }
    else if (op === "pushin") { stack.push(input); }
    else if (op === "add") { var b = stack.pop(); var a = stack.pop(); stack.push(a + b); }
    else if (op === "mul") { var b = stack.pop(); var a = stack.pop(); stack.push(a * b); }
    else if (op === "sub") { var b = stack.pop(); var a = stack.pop(); stack.push(a - b); }
    i = i + 1;
  }
  return stack.pop();
}

// ---- target: compiled expression trees, and their evaluator ----
function evalTree(t, input) {
  var tag = t[0];
  if (tag === "lit") { return t[1]; }
  if (tag === "in") { return input; }
  if (tag === "add") { return evalTree(t[1], input) + evalTree(t[2], input); }
  if (tag === "mul") { return evalTree(t[1], input) * evalTree(t[2], input); }
  if (tag === "sub") { return evalTree(t[1], input) - evalTree(t[2], input); }
  return 0;
}

// ---- gmix: GENERIC PE. Takes the interpreter's per-op semantics as a STATIC
//      clause table + a src program; emits a compiled tree. With `clauses` static
//      and `src` dynamic (P2), RustPE folds the clause lookup into an op-specific
//      chain -> an interpreter-specific COMPILER. ----
function gmix(clauses, prog) {
  var stack = [];
  var i = 0;
  while (i < prog.length) {
    var instr = prog[i];
    var op = instr[0];
    // dynamic-key lookup into the static clause table (partially-static map)
    var kind = "?"; var tag = "?";
    var c = 0;
    while (c < clauses.length) {
      var cl = clauses[c];
      if (cl[0] === op) { kind = cl[1]; tag = cl[2]; }
      c = c + 1;
    }
    if (kind === "leaflit") { stack.push([tag, instr[1]]); }
    else if (kind === "leafin") { stack.push([tag]); }
    else if (kind === "binary") { var b = stack.pop(); var a = stack.pop(); stack.push([tag, a, b]); }
    i = i + 1;
  }
  return stack.pop();
}

var CLAUSES = [
  ["pushlit", "leaflit", "lit"],
  ["pushin",  "leafin",  "in"],
  ["add",     "binary",  "add"],
  ["mul",     "binary",  "mul"],
  ["sub",     "binary",  "sub"]
];

// the compiler = gmix with the interpreter's clauses baked in
function compile(prog) { return gmix(CLAUSES, prog); }

// ---- correctness oracle: compiler(prog) then evalTree == interp, over random progs ----
function rndProg(rng, n) {
  // build a stack-valid program of ~n ops
  var prog = []; var depth = 0;
  while (prog.length < n || depth < 1) {
    var canBin = depth >= 2;
    var r = Math.floor(rng() * (canBin ? 5 : 2));
    if (r === 0) { prog.push(["pushlit", Math.floor(rng()*9)-4]); depth++; }
    else if (r === 1) { prog.push(["pushin", 0]); depth++; }
    else if (r === 2) { prog.push(["add", 0]); depth--; }
    else if (r === 3) { prog.push(["mul", 0]); depth--; }
    else { prog.push(["sub", 0]); depth--; }
    if (prog.length > n*3) break;
  }
  // reduce to single value
  while (depth > 1) { prog.push(["add", 0]); depth--; }
  return prog;
}
var seed = 12345;
function rng() { seed = (seed*1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }
var bad = 0;
for (var t = 0; t < 2000; t++) {
  var prog = rndProg(rng, 2 + Math.floor(rng()*6));
  for (var x = -3; x <= 3; x++) {
    var viaInterp = interp(prog, x);
    var viaCompiler = evalTree(compile(prog), x);
    if (viaInterp !== viaCompiler) { console.log("MISMATCH", JSON.stringify(prog), "x="+x, viaInterp, viaCompiler); bad++; if(bad>5)process.exit(1); }
  }
}
console.log(bad === 0 ? "compiler CORRECT vs interpreter (2000 progs x 7 inputs)" : "BAD "+bad);
