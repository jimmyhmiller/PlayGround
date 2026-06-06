// Generic partial evaluator (subset). Interpreter semantics are the static
// CLAUSES table; per-op actions EMIT JS SOURCE STRINGS (real target code).
// RustPE(main(prog) = gmix(CLAUSES, prog)) derives a source-to-source compiler.
function gmix(clauses, prog) {
  var stack = [];
  var i = 0;
  while (i < prog.length) {
    var instr = prog[i];
    var op = instr[0];
    var kind = "?"; var sym = "?";
    var c = 0;
    while (c < clauses.length) {
      var cl = clauses[c];
      if (cl[0] === op) { kind = cl[1]; sym = cl[2]; }
      c = c + 1;
    }
    if (kind === "leaflit") { stack.push("" + instr[1]); }
    else if (kind === "leafin") { stack.push("x"); }
    else if (kind === "binary") {
      var b = stack.pop(); var a = stack.pop();
      stack.push("(" + a + " " + sym + " " + b + ")");
    }
    i = i + 1;
  }
  return stack.pop();
}
function main(prog) {
  var CLAUSES = [
    ["pushlit", "leaflit", "+"],
    ["pushin",  "leafin",  "+"],
    ["add",     "binary",  "+"],
    ["mul",     "binary",  "*"],
    ["sub",     "binary",  "-"]
  ];
  return gmix(CLAUSES, prog);
}
