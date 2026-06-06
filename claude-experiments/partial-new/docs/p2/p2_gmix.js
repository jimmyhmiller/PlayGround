function gmix(clauses, prog) {
  var stack = [];
  var i = 0;
  while (i < prog.length) {
    var instr = prog[i];
    var op = instr[0];
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
function main(prog) {
  var CLAUSES = [
    ["pushlit", "leaflit", "lit"],
    ["pushin",  "leafin",  "in"],
    ["add",     "binary",  "add"],
    ["mul",     "binary",  "mul"],
    ["sub",     "binary",  "sub"]
  ];
  return gmix(CLAUSES, prog);   // prog DYNAMIC -> residual = the derived compiler
}
