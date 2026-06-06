function step(instr, acc, input) {
  return instr[0] === "addlit" ? acc + instr[1]
       : instr[0] === "mullit" ? acc * instr[1]
       : instr[0] === "addin"  ? acc + input
       : instr[0] === "subin"  ? acc - input
       : acc;
}
function interp(prog, input) {
  return prog.reduce(function (acc, instr) { return step(instr, acc, input); }, 0);
}
