// int(prog,input) = fold step over prog, acc starts at 0; step dispatches opcodes.
var INT = [
  ["int", ["prog","input"],
    ["fold", ["var","prog"], ["lift",["num",0]], "step", [["var","input"]]]],
  ["step", ["instr","acc","input"],
    ["if",["prim","==",[["prim","nth0",[["var","instr"]]],["str","addlit"]]],
       ["prim","+",[["var","acc"],["lift",["prim","nth1",[["var","instr"]]]]]],
    ["if",["prim","==",[["prim","nth0",[["var","instr"]]],["str","mullit"]]],
       ["prim","*",[["var","acc"],["lift",["prim","nth1",[["var","instr"]]]]]],
    ["if",["prim","==",[["prim","nth0",[["var","instr"]]],["str","addin"]]],
       ["prim","+",[["var","acc"],["var","input"]]],
    ["if",["prim","==",[["prim","nth0",[["var","instr"]]],["str","subin"]]],
       ["prim","-",[["var","acc"],["var","input"]]],
       ["var","acc"]]]]]]
];
if (typeof module !== "undefined") module.exports = INT;
