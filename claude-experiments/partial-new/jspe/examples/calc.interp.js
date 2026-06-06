// An interpreter for a tiny arithmetic-expression language, written in jspe's subset.
// Source programs are tagged-array ASTs, e.g.  ["add", ["mul", ["x"], ["x"]], ["num", 5]]
//   ["num", k]      a constant            ["x"]            the input variable
//   ["add", a, b]   a + b                 ["mul", a, b]    a * b
//   ["sub", a, b]   a - b                 ["neg", a]       -a   (here: 0 - a)
//
// Feed this file to `compgen` to turn it into a COMPILER (2nd Futamura projection):
//   node compgen.js examples/calc.interp.js > calc-compiler.js
//   node calc-compiler.js '["add",["mul",["x"],["x"]],["num",5]]'        # -> target JS
//   node calc-compiler.js '["add",["mul",["x"],["x"]],["num",5]]' 7      # -> compiles AND runs at x=7
function interp(e, x) {
  if (e[0] === "num") { return e[1]; }
  if (e[0] === "x")   { return x; }
  if (e[0] === "add") { return interp(e[1], x) + interp(e[2], x); }
  if (e[0] === "mul") { return interp(e[1], x) * interp(e[2], x); }
  if (e[0] === "sub") { return interp(e[1], x) - interp(e[2], x); }
  if (e[0] === "neg") { return 0 - interp(e[1], x); }
  return 0;
}
