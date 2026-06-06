// A different little language: unary transforms on the input x.
//   ["x"]          x              ["k", n]       the constant n
//   ["inc", a]     a + 1          ["dbl", a]     a * 2
//   ["sq", a]      a * a          ["sum", a, b]  a + b
function interp(e, x) {
  if (e[0] === "x")   { return x; }
  if (e[0] === "k")   { return e[1]; }
  if (e[0] === "inc") { return interp(e[1], x) + 1; }
  if (e[0] === "dbl") { return interp(e[1], x) * 2; }
  if (e[0] === "sq")  { return interp(e[1], x) * interp(e[1], x); }
  if (e[0] === "sum") { return interp(e[1], x) + interp(e[2], x); }
  return 0;
}
