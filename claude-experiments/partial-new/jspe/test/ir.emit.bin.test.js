// REFERENCE TEST (the pattern every task copies). Cases derived from rustRef.
const { emit } = require("../ir.js");
const { RE } = require("../contracts.js");
const eq = (got, want) => { if (got !== want) throw new Error(`got ${JSON.stringify(got)} want ${JSON.stringify(want)}`); };
module.exports = [
  { name: "num + num", fn: () => eq(emit(RE.Bin("+", RE.Num(1), RE.Num(2))), "(1 + 2)") },
  { name: "nested mul/add", fn: () => eq(emit(RE.Bin("*", RE.Bin("+", RE.Var(0), RE.Num(3)), RE.Num(2))), "((v0 + 3) * 2)") },
  { name: "comparison", fn: () => eq(emit(RE.Bin("===", RE.Var(1), RE.Str("addlit"))), '(v1 === "addlit")') },
];
