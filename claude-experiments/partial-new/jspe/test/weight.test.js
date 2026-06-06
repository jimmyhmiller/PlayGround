module.exports = [
  { name: "empty state", fn: () => {
    const { weight } = require("../state.js");
    const state = { frames: [], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
    const w = weight(state);
    if (w !== 0) throw new Error(`Expected 0, got ${w}`);
  }},
  { name: "simple Num abs", fn: () => {
    const { weight } = require("../state.js");
    const { AB } = require("../contracts.js");
    const state = {
      frames: [{ pc: 0, func: 0, locals: [AB.Num(5)], ostack: [] }],
      heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: []
    };
    const w = weight(state);
    if (w !== 1) throw new Error(`Expected 1, got ${w}`);
  }},
  { name: "Dyn with Bin", fn: () => {
    const { weight } = require("../state.js");
    const { AB, RE } = require("../contracts.js");
    const expr = RE.Bin("+", RE.Num(1), RE.Num(2));
    const state = {
      frames: [{ pc: 0, func: 0, locals: [AB.Dyn(expr)], ostack: [] }],
      heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: []
    };
    const w = weight(state);
    // Bin: 1 (root) + 1 (left Num) + 1 (right Num) = 3
    if (w !== 3) throw new Error(`Expected 3, got ${w}`);
  }},
  { name: "nested Bin", fn: () => {
    const { weight } = require("../state.js");
    const { AB, RE } = require("../contracts.js");
    const inner = RE.Bin("*", RE.Num(3), RE.Num(4));
    const outer = RE.Bin("+", inner, RE.Num(5));
    const state = {
      frames: [{ pc: 0, func: 0, locals: [AB.Dyn(outer)], ostack: [] }],
      heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: []
    };
    const w = weight(state);
    // outer Bin: 1 + inner(1+1+1=3) + 1 = 5
    if (w !== 5) throw new Error(`Expected 5, got ${w}`);
  }}
];