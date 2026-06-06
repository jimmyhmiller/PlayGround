module.exports = [
  { name: "store single slot", fn: () => {
    const H = require("../step/arith.js");
    const { AB } = require("../contracts.js");
    const s = { frames: [{ pc: 0, func: 0, locals: [AB.Undef()], ostack: [AB.Num(42)] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
    const i = { slot: 0 };
    const result = H.Store(s, i, null);
    if (result.tag !== "Continue") throw new Error("Expected Continue");
    if (s.frames[0].locals[0].tag !== "Num" || s.frames[0].locals[0].n !== 42) throw new Error("Store failed");
    if (s.frames[0].ostack.length !== 0) throw new Error("ostack should be empty");
  }},
  { name: "store multiple slots", fn: () => {
    const H = require("../step/arith.js");
    const { AB } = require("../contracts.js");
    const s = { frames: [{ pc: 0, func: 0, locals: [AB.Undef(), AB.Undef()], ostack: [AB.Num(10), AB.Num(20)] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
    let i = { slot: 0 };
    let result = H.Store(s, i, null);
    if (result.tag !== "Continue") throw new Error("Expected Continue");
    if (s.frames[0].locals[0].tag !== "Num" || s.frames[0].locals[0].n !== 20) throw new Error("First store wrong");
    if (s.frames[0].ostack.length !== 1) throw new Error("ostack should have 1 element");
    i = { slot: 1 };
    result = H.Store(s, i, null);
    if (result.tag !== "Continue") throw new Error("Expected Continue");
    if (s.frames[0].locals[1].tag !== "Num" || s.frames[0].locals[1].n !== 10) throw new Error("Second store wrong");
    if (s.frames[0].ostack.length !== 0) throw new Error("ostack should be empty");
  }},
  { name: "store overwrites existing", fn: () => {
    const H = require("../step/arith.js");
    const { AB } = require("../contracts.js");
    const s = { frames: [{ pc: 0, func: 0, locals: [AB.Num(1)], ostack: [AB.Num(2)] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
    const i = { slot: 0 };
    const result = H.Store(s, i, null);
    if (result.tag !== "Continue") throw new Error("Expected Continue");
    if (s.frames[0].locals[0].tag !== "Num" || s.frames[0].locals[0].n !== 2) throw new Error("Overwrite failed");
    if (s.frames[0].ostack.length !== 0) throw new Error("ostack should be empty");
  }}
];