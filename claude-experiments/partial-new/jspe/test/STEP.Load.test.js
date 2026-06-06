module.exports = [
  {
    name: "Load pushes local value onto ostack",
    fn: () => {
      const { RE, OP, AB } = require("../contracts.js");
      const H = require("../step/arith.js");
      const s = {
        frames: [{ pc: 0, func: 0, locals: [AB.Num(42), AB.Str("hello")], ostack: [] }],
        heap: new Map(),
        nextAddr: 0,
        pendingJoins: [],
        handlers: []
      };
      const instr = { tag: "Load", slot: 0 };
      const result = H.Load(s, instr, null);
      if (result.tag !== "Continue") throw new Error("Expected Continue");
      if (s.frames[0].ostack.length !== 1) throw new Error("ostack length should be 1");
      const val = s.frames[0].ostack[0];
      if (val.tag !== "Num" || val.n !== 42) throw new Error("Expected Num 42");
    }
  },
  {
    name: "Load pushes string local",
    fn: () => {
      const { RE, OP, AB } = require("../contracts.js");
      const H = require("../step/arith.js");
      const s = {
        frames: [{ pc: 0, func: 0, locals: [AB.Str("world")], ostack: [] }],
        heap: new Map(),
        nextAddr: 0,
        pendingJoins: [],
        handlers: []
      };
      const instr = { tag: "Load", slot: 0 };
      const result = H.Load(s, instr, null);
      if (result.tag !== "Continue") throw new Error("Expected Continue");
      const val = s.frames[0].ostack[0];
      if (val.tag !== "Str" || val.s !== "world") throw new Error("Expected Str world");
    }
  },
  {
    name: "Load pushes dynamic value",
    fn: () => {
      const { RE, OP, AB } = require("../contracts.js");
      const H = require("../step/arith.js");
      const dynVal = AB.Dyn(RE.Var(0));
      const s = {
        frames: [{ pc: 0, func: 0, locals: [dynVal], ostack: [] }],
        heap: new Map(),
        nextAddr: 0,
        pendingJoins: [],
        handlers: []
      };
      const instr = { tag: "Load", slot: 0 };
      const result = H.Load(s, instr, null);
      if (result.tag !== "Continue") throw new Error("Expected Continue");
      const val = s.frames[0].ostack[0];
      if (val.tag !== "Dyn" || val.expr.tag !== "Var" || val.expr.id !== 0) throw new Error("Expected Dyn Var 0");
    }
  }
];