module.exports = [
  {
    name: "canonicalize empty state",
    fn: () => {
      const { canonicalize } = require("../state.js");
      const state = { frames: [], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
      const result = canonicalize(state);
      if (result.frames.length !== 0) throw new Error("frames not empty");
      if (result.heap.size !== 0) throw new Error("heap not empty");
      if (result.nextAddr !== 0) throw new Error("nextAddr not 0");
    }
  },
  {
    name: "canonicalize remaps refs in nested objects",
    fn: () => {
      const { canonicalize } = require("../state.js");
      const { AB } = require("../contracts.js");
      const heap = new Map();
      heap.set(0, { tag: "Object", fields: [["x", AB.Ref(1)]] });
      heap.set(1, { tag: "Object", fields: [["y", AB.Num(42)]] });
      const state = {
        frames: [{ pc: 0, func: 0, locals: [AB.Ref(0)], ostack: [] }],
        heap: heap,
        nextAddr: 2,
        pendingJoins: [],
        handlers: []
      };
      const result = canonicalize(state);
      if (result.heap.size !== 2) throw new Error("heap size not 2");
      const obj0 = result.heap.get(0);
      if (!obj0 || obj0.tag !== "Object") throw new Error("obj0 missing");
      const field = obj0.fields[0];
      if (field[1].tag !== "Ref" || field[1].addr !== 1) throw new Error("ref not remapped");
      const obj1 = result.heap.get(1);
      if (!obj1 || obj1.tag !== "Object") throw new Error("obj1 missing");
      if (obj1.fields[0][1].n !== 42) throw new Error("value changed");
      if (result.frames[0].locals[0].addr !== 0) throw new Error("frame ref not remapped");
    }
  },
  {
    name: "canonicalize drops unreachable",
    fn: () => {
      const { canonicalize } = require("../state.js");
      const { AB } = require("../contracts.js");
      const heap = new Map();
      heap.set(0, { tag: "Object", fields: [["a", AB.Num(1)]] });
      heap.set(1, { tag: "Object", fields: [["b", AB.Num(2)]] });
      const state = {
        frames: [{ pc: 0, func: 0, locals: [AB.Ref(0)], ostack: [] }],
        heap: heap,
        nextAddr: 2,
        pendingJoins: [],
        handlers: []
      };
      const result = canonicalize(state);
      if (result.heap.size !== 1) throw new Error("heap size not 1");
      if (!result.heap.has(0)) throw new Error("reachable addr missing");
      if (result.heap.has(1)) throw new Error("unreachable addr present");
      if (result.nextAddr !== 1) throw new Error("nextAddr not 1");
    }
  }
];