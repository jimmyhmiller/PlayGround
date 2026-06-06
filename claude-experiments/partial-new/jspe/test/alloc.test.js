module.exports = [
  {
    name: "alloc returns nextAddr and increments",
    fn: () => {
      const { alloc } = require("../state.js");
      const state = { frames: [], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
      const obj = { tag: "Object", fields: [] };
      const addr = alloc(state, obj);
      if (addr !== 0) throw new Error("expected 0");
      if (state.nextAddr !== 1) throw new Error("expected nextAddr 1");
      if (state.heap.get(0) !== obj) throw new Error("heap not set");
    }
  },
  {
    name: "alloc multiple objects",
    fn: () => {
      const { alloc } = require("../state.js");
      const state = { frames: [], heap: new Map(), nextAddr: 10, pendingJoins: [], handlers: [] };
      const a1 = alloc(state, { tag: "Array", elems: [] });
      const a2 = alloc(state, { tag: "Closure", fid: 0, captured: [] });
      if (a1 !== 10) throw new Error("expected 10");
      if (a2 !== 11) throw new Error("expected 11");
      if (state.nextAddr !== 12) throw new Error("expected nextAddr 12");
    }
  },
  {
    name: "alloc does not reuse addresses",
    fn: () => {
      const { alloc } = require("../state.js");
      const state = { frames: [], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
      const addr1 = alloc(state, { tag: "Object", fields: [["x", { tag: "Num", n: 1 }]] });
      const addr2 = alloc(state, { tag: "Object", fields: [] });
      if (addr1 === addr2) throw new Error("addresses must be distinct");
    }
  }
];