module.exports = [
  {
    name: "PushNull pushes Null value",
    fn: () => {
      const H = require("../step/arith.js");
      const s = { frames: [{ locals: [], ostack: [] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
      const i = { tag: "PushNull" };
      const out = {};
      const result = H.PushNull(s, i, out);
      if (result.tag !== "Continue") throw new Error("Expected Continue");
      const top = s.frames[0].ostack[0];
      if (top.tag !== "Null") throw new Error("Expected Null, got " + top.tag);
    }
  },
  {
    name: "PushNull pushes onto ostack",
    fn: () => {
      const H = require("../step/arith.js");
      const s = { frames: [{ locals: [], ostack: [1,2] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
      const i = { tag: "PushNull" };
      const out = {};
      H.PushNull(s, i, out);
      if (s.frames[0].ostack.length !== 3) throw new Error("Expected length 3, got " + s.frames[0].ostack.length);
      const top = s.frames[0].ostack[2];
      if (top.tag !== "Null") throw new Error("Expected Null");
    }
  },
  {
    name: "PushNull returns Continue",
    fn: () => {
      const H = require("../step/arith.js");
      const s = { frames: [{ locals: [], ostack: [] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
      const i = { tag: "PushNull" };
      const out = {};
      const result = H.PushNull(s, i, out);
      if (result.tag !== "Continue") throw new Error("Expected Continue");
    }
  }
];