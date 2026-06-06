module.exports = [
  { name: "push undef pushes Undef value", fn: () => {
    const H = require("../step/arith.js");
    const s = { frames: [{ locals: [], ostack: [] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
    const i = { tag: "PushUndef" };
    const out = {};
    const result = H.PushUndef(s, i, out);
    if (result.tag !== "Continue") throw new Error("Expected Continue");
    const top = s.frames[0].ostack[0];
    if (top.tag !== "Undef") throw new Error("Expected Undef, got " + top.tag);
  }},
  { name: "push undef multiple times", fn: () => {
    const H = require("../step/arith.js");
    const s = { frames: [{ locals: [], ostack: [] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
    const i = { tag: "PushUndef" };
    H.PushUndef(s, i, {});
    H.PushUndef(s, i, {});
    if (s.frames[0].ostack.length !== 2) throw new Error("Expected 2 items");
    if (s.frames[0].ostack[0].tag !== "Undef" || s.frames[0].ostack[1].tag !== "Undef") throw new Error("Both should be Undef");
  }},
  { name: "push undef after other values", fn: () => {
    const H = require("../step/arith.js");
    const s = { frames: [{ locals: [], ostack: [{ tag: "Num", n: 42 }] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
    const i = { tag: "PushUndef" };
    H.PushUndef(s, i, {});
    if (s.frames[0].ostack.length !== 2) throw new Error("Expected 2 items");
    if (s.frames[0].ostack[1].tag !== "Undef") throw new Error("Second should be Undef");
  }}
];