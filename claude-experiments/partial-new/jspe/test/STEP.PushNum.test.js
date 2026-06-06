module.exports = [
  {name: "PushNum pushes a Num value", fn: () => {
    const H = require("../step/arith.js");
    const s = { frames: [{ locals: [], ostack: [] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
    const i = { tag: "PushNum", n: 42 };
    const result = H.PushNum(s, i, null);
    if (result.tag !== "Continue") throw new Error("Expected Continue");
    const top = s.frames[0].ostack.pop();
    if (top.tag !== "Num" || top.n !== 42) throw new Error("Expected Num 42");
  }},
  {name: "PushNull pushes Null", fn: () => {
    const H = require("../step/arith.js");
    const s = { frames: [{ locals: [], ostack: [] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
    const i = { tag: "PushNull" };
    const result = H.PushNull(s, i, null);
    if (result.tag !== "Continue") throw new Error("Expected Continue");
    const top = s.frames[0].ostack.pop();
    if (top.tag !== "Null") throw new Error("Expected Null");
  }},
  {name: "PushStr pushes a Str value", fn: () => {
    const H = require("../step/arith.js");
    const s = { frames: [{ locals: [], ostack: [] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
    const i = { tag: "PushStr", s: "hello" };
    const result = H.PushStr(s, i, null);
    if (result.tag !== "Continue") throw new Error("Expected Continue");
    const top = s.frames[0].ostack.pop();
    if (top.tag !== "Str" || top.s !== "hello") throw new Error("Expected Str hello");
  }}
];