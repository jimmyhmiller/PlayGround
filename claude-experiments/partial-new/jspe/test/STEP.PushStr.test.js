module.exports = [
  { name: "push string", fn: () => {
    const H = require("../step/arith.js");
    const s = { frames: [{ locals: [], ostack: [] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
    const i = { s: "hello" };
    const result = H.PushStr(s, i, null);
    if (result.tag !== "Continue") throw new Error("expected Continue");
    if (s.frames[0].ostack.length !== 1) throw new Error("expected 1 value on stack");
    const val = s.frames[0].ostack[0];
    if (val.tag !== "Str" || val.s !== "hello") throw new Error("expected Str hello");
  }},
  { name: "push empty string", fn: () => {
    const H = require("../step/arith.js");
    const s = { frames: [{ locals: [], ostack: [] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
    const i = { s: "" };
    H.PushStr(s, i, null);
    const val = s.frames[0].ostack[0];
    if (val.tag !== "Str" || val.s !== "") throw new Error("expected empty string");
  }},
  { name: "push string with special chars", fn: () => {
    const H = require("../step/arith.js");
    const s = { frames: [{ locals: [], ostack: [] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
    const i = { s: "a\nb" };
    H.PushStr(s, i, null);
    const val = s.frames[0].ostack[0];
    if (val.tag !== "Str" || val.s !== "a\nb") throw new Error("expected string with newline");
  }}
];