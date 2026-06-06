module.exports = [
  { name: "push true", fn: () => {
    const H = require("../step/arith.js");
    const s = { frames: [{ ostack: [] }] };
    const i = { tag: "PushBool", b: true };
    const out = null;
    const res = H.PushBool(s, i, out);
    if (res.tag !== "Continue") throw new Error("expected Continue");
    if (s.frames[0].ostack.length !== 1) throw new Error("stack length");
    const val = s.frames[0].ostack[0];
    if (val.tag !== "Bool" || val.b !== true) throw new Error("value mismatch");
  }},
  { name: "push false", fn: () => {
    const H = require("../step/arith.js");
    const s = { frames: [{ ostack: [] }] };
    const i = { tag: "PushBool", b: false };
    const out = null;
    const res = H.PushBool(s, i, out);
    if (res.tag !== "Continue") throw new Error("expected Continue");
    if (s.frames[0].ostack.length !== 1) throw new Error("stack length");
    const val = s.frames[0].ostack[0];
    if (val.tag !== "Bool" || val.b !== false) throw new Error("value mismatch");
  }},
  { name: "push multiple", fn: () => {
    const H = require("../step/arith.js");
    const s = { frames: [{ ostack: [] }] };
    const i1 = { tag: "PushBool", b: true };
    const i2 = { tag: "PushBool", b: false };
    H.PushBool(s, i1, null);
    H.PushBool(s, i2, null);
    if (s.frames[0].ostack.length !== 2) throw new Error("stack length");
    if (s.frames[0].ostack[0].b !== true) throw new Error("first");
    if (s.frames[0].ostack[1].b !== false) throw new Error("second");
  }}
];