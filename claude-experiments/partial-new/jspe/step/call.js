// step/call.js — direct function calls + Ret (frames). Ports js.rs Call/Ret.
const meta = require("./meta.js");
const { absToRExpr } = require("../state.js");
const H = {};
H.Call = (s, i, out) => {
  const M = meta.get();
  const caller = s.frames[s.frames.length - 1];
  const args = [];
  for (let j = 0; j < i.nargs; j++) args.unshift(caller.ostack.pop());
  caller.pc++; // resume after the call when the callee returns
  const nl = M.nslots[i.fid];
  const locals = [];
  for (let j = 0; j < nl; j++) locals.push(j < args.length ? args[j] : { tag: "Undef" });
  s.frames.push({ pc: M.entries[i.fid], func: i.fid, locals, ostack: [] });
  return { tag: "Continue" };
};
H.Ret = (s, i, out) => {
  const top = s.frames[s.frames.length - 1];
  const retVal = top.ostack.pop();
  s.frames.pop();
  if (s.frames.length === 0) return { tag: "Halt", ret: absToRExpr(retVal) };
  s.frames[s.frames.length - 1].ostack.push(retVal);
  return { tag: "Continue" };
};
H.MakeClosure = (s, i, out) => { throw new Error("first-class closures not supported"); };
H.NewOp = (s, i, out) => { throw new Error("NewOp not supported"); };
module.exports = H;
