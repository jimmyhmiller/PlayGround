// step/control.js — Jmp / JmpIfFalsy / Dup / Pop + the shared jumpTo helper.
// jumpTo is the loop-head materialize point (both the back-edge AND fall-through into
// a head go through it), so a data loop's carried slots become stable runtime vars and
// the canonical-key memo ties the loop.
const meta = require("./meta.js");
const { gc, absTruthy, absToRExpr, cloneState } = require("../state.js");
const { dynamicallyControlled, materialize, consumePendingJoins } = require("../whistle.js");
const { RE } = require("../contracts.js");

function jumpTo(s, target, out) {
  const M = meta.get();
  const ns = cloneState(s);
  ns.frames[ns.frames.length - 1].pc = target;
  if (M.loopHeads.has(target) && dynamicallyControlled(ns)) {
    // Materialize loop-carried SCALARS that change + ARRAYS that are MUTATED in the
    // loop (SetIndexOp base). A read-only array (e.g. a static program/AST that drives
    // dispatch) stays STATIC so the dispatch folds away.
    const slots = [...(M.loopModified[target] || [])];
    for (const r of (M.loopMutRefs[target] || [])) if (!slots.includes(r)) slots.push(r);
    materialize(ns, slots, out, target);
  }
  consumePendingJoins(ns, target, out);
  gc(ns);
  return { tag: "Jump", state: ns };
}

const H = {};
H.Jmp = (s, i, out) => jumpTo(s, i.target, out);

H.JmpIfFalsy = (s, i, out) => {
  const top = s.frames[s.frames.length - 1];
  const c = top.ostack.pop();
  const t = absTruthy(c);
  if (t !== null) {                       // static condition: pick a side, no branch
    top.pc = t ? top.pc + 1 : i.target;   // falsy -> jump to target; truthy -> fall through
    return { tag: "Continue" };
  }
  const cond = RE.Unary("!", absToRExpr(c));  // dynamic: residual branch  if(!c) target else pc+1
  const tState = cloneState(s); tState.frames[tState.frames.length - 1].pc = i.target;
  const fState = cloneState(s); fState.frames[fState.frames.length - 1].pc = top.pc + 1;
  return { tag: "Branch", cond, t: tState, f: fState };
};

H.Dup = (s, i, out) => { const o = s.frames[s.frames.length - 1].ostack; o.push(o[o.length - 1]); return { tag: "Continue" }; };
H.Pop = (s, i, out) => { s.frames[s.frames.length - 1].ostack.pop(); return { tag: "Continue" }; };

module.exports = { Jmp: H.Jmp, JmpIfFalsy: H.JmpIfFalsy, Dup: H.Dup, Pop: H.Pop, jumpTo };
