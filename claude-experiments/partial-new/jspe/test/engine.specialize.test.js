// Engine test with a MOCK client (no real JS semantics needed). Models an abstract
// loop that counts up forever; the whistle must generalize and tie it into a finite
// program. Proves: memo dedup + whistle/generalize loop-tying + worklist drain.
const { specialize } = require("../engine.js");

// state = {pc, v}.  v is a number (counting up) or "DYN" (generalized).
const client = {
  key: (s) => s.pc + ":" + s.v,
  point: (s) => "" + s.pc,
  step: (s, out, atEntry) => {
    if (s.pc === 1) return { tag: "Halt" };                 // exit block
    out.push({ tag: "Eval", dst: 0, expr: { tag: "Num", n: 1 } }); // some residual op
    const nv = s.v === "DYN" ? "DYN" : s.v + 1;
    return { tag: "Jump", state: { pc: 0, v: nv } };        // back-edge
  },
  whistle: (seen, cand) =>
    JSON.stringify(seen) !== JSON.stringify(cand) &&
    seen.pc === cand.pc &&
    (cand.v === "DYN" || (typeof seen.v === "number" && typeof cand.v === "number" && cand.v > seen.v)),
  generalize: (seen, from, out) => ({ pc: from.pc, v: "DYN" }),
};

module.exports = [
  {
    name: "loop ties into a finite program (does not unroll forever)",
    fn: () => {
      const prog = specialize(client, { pc: 0, v: 0 });
      if (prog.blocks.length > 5) throw new Error("expected the loop to tie (<=5 blocks), got " + prog.blocks.length);
      // there must be a back-edge (a Br whose target is an existing block)
      const hasBackEdge = prog.blocks.some((b, i) => b.term.tag === "Br" && b.term.b <= i);
      if (!hasBackEdge) throw new Error("expected a residual back-edge (the tied loop)");
    },
  },
  {
    name: "memo dedup: identical states share a block",
    fn: () => {
      let calls = 0;
      const c2 = { ...client, step: (s, out, ae) => { calls++; return s.pc === 1 ? { tag: "Halt" } : { tag: "Jump", state: { pc: 1, v: s.v } }; } };
      specialize(c2, { pc: 0, v: 0 });
      if (calls > 3) throw new Error("expected dedup to bound step calls, got " + calls);
    },
  },
];
