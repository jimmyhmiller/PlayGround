// engine.js — the generic driving/memoization/whistle engine. REFERENCE MODULE #2.
// Direct port of src/engine.rs. This is the part that NEVER changes per client.
// Implemented (not stubbed) so it anchors the contract; tested with a mock client.
//
// client = { key(s)->string, point(s)->string, step(s,out,atEntry)->Step,
//            whistle(seen,cand)->bool, generalize(seen,from,out)->State }
// Step  = {tag:"Continue"} | {tag:"Halt"} | {tag:"Jump",state} | {tag:"Branch",cond,t,f}
// Program = {blocks:[{ops,term}], entry:int}
// Terminator = {tag:"Halt"} | {tag:"Br",b} | {tag:"CondBr",cond,t,f} | {tag:"Unset"}

const UNROLL_BOUND = 5000; // backstop against unbounded static unrolling

function createOrGet(client, st, cx) {
  const k = client.key(st);
  if (cx.memo.has(k)) return cx.memo.get(k);
  const id = cx.blocks.length;
  cx.blocks.push({ ops: [], term: { tag: "Unset" } });
  cx.memo.set(k, id);
  const p = client.point(st);
  if (!cx.seen.has(p)) cx.seen.set(p, []);
  cx.seen.get(p).push(st);
  cx.work.push([id, structuredClone(st)]);
  return id;
}

function resolveJump(client, target, ops, cx) {
  for (;;) {
    const k = client.key(target);
    if (cx.memo.has(k)) return cx.memo.get(k);
    const p = client.point(target);
    const seenList = cx.seen.get(p) || [];
    const force = seenList.length >= UNROLL_BOUND;
    let cand = null;
    for (let i = seenList.length - 1; i >= 0; i--) {
      if (force || client.whistle(seenList[i], target)) { cand = seenList[i]; break; }
    }
    if (cand === null) return createOrGet(client, target, cx);
    const general = client.generalize(cand, target, ops);
    if (client.key(general) === k) return createOrGet(client, target, cx); // no progress
    target = general;
  }
}

function specialize(client, init) {
  const cx = { blocks: [], memo: new Map(), seen: new Map(), work: [] };
  const entry = createOrGet(client, init, cx);
  while (cx.work.length) {
    const [bid, state] = cx.work.pop();
    let s = state, ops = [], atEntry = true, term;
    for (;;) {
      const st = client.step(s, ops, atEntry);
      if (st.tag === "Continue") { atEntry = false; continue; }
      if (st.tag === "Halt") { term = { tag: "Halt", ret: st.ret }; break; }
      if (st.tag === "Jump") { term = { tag: "Br", b: resolveJump(client, st.state, ops, cx) }; break; }
      if (st.tag === "Branch") {
        const t = createOrGet(client, st.t, cx), f = createOrGet(client, st.f, cx);
        term = { tag: "CondBr", cond: st.cond, t, f }; break;
      }
      throw new Error("bad Step tag: " + st.tag);
    }
    cx.blocks[bid].ops = ops;
    cx.blocks[bid].term = term;
  }
  return { blocks: cx.blocks, entry };
}

module.exports = { specialize, createOrGet, resolveJump, UNROLL_BOUND };
