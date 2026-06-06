// jspe.js — INTEGRATED ENTRY (Wave-0 wiring; frozen). Assembles the modules into
//   compile(source) -> residual JS string.
// Works as soon as the leaf tasks land; throws UNIMPLEMENTED on the first missing one.
const ir = require("./ir.js");
const state = require("./state.js");
const engine = require("./engine.js");
const lower = require("./lower.js");
const stepmod = require("./step/index.js");
const whistle = require("./whistle.js");

function compile(source, paramName) {
  const L = lower.lower(source);
  require("./step/meta.js").setMeta({ code: L.code, leaders: L.leaders, loopHeads: L.loopHeads, loopModified: L.loopModified, loopMutRefs: L.loopMutRefs, entries: L.entries, nslots: L.nslots });
  const client = {
    key: state.keyOf,
    clone: state.cloneState,
    point: (s) => s.frames.map((f) => f.pc).join(","),
    step: stepmod.step,
    whistle: (seen, cand) => whistle.whistle(seen, cand, L),
    generalize: whistle.generalize,
  };
  const program = engine.specialize(client, L.initState);
  return ir.emitProgram(program, paramName || L.paramName || "v0");
}

// ---- deep-load a concrete JS value into jspe's abstract heap as STATIC structure ----
const { AB, RE } = require("./contracts.js");
function loadStatic(st, value) {
  if (value && typeof value === "object" && value.__dyn) return AB.Dyn(value.__dyn); // dynamic hole
  if (typeof value === "number") return AB.Num(value);
  if (typeof value === "string") return AB.Str(value);
  if (typeof value === "boolean") return AB.Bool(value);
  if (value === null) return AB.Null();
  if (value === undefined) return AB.Undef();
  if (!st.frozen) st.frozen = new Set();
  if (Array.isArray(value)) {
    const elems = value.map((v) => loadStatic(st, v));
    const addr = state.alloc(st, { tag: "Array", elems });
    st.frozen.add(addr); // immutable static input -> excluded from per-state keys, never cloned
    return AB.Ref(addr);
  }
  const fields = Object.entries(value).map(([k, v]) => [k, loadStatic(st, v)]);
  const addr = state.alloc(st, { tag: "Object", fields });
  st.frozen.add(addr);
  return AB.Ref(addr);
}

// ---- GENERAL entry: specialize fn `entryName`, each param static (a JS value) or dynamic.
// argSpecs[i] = {s: jsValue} (static) | {d: true} (dynamic -> Var(slot i)). Returns residual JS.
function specializeGeneral(source, entryName, argSpecs, paramName) {
  const L = lower.lower(source);
  require("./step/meta.js").setMeta({ code: L.code, leaders: L.leaders, loopHeads: L.loopHeads, loopModified: L.loopModified, loopMutRefs: L.loopMutRefs, entries: L.entries, nslots: L.nslots });
  const fid = L.funcs.findIndex((f) => f[1] === entryName);
  if (fid < 0) throw new Error("no entry function " + entryName);
  const st = { frames: [], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
  const locals = [];
  for (let i = 0; i < L.nslots[fid]; i++) locals.push(AB.Undef());
  let dynParam = null;
  argSpecs.forEach((spec, i) => {
    if (spec.d) { locals[i] = AB.Dyn(RE.Var(i)); if (dynParam === null) dynParam = "v" + i; }
    else locals[i] = loadStatic(st, spec.s);
  });
  st.frames.push({ pc: L.entries[fid], func: fid, locals, ostack: [] });
  const client = {
    key: state.keyOf,
    clone: state.cloneState,
    point: (s) => s.frames.map((f) => f.pc).join(","),
    step: stepmod.step,
    whistle: (seen, cand) => whistle.whistle(seen, cand, L),
    generalize: whistle.generalize,
  };
  const program = engine.specialize(client, st);
  return ir.emitProgram(program, paramName || dynParam || "v0");
}

module.exports = { compile, specializeGeneral, loadStatic };
