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
    point: (s) => s.frames.map((f) => f.pc).join(","),
    step: stepmod.step,
    whistle: (seen, cand) => whistle.whistle(seen, cand, L),
    generalize: whistle.generalize,
  };
  const program = engine.specialize(client, L.initState);
  return ir.emitProgram(program, paramName || L.paramName || "v0");
}

module.exports = { compile };
