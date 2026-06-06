// step/index.js — the client `step`. Assembles the STEP table + dispatches on
// instr.tag. Handles two cross-cutting concerns the per-instr handlers don't:
//   (1) advancing pc after a straight-line instr, and
//   (2) forcing a block boundary when control falls through into a LEADER (a loop
//       head / branch join), so leaders become memoized points the engine can tie.
const meta = require("./meta.js");
const control = require("./control.js");
const STEP = {};
Object.assign(STEP, require("./arith.js"));
Object.assign(STEP, require("./mem.js"));
Object.assign(STEP, require("./control.js"));
Object.assign(STEP, require("./call.js"));

const STRAIGHTLINE = new Set(["PushNum", "PushStr", "PushBool", "PushUndef", "PushNull",
  "Load", "Store", "Bin", "Unary", "GetIndex", "GetProp", "SetIndexOp", "SetPropOp",
  "NewArray", "NewObject", "MakeClosure", "Dup", "Pop"]);

function currentInstr(s) { return meta.get().code[s.frames[s.frames.length - 1].pc]; }

function step(s, out, atEntry) {
  const top = s.frames[s.frames.length - 1];
  // (2) fell through into a leader mid-block -> end the block here (jump to it).
  if (!atEntry && meta.get().leaders.has(top.pc)) {
    return control.jumpTo(s, top.pc, out);
  }
  const instr = currentInstr(s);
  const f = STEP[instr.tag];
  if (!f) throw new Error("no STEP." + instr.tag);
  const r = f(s, instr, out, atEntry);
  if (r.tag === "Continue" && STRAIGHTLINE.has(instr.tag)) s.frames[s.frames.length - 1].pc++; // (1)
  return r;
}

module.exports = { STEP, step, setCode: (code) => meta.setMeta({ ...meta.get(), code }) };
