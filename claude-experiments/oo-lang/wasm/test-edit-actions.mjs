// Regression: editing a class that has `action` blocks must round-trip.
// `action "L" for T {…}` desugars to a hidden __action_<i> method ON T, so the class as
// RUNNING has members no source file contains. Re-applying T's source must not look like
// those methods were removed, and the actions must still work afterwards.
import { readFile } from 'node:fs/promises';
import { ScryWasm } from './scry-wasm.js';
const base = new URL('./', import.meta.url);
const prog = `class Counter {
  n: Int
  fn bump() { self.n = self.n + 1 }
  fn label() -> String { "count" }
}

action "Bump" for Counter { self.bump() }
action "Reset" for Counter { self.n = 0 }

fn main() { let c = Counter(n: 0) }
`;
const s = await ScryWasm.instantiate(await readFile(new URL('scry.wasm', base)),
  { onStdout(){}, onStderr(){}, vfs: { "/c.scry": prog } });
s.boot("/c.scry");
let ok = true;
const check = (n, c, extra="") => { console.log(`${c?"ok  ":"FAIL"} ${n}${extra?" — "+extra:""}`); if(!c) ok=false; };
const n = () => s.eval("Counter.instances().get(0).n").value?.value;
const actionsFor = () => { const v = s.eval("actions()").value; const a = Array.isArray(v) ? v : (v && (v.actions || v.items)) || []; return a.filter(x=>x.target==="Counter").map(x=>x.invoke); };

check("two actions registered", actionsFor().length === 2, JSON.stringify(actionsFor()));
s.eval("Counter.at(0,0).__action_0()");
check("action works before the edit", n() === 1, "n=" + n());

// edit the class body (change label()) and re-apply — this used to fail with
// "method '__action_0' was removed while live"
const so = s.eval('source("Counter")').value;
check("source() is the real class", /fn bump/.test(so.source) && !/edit this body/.test(so.source));
const edited = so.source.replace('"count"', '"edited-count"');
const r = s.eval((so.module ? `module ${so.module}\n\n` : "") + edited);
check("edit applies (actions not seen as removed)", !r.error, JSON.stringify(r.error || r.value?.message).slice(0, 90));
check("the edit took effect", s.eval("Counter.instances().get(0).label()").value?.value === "edited-count");
check("actions survived the edit", actionsFor().length === 2, JSON.stringify(actionsFor()));
s.eval("Counter.at(0,0).__action_0()");
check("action still WORKS after the edit", n() === 2, "n=" + n());
s.eval("Counter.at(0,0).__action_1()");
check("second action still works", n() === 0, "n=" + n());

console.log(ok ? "PASS test-edit-actions" : "FAIL test-edit-actions");
process.exit(ok ? 0 : 1);
