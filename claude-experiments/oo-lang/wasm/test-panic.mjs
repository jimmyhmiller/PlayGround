import { readFile } from 'node:fs/promises';
import { ScryWasm } from './scry-wasm.js';
const prog = `class Point { x: Int\n  y: Int }\nfn main() { let p = Point(x: 3, y: 4) }\n`;
const scry = await ScryWasm.instantiate(await readFile(new URL('./scry.wasm', import.meta.url)),
  { onStdout(){}, onStderr(){}, vfs: { "/p.scry": prog } });
console.log("boot:", scry.boot("/p.scry"));
console.log("healthy eval :", JSON.stringify(scry.eval("1 + 1")));
console.log("PANIC (oob)  :", JSON.stringify(scry.eval("Point.instances().get(99)")));
console.log("still alive  :", JSON.stringify(scry.eval("2 + 2")));
console.log("heap intact  :", JSON.stringify(scry.eval("Point.instances().len()")));
// hammer it: the instance must survive many panics
let ok = 0, panics = 0;
for (let i = 0; i < 200; i++) {
  const e = scry.eval("Point.instances().get(99)");
  if (e.error) panics++;
  if (scry.eval("1+1").value?.value === 2) ok++;
}
console.log(`after 200 panics: ${panics} typed errors, ${ok}/200 healthy evals between them`);
console.log("heap still intact:", JSON.stringify(scry.eval("Point.instances().len()")));

// The unwind must leave the shadow stack exactly where it started. Native's longjmp restores
// SP for us; on wasm the host does it via the exported __stack_pointer global. Any drift is a
// per-panic leak that eventually kills the instance (it used to die at ~9421 panics).
const sp = scry.instance.exports.__stack_pointer;
if (!sp) { console.log("FAIL: __stack_pointer not exported — panics leak the shadow stack"); process.exit(1); }
const base = sp.value;
for (let i = 0; i < 2000; i++) scry.eval("Point.instances().get(99)");
const drift = sp.value - base;
console.log(`shadow stack after 2000 more panics: drift=${drift}, unrestorable=${scry._spLeaked || 0}`);
if (drift !== 0 || (scry._spLeaked || 0) !== 0) { console.log("FAIL: shadow stack leaked"); process.exit(1); }
if (scry.eval("1+1").value?.value !== 2) { console.log("FAIL: VM unhealthy after panics"); process.exit(1); }
console.log("PASS: no shadow-stack drift; VM healthy");
