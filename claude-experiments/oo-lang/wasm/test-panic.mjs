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
