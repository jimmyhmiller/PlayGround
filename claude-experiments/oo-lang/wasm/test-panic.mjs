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
let ok = 0;
for (let i = 0; i < 50; i++) { scry.eval("Point.instances().get(99)"); if (scry.eval("1+1").value.value === 2) ok++; }
console.log(`after 50 panics: ${ok}/50 healthy evals; sp-leaks=${scry._spLeaked || 0}`);
