// Regression: (1) source("X") returns the REAL declaration text, not a skeleton;
//             (2) an instance field can be SET through the eval channel.
import { readFile } from 'node:fs/promises';
import { ScryWasm } from './scry-wasm.js';
const base = new URL('./', import.meta.url);
const prog = `class Point {
  x: Int
  y: Int
  fn sum() -> Int { self.x + self.y }
}
fn main() { let p = Point(x: 3, y: 4) }
`;
const s = await ScryWasm.instantiate(await readFile(new URL('scry.wasm', base)),
  { onStdout(){}, onStderr(){}, vfs: { "/p.scry": prog } });
s.boot("/p.scry");
let ok = true;
const check = (name, cond, extra="") => { console.log(`${cond ? "ok  " : "FAIL"} ${name}${extra ? " — " + extra : ""}`); if (!cond) ok = false; };

// (1) real source
const r = s.eval('source("Point")');
const src = r.value?.source || "";
check("source() found the decl", r.value?.found === true);
check("source() returns the REAL body", /self\.x \+ self\.y/.test(src), JSON.stringify(src.slice(0, 60)));
check("source() is not a skeleton", !/edit this body/.test(src));
check("source() spans the whole class", src.trimStart().startsWith("class Point") && src.trimEnd().endsWith("}"));

// (2) set a field through the wire op, exactly as the viewer's field editor does
check("field starts at 3", s.eval("Point.instances().get(0).x").value?.value === 3);
s.eval("Point.at(0, 0).x = 42");
check("field assignment took effect", s.eval("Point.instances().get(0).x").value?.value === 42);
check("running code sees the new value", s.eval("Point.instances().get(0).sum()").value?.value === 46);

console.log(ok ? "PASS test-source-edit" : "FAIL test-source-edit");
process.exit(ok ? 0 : 1);
