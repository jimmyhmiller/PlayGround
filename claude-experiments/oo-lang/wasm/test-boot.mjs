import { readFile } from 'node:fs/promises';
import { ScryWasm } from './scry-wasm.js';

const prog = `
class Point {
  x: Int
  y: Int
}
fn main() {
  let p = Point(x: 3, y: 4)
  Console.log("booted with a Point")
}
`;

let stdout = "";
const scry = await ScryWasm.instantiate(await readFile(new URL('./scry.wasm', import.meta.url)), {
  onStdout: t => stdout += t,
  onStderr: t => process.stderr.write("[stderr] " + t),
  vfs: { "/prog.scry": prog },
});

// boot
const pathPtr = scry.writeStr("/prog.scry");
const rc = Number(scry.exports.scry_boot(pathPtr));
console.log("scry_boot rc =", rc, rc === 0 ? "(OK)" : "(FAIL)");
console.log("stdout during boot:", JSON.stringify(stdout));

// eval helper
function ev(src) {
  const b = scry.enc.encode(src);
  const p = scry._malloc(b.length);
  scry.u8.set(b, p);
  const outPtr = Number(scry.exports.scry_eval(p, BigInt(b.length)));
  // read NUL-terminated JSON
  const u = scry.u8; let e = outPtr; while (u[e]) e++;
  return scry.dec.decode(u.subarray(outPtr, e));
}

console.log("eval 1+2        ->", ev("1 + 2"));
console.log("eval types()    ->", ev("types()"));
console.log("eval Point.instances().len() ->", ev("Point.instances().len()"));
