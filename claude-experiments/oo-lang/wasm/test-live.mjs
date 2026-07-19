import { readFile } from 'node:fs/promises';
import { ScryWasm } from './scry-wasm.js';
const prog = `
class Point { x: Int
  y: Int
  fn sum() -> Int { self.x + self.y } }
fn main() { let p = Point(x: 3, y: 4) }
`;
let stdout = "";
const scry = await ScryWasm.instantiate(await readFile(new URL('./scry.wasm', import.meta.url)), {
  onStdout: t => stdout += t, onStderr: t => process.stderr.write("[err] "+t), vfs: { "/prog.scry": prog } });
console.log("boot:", Number(scry.exports.scry_boot(scry.writeStr("/prog.scry"))));
const ev = (src) => { const b = scry.enc.encode(src); const p = scry._malloc(b.length); scry.u8.set(b, p);
  const o = Number(scry.exports.scry_eval(p, BigInt(b.length))); const u = scry.u8; let e = o; while (u[e]) e++;
  return scry.dec.decode(u.subarray(o, e)); };
const count = () => JSON.parse(ev('Point.instances().len()')).value?.value;
console.log("instances before:", count());
console.log("allocate 2 more :", ev('Point(x: 10, y: 20)').slice(0, 60));
ev('Point(x: 1, y: 1)');
console.log("instances after :", count());
console.log("call a method   :", ev('Point.instances().get(0).sum()'));
console.log("live redefine   :", ev('fn double(n: Int) -> Int { n * 2 }'));
console.log("call new fn     :", ev('double(21)'));
