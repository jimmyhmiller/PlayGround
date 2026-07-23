// run-standalone.mjs — instantiate a standalone (no-import) memory64 Coil module
// produced by the native wasm backend (--backend wasm) and call its exported `main`.
// Node v26 has memory64 on by default. Usage: node run-standalone.mjs <file.wasm>
import fs from 'node:fs';

const path = process.argv[2];
if (!path) { console.error('usage: node run-standalone.mjs <file.wasm>'); process.exit(2); }

const bytes = fs.readFileSync(path);
const mod = new WebAssembly.Module(bytes);
// A standalone Coil module declares no imports; pass an empty import object.
const inst = new WebAssembly.Instance(mod, {});
if (typeof inst.exports.main !== 'function') {
  console.error('module has no exported main');
  process.exit(2);
}
const r = inst.exports.main();
// memory64 => i64 results come back as BigInt.
console.log(String(r));
