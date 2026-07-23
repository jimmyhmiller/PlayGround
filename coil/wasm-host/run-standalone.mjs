// run-standalone.mjs — instantiate a standalone memory64 Coil module produced by the
// native wasm backend (--backend wasm) and call its exported `main`. Node v26 has
// memory64 on by default. Usage: node run-standalone.mjs <file.wasm>
//
// Provides a small `env.*` import surface for testing extern calls (memory64 ⇒ i64
// args/results are BigInt in JS):
//   env.host_add(a,b)  -> a+b            (a simple test callback)
//   env.host_sub(a,b)  -> a-b
//   env.malloc(n)      -> bump allocator over the module's linear memory (heap at 8 MiB)
//   env.free(p)        -> no-op          (bump allocator never reclaims)
// Any import the module declares that isn't listed throws a clear error.
import fs from 'node:fs';

const path = process.argv[2];
if (!path) { console.error('usage: node run-standalone.mjs <file.wasm>'); process.exit(2); }

let heap = 0x800000n;                 // 8 MiB — above data (1KiB+) and below the shadow stack (grows down from 16 MiB)
const known = {
  host_add: (a, b) => a + b,
  host_sub: (a, b) => a - b,
  free: (_p) => {},
  malloc: (n) => { const p = heap; heap = (heap + BigInt(n) + 15n) & ~15n; return p; },
};
const env = new Proxy(known, {
  get(t, k) {
    if (k in t) return t[k];
    return (...args) => { throw new Error(`module called unprovided import env.${String(k)}(${args})`); };
  },
});

const bytes = fs.readFileSync(path);
const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), { env });
if (typeof inst.exports.main !== 'function') { console.error('module has no exported main'); process.exit(2); }
// memory64 => i64 results come back as BigInt.
console.log(String(inst.exports.main()));
