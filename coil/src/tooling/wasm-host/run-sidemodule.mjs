// run-sidemodule.mjs — instantiate a codegen_wasm SIDE-MODULE (built with
// --wasm-side-module) against a SHARED memory64, mirroring src/tooling/wasm-host/b1-test.mjs but
// with a codegen_wasm-emitted module. Proves the PIC contract:
//   - the module imports env.memory / env.__memory_base / env.__stack_pointer,
//   - its static data (a string literal) is placed at __memory_base by the active
//     data segment `(data (global.get $__memory_base) …)`,
//   - an exported entry reads a pointer arg out of the SHARED memory and writes back.
//
// Layout the host picks in the shared heap (memory64 ⇒ all addresses are BigInt):
//   arg cell      @ 0x100000 (1 MiB)
//   __memory_base @ 0x200000 (2 MiB)  — where the module's data lands
//   __stack_pointer(top) @ 0xF00000 (15 MiB) — shadow stack grows down from here
import fs from 'node:fs';

const path = process.argv[2] ?? '/tmp/m4_side.wasm';
const ARG_ADDR = 0x100000n;
const MEMORY_BASE = 0x200000n;
const STACK_TOP = 0xF00000n;

const memory = new WebAssembly.Memory({ initial: 256n, address: 'i64' }); // 16 MiB shared memory64 heap
const env = {
  memory,
  __memory_base: new WebAssembly.Global({ value: 'i64', mutable: false }, MEMORY_BASE),
  __stack_pointer: new WebAssembly.Global({ value: 'i64', mutable: true }, STACK_TOP),
  // stubs, in case a metaprogram module also imports me-host/libc functions:
  free: (_p) => {},
  malloc: (n) => { const p = 0x800000n; return p; },
};

const dv = new DataView(memory.buffer);
dv.setBigInt64(Number(ARG_ADDR), 7n, true);   // host writes the arg (7) into shared memory

const inst = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(path)), { env });

// 1) the string data landed at __memory_base (PIC): first byte should be 'Q' (81)
const dataByte = dv.getUint8(Number(MEMORY_BASE));
// 2) call the exported entry with a POINTER into shared memory
const ret = inst.exports.mp_run(ARG_ADDR);
// 3) the entry wrote arg+81 back into *p
const writeback = dv.getBigInt64(Number(ARG_ADDR), true);

console.log(`data@__memory_base = ${dataByte} (expect 81 = 'Q')`);
console.log(`mp_run(&arg) returned = ${ret} (expect 7081 = 7*1000+81)`);
console.log(`*arg after call = ${writeback} (expect 88 = 7+81)`);
const ok = dataByte === 81 && ret === 7081n && writeback === 88n;
console.log(ok ? 'SIDE-MODULE OK ✓' : 'SIDE-MODULE FAILED ✗');
process.exit(ok ? 0 : 1);
