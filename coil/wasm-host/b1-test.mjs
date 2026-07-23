import fs from 'node:fs';
const main = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync('b1-main.wasm')));
const side = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync('b1-side.wasm')), {
  main: { memory: main.exports.memory, __t: main.exports.__t, host_double: main.exports.host_double }
});
// compiler writes a "boxed arg" (21) into shared memory at offset 0
const dv = new DataView(main.exports.memory.buffer);
dv.setBigInt64(0, 21n, true);
const rc = side.exports.run(0n);                 // run the "metaprogram"
console.log('metaprogram returned:', rc, '(expect 84 = 21*2*2)');
console.log('shared-memory writeback at [8]:', dv.getBigInt64(8, true), '(expect 84)');
console.log(rc === 84n && dv.getBigInt64(8,true) === 84n ? 'MECHANISM OK ✓' : 'MECHANISM FAILED');
