import { readFileSync } from "fs";
import loader from "@assemblyscript/loader";

// Usage: node test_runner_bin.mjs <wasm> <inputs.bin> <manifest.json>
// manifest.json: array of { n_elements: number } describing each input tensor
// inputs.bin: all tensors concatenated as little-endian f32

const wasmPath = process.argv[2];
const inputsBinPath = process.argv[3];
const manifestPath = process.argv[4];

const wasmModule = await loader.instantiate(readFileSync(wasmPath));
const exports = wasmModule.exports;

const manifest = JSON.parse(readFileSync(manifestPath, "utf-8"));
const inputsBuf = readFileSync(inputsBinPath);

const arrayId = exports.Float32Array_ID.value;
const wasmInputs = [];
let offset = 0;

for (let idx = 0; idx < manifest.length; idx++) {
    const n = manifest[idx].n_elements;
    const f32arr = new Float32Array(inputsBuf.buffer, inputsBuf.byteOffset + offset, n);
    const ptr = exports.__newArray(arrayId, Array.from(f32arr));
    exports.__pin(ptr);  // prevent GC from collecting during later allocations
    wasmInputs.push(ptr);
    offset += n * 4;
}

console.error(`Loaded ${manifest.length} inputs, running execute()...`);
const t0 = performance.now();
const result = exports.execute(...wasmInputs);
const elapsed = performance.now() - t0;
console.error(`execute() took ${(elapsed / 1000).toFixed(1)}s`);
// Unpin inputs
for (const ptr of wasmInputs) exports.__unpin(ptr);

const output = exports.__getFloat32Array(result);
console.log(JSON.stringify(Array.from(output)));
