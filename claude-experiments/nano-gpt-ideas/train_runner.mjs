import { readFileSync, writeFileSync } from "fs";
import loader from "@assemblyscript/loader";

// Usage: node train_runner.mjs <wasm> <inputs.bin> <manifest.json> <output.bin>
// Like test_runner_bin.mjs but writes output as raw f32 binary instead of JSON.

const wasmPath = process.argv[2];
const inputsBinPath = process.argv[3];
const manifestPath = process.argv[4];
const outputBinPath = process.argv[5];

const wasmModule = await loader.instantiate(readFileSync(wasmPath));
const exports = wasmModule.exports;

const manifestRaw = JSON.parse(readFileSync(manifestPath, "utf-8"));
// Copy to a fresh ArrayBuffer to guarantee alignment (Node Buffers can have
// non-zero byteOffset into a shared pool, which breaks Float32Array views).
const rawBuf = readFileSync(inputsBinPath);
const inputsBuf = new Uint8Array(rawBuf).buffer;

const dimParams = Array.isArray(manifestRaw) ? [] : (manifestRaw.dim_params || []);
const inputsManifest = Array.isArray(manifestRaw) ? manifestRaw : manifestRaw.inputs;

const arrayId = exports.Float32Array_ID.value;
const wasmInputs = [];
let offset = 0;

// Verify total expected size
const totalExpected = inputsManifest.reduce((sum, e) => sum + e.n_elements * 4, 0);
if (inputsBuf.byteLength !== totalExpected) {
    console.error(`ERROR: Binary file has ${inputsBuf.byteLength} bytes but manifest expects ${totalExpected}`);
    process.exit(1);
}

for (let idx = 0; idx < inputsManifest.length; idx++) {
    const n = inputsManifest[idx].n_elements;
    const f32arr = new Float32Array(inputsBuf, offset, n);
    const ptr = exports.__newArray(arrayId, Array.from(f32arr));
    exports.__pin(ptr);
    wasmInputs.push(ptr);
    offset += n * 4;
}

const result = exports.execute(...dimParams, ...wasmInputs);

for (const ptr of wasmInputs) exports.__unpin(ptr);

const output = exports.__getFloat32Array(result);

// Write as raw little-endian f32 binary
const buf = Buffer.from(output.buffer, output.byteOffset, output.byteLength);
writeFileSync(outputBinPath, buf);

// Print the element count to stdout so Rust can verify
console.log(output.length);
