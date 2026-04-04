// Raw WASM test runner with binary inputs (no AssemblyScript runtime).
//
// Usage: node test_runner_wasm_bin.mjs <wasm> <inputs.bin> <manifest.json>
//
// manifest.json format:
//   { "dim_params": [T, ...], "inputs": [{"n_elements": N}, ...], "output_size": M }
// or old format:
//   [{"n_elements": N}, ...]

import { readFileSync } from "fs";

const wasmPath = process.argv[2];
const inputsBinPath = process.argv[3];
const manifestPath = process.argv[4];

const wasmBytes = readFileSync(wasmPath);
const { instance } = await WebAssembly.instantiate(wasmBytes);
const exports = instance.exports;
const memory = exports.memory;

const manifestRaw = JSON.parse(readFileSync(manifestPath, "utf-8"));
const inputsBuf = readFileSync(inputsBinPath);

const dimParams = Array.isArray(manifestRaw) ? [] : (manifestRaw.dim_params || []);
const inputsManifest = Array.isArray(manifestRaw) ? manifestRaw : manifestRaw.inputs;
const outputSize = manifestRaw.output_size || 0;

function align16(n) { return (n + 15) & ~15; }

function ensureMemory(neededBytes) {
    const currentBytes = memory.buffer.byteLength;
    if (neededBytes > currentBytes) {
        const neededPages = Math.ceil((neededBytes - currentBytes) / 65536);
        memory.grow(neededPages);
    }
}

// Copy inputs into linear memory starting at offset 16
let offset = 16;
const inputPtrs = [];
let binOffset = 0;

for (const entry of inputsManifest) {
    const n = entry.n_elements;
    offset = align16(offset);
    const byteLen = n * 4;
    ensureMemory(offset + byteLen);
    // Copy bytes directly
    const src = new Uint8Array(inputsBuf.buffer, inputsBuf.byteOffset + binOffset, byteLen);
    new Uint8Array(memory.buffer).set(src, offset);
    inputPtrs.push(offset);
    binOffset += byteLen;
}

// Set heap pointer
offset = align16(offset + inputsManifest[inputsManifest.length - 1].n_elements * 4);
ensureMemory(offset + 4);
new DataView(memory.buffer).setInt32(0, offset, true);

// Grow memory for intermediates
try { memory.grow(1024); } catch(e) {}

console.error(`Loaded ${inputsManifest.length} inputs${dimParams.length ? ` + ${dimParams.length} dim params` : ''}, running execute()...`);
const t0 = performance.now();
const resultPtr = exports.execute(...dimParams, ...inputPtrs);
const elapsed = performance.now() - t0;
console.error(`execute() took ${(elapsed / 1000).toFixed(1)}s`);

const outView = new Float32Array(memory.buffer, resultPtr, outputSize);
console.log(JSON.stringify(Array.from(outView)));
