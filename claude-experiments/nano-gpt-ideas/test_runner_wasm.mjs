// Raw WASM test runner (no AssemblyScript runtime).
//
// Usage: node test_runner_wasm.mjs <wasm> <inputs_json>
//
// The WASM module exports:
//   memory: WebAssembly.Memory
//   execute(dim_params..., input_ptrs...) -> i32 (output byte offset)
//
// This runner:
// 1. Instantiates the WASM module.
// 2. Writes a heap pointer at byte 0 of memory.
// 3. Copies input arrays into linear memory.
// 4. Calls execute() with the input pointers.
// 5. Reads the output from the returned pointer.

import { readFileSync } from "fs";

const wasmPath = process.argv[2];
const inputsJson = process.argv[3];

const wasmBytes = readFileSync(wasmPath);
const { instance } = await WebAssembly.instantiate(wasmBytes);
const exports = instance.exports;
const memory = exports.memory;

const inputs = JSON.parse(inputsJson);

// The last element may be an output_size integer
const outputSize = inputs.pop();

// Start laying out data at offset 16 (leave room for heap ptr at byte 0)
let offset = 16;

// Align to 16 bytes
function align16(n) {
    return (n + 15) & ~15;
}

// Ensure memory is large enough
function ensureMemory(neededBytes) {
    const currentBytes = memory.buffer.byteLength;
    if (neededBytes > currentBytes) {
        const neededPages = Math.ceil((neededBytes - currentBytes) / 65536);
        memory.grow(neededPages);
    }
}

// Copy inputs into WASM linear memory
const inputPtrs = [];
for (const arr of inputs) {
    offset = align16(offset);
    const byteLen = arr.length * 4;
    ensureMemory(offset + byteLen);
    const view = new Float32Array(memory.buffer, offset, arr.length);
    for (let i = 0; i < arr.length; i++) {
        view[i] = arr[i];
    }
    inputPtrs.push(offset);
    offset += byteLen;
}

// Set heap pointer past all inputs (at byte 0)
offset = align16(offset);
ensureMemory(offset + 4);
new DataView(memory.buffer).setInt32(0, offset, true);

// Grow memory generously for intermediate buffers
const extraPages = 1024; // ~64MB extra
try { memory.grow(extraPages); } catch(e) { /* ok */ }

// Call execute
const resultPtr = exports.execute(...inputPtrs);

// Read output: we need to know the output size
// The host passes it as the last element of the inputs array
const outView = new Float32Array(memory.buffer, resultPtr, outputSize);
const result = Array.from(outView);

console.log(JSON.stringify(result));
