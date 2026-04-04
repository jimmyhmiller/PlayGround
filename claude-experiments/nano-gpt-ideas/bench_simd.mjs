// Benchmark WASM SIMD vs scalar for GPT-2 inference.
//
// Usage: node bench_simd.mjs <simd.wasm> <scalar.wasm> <inputs.bin> <manifest.json> [warmup] [iters]

import { readFileSync } from "fs";

const simdPath = process.argv[2];
const scalarPath = process.argv[3];
const inputsBinPath = process.argv[4];
const manifestPath = process.argv[5];
const warmup = parseInt(process.argv[6] || "2");
const iters = parseInt(process.argv[7] || "5");

const manifestRaw = JSON.parse(readFileSync(manifestPath, "utf-8"));
const inputsBuf = readFileSync(inputsBinPath);
const dimParams = manifestRaw.dim_params || [];
const inputsManifest = manifestRaw.inputs;
const outputSize = manifestRaw.output_size;

function align16(n) { return (n + 15) & ~15; }

async function loadAndRun(wasmPath, label) {
    const wasmBytes = readFileSync(wasmPath);
    const { instance } = await WebAssembly.instantiate(wasmBytes);
    const exports = instance.exports;
    const memory = exports.memory;

    function setup() {
        // Copy inputs into linear memory
        let offset = 16;
        const inputPtrs = [];
        let binOffset = 0;

        for (const entry of inputsManifest) {
            const n = entry.n_elements;
            offset = align16(offset);
            const byteLen = n * 4;
            const currentBytes = memory.buffer.byteLength;
            if (offset + byteLen > currentBytes) {
                memory.grow(Math.ceil((offset + byteLen - currentBytes) / 65536));
            }
            const src = new Uint8Array(inputsBuf.buffer, inputsBuf.byteOffset + binOffset, byteLen);
            new Uint8Array(memory.buffer).set(src, offset);
            inputPtrs.push(offset);
            binOffset += byteLen;
            offset += byteLen;
        }

        // Set heap pointer
        offset = align16(offset);
        const neededBytes = offset + 4;
        if (neededBytes > memory.buffer.byteLength) {
            memory.grow(Math.ceil((neededBytes - memory.buffer.byteLength) / 65536));
        }
        new DataView(memory.buffer).setInt32(0, offset, true);

        // Grow memory for intermediates
        try { memory.grow(1024); } catch(e) {}

        return inputPtrs;
    }

    // Warmup
    for (let i = 0; i < warmup; i++) {
        const ptrs = setup();
        exports.execute(...dimParams, ...ptrs);
    }

    // Timed runs
    const times = [];
    let resultPtr;
    for (let i = 0; i < iters; i++) {
        const ptrs = setup();
        const t0 = performance.now();
        resultPtr = exports.execute(...dimParams, ...ptrs);
        const elapsed = performance.now() - t0;
        times.push(elapsed);
    }

    // Read output for verification
    const outView = new Float32Array(memory.buffer, resultPtr, Math.min(outputSize, 10));
    const sample = Array.from(outView).map(v => v.toFixed(4));

    const mean = times.reduce((a, b) => a + b, 0) / times.length;
    const min = Math.min(...times);
    const max = Math.max(...times);

    console.log(`${label}:`);
    console.log(`  mean: ${mean.toFixed(1)}ms  min: ${min.toFixed(1)}ms  max: ${max.toFixed(1)}ms  (${iters} runs, ${warmup} warmup)`);
    console.log(`  sample output: [${sample.join(", ")}]`);

    return { mean, min, times };
}

console.log(`GPT-2 WASM Benchmark (T=${dimParams[0] || "?"}, ${warmup} warmup, ${iters} iters)\n`);

const simdResult = await loadAndRun(simdPath, "SIMD");
console.log();
const scalarResult = await loadAndRun(scalarPath, "Scalar");

console.log();
const speedup = scalarResult.mean / simdResult.mean;
console.log(`Speedup: ${speedup.toFixed(2)}x (SIMD mean / Scalar mean)`);
