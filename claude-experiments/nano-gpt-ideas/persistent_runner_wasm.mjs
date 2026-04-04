// Persistent raw WASM runner: loads WASM once, then reads binary inputs from stdin
// and writes binary outputs to stdout in a loop. No AssemblyScript runtime.
//
// Protocol (all little-endian):
//   Request:  [n_inputs: u32] [output_size: u32] [size_0: u32] [data_0: f32×size_0] ...
//   Response: [n_outputs: u32] [data: f32×n_outputs]
//
// The output_size field tells us how many f32s to read from the returned pointer.

import { readFileSync } from "fs";

const wasmPath = process.argv[2];
const wasmBytes = readFileSync(wasmPath);
const { instance } = await WebAssembly.instantiate(wasmBytes);
const exports = instance.exports;
const memory = exports.memory;

function align16(n) { return (n + 15) & ~15; }

function ensureMemory(neededBytes) {
    const currentBytes = memory.buffer.byteLength;
    if (neededBytes > currentBytes) {
        const neededPages = Math.ceil((neededBytes - currentBytes) / 65536);
        memory.grow(neededPages);
    }
}

// Pre-grow memory generously
try { memory.grow(2048); } catch(e) {}

function readBytes(stream, n) {
    return new Promise((resolve, reject) => {
        const chunks = [];
        let got = 0;
        function tryRead() {
            while (got < n) {
                const remaining = n - got;
                const chunk = stream.read(remaining);
                if (chunk === null) {
                    stream.once('readable', tryRead);
                    return;
                }
                chunks.push(chunk);
                got += chunk.length;
            }
            resolve(Buffer.concat(chunks, n));
        }
        stream.once('end', () => reject(new Error('stdin closed')));
        tryRead();
    });
}

const stdin = process.stdin;

try {
    while (true) {
        // Read n_inputs and output_size
        const headerBuf = await readBytes(stdin, 8);
        const nInputs = headerBuf.readUInt32LE(0);
        const outputSize = headerBuf.readUInt32LE(4);

        // Read each input and copy into WASM linear memory
        let offset = 16; // skip heap ptr area
        const inputPtrs = [];

        for (let i = 0; i < nInputs; i++) {
            const sizeBuf = await readBytes(stdin, 4);
            const size = sizeBuf.readUInt32LE(0);
            const dataBuf = await readBytes(stdin, size * 4);

            offset = align16(offset);
            const byteLen = size * 4;
            ensureMemory(offset + byteLen);
            new Uint8Array(memory.buffer).set(new Uint8Array(dataBuf.buffer, dataBuf.byteOffset, byteLen), offset);
            inputPtrs.push(offset);
            offset += byteLen;
        }

        // Set heap pointer
        offset = align16(offset);
        ensureMemory(offset + 4);
        new DataView(memory.buffer).setInt32(0, offset, true);

        // Run execute
        const resultPtr = exports.execute(...inputPtrs);

        // Read output
        const output = new Float32Array(memory.buffer, resultPtr, outputSize);

        // Write response
        const respHeader = Buffer.alloc(4);
        respHeader.writeUInt32LE(output.length, 0);
        process.stdout.write(respHeader);

        const respData = Buffer.alloc(output.length * 4);
        for (let i = 0; i < output.length; i++) {
            respData.writeFloatLE(output[i], i * 4);
        }
        process.stdout.write(respData);
    }
} catch (e) {
    if (e.message === 'stdin closed') {
        // Normal exit
    } else {
        console.error('Runner error:', e.message);
        process.exit(1);
    }
}
