import { readFileSync } from "fs";
import loader from "@assemblyscript/loader";

// Persistent WASM runner: loads WASM once, then reads binary inputs from stdin
// and writes binary outputs to stdout in a loop.
//
// Protocol (all little-endian):
//   Request:  [n_inputs: u32] [size_0: u32] [data_0: f32×size_0] [size_1: u32] [data_1: f32×size_1] ...
//   Response: [n_outputs: u32] [data: f32×n_outputs]
//
// Reads requests until stdin closes.

const wasmPath = process.argv[2];
const wasmModule = await loader.instantiate(readFileSync(wasmPath));
const exports = wasmModule.exports;
const arrayId = exports.Float32Array_ID.value;

// Read exactly n bytes from stdin
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
        // Read number of inputs
        const headerBuf = await readBytes(stdin, 4);
        const nInputs = headerBuf.readUInt32LE(0);

        // Read each input array
        const ptrs = [];
        for (let i = 0; i < nInputs; i++) {
            const sizeBuf = await readBytes(stdin, 4);
            const size = sizeBuf.readUInt32LE(0);
            const dataBuf = await readBytes(stdin, size * 4);
            const f32 = new Float32Array(new Uint8Array(dataBuf).buffer);
            const ptr = exports.__newArray(arrayId, Array.from(f32));
            exports.__pin(ptr);
            ptrs.push(ptr);
        }

        // Run execute
        const result = exports.execute(...ptrs);

        // Unpin inputs
        for (const ptr of ptrs) exports.__unpin(ptr);

        // Get output
        const output = exports.__getFloat32Array(result);

        // Write response: [n_outputs: u32] [data: f32×n_outputs]
        const respHeader = Buffer.alloc(4);
        respHeader.writeUInt32LE(output.length, 0);
        process.stdout.write(respHeader);

        const respData = Buffer.from(output.buffer, output.byteOffset, output.byteLength);
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
