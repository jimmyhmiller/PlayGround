import { readFileSync } from "fs";
import loader from "@assemblyscript/loader";

const wasmModule = await loader.instantiate(readFileSync("repro_test.wasm"));
const exports = wasmModule.exports;
const arrayId = exports.Float32Array_ID.value;

// Run 1000 times to check for intermittent failures
let failures = 0;
for (let run = 0; run < 1000; run++) {
    const input = new Float32Array(100);
    for (let i = 0; i < 100; i++) input[i] = Math.random() * run;

    const inputPtr = exports.__newArray(arrayId, Array.from(input));
    exports.__pin(inputPtr);

    const resultPtr = exports.test_return(inputPtr);
    exports.__unpin(inputPtr);

    const output = exports.__getFloat32Array(resultPtr);

    if (output.length !== 100000) {
        console.log(`FAIL run ${run}: got ${output.length} elements, expected 100000`);
        failures++;
        if (failures >= 5) break;
    }
}

if (failures === 0) {
    console.log("PASS: all 1000 runs returned correct size");
} else {
    console.log(`FAIL: ${failures} failures`);
}
