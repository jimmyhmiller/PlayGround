import { readFileSync } from "fs";
import loader from "@assemblyscript/loader";

const wasmPath = process.argv[2];
const inputsJson = process.argv[3];

const wasmModule = await loader.instantiate(readFileSync(wasmPath));
const exports = wasmModule.exports;

const inputs = JSON.parse(inputsJson);
const arrayId = exports.Float32Array_ID.value;

const wasmInputs = inputs.map(arr => exports.__newArray(arrayId, arr));
const result = exports.execute(...wasmInputs);
const output = exports.__getFloat32Array(result);

console.log(JSON.stringify(Array.from(output)));
