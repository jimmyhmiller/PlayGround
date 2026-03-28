import { readFileSync } from "fs";

// Check if the input binary has any NaN or Inf values
const path = process.argv[2] || "/var/folders/b3/821wsm8x1vgg7xqhxhqgtftc0000gn/T/train_inputs.bin";
const raw = readFileSync(path);
const buf = new Float32Array(new Uint8Array(raw).buffer);

let nanCount = 0, infCount = 0;
for (let i = 0; i < buf.length; i++) {
    if (isNaN(buf[i])) nanCount++;
    if (!isFinite(buf[i]) && !isNaN(buf[i])) infCount++;
}

console.log(`Total elements: ${buf.length}`);
console.log(`NaN: ${nanCount}, Inf: ${infCount}`);
if (nanCount > 0 || infCount > 0) {
    // Find first NaN/Inf
    for (let i = 0; i < buf.length; i++) {
        if (!isFinite(buf[i])) {
            console.log(`First bad value at index ${i}: ${buf[i]}`);
            break;
        }
    }
}
