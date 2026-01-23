// Test opaque handler with dynamic values
const decoder = new TextDecoder();
const buffer = new ArrayBuffer(8);
const view = new DataView(buffer);
const arr = new Uint8Array(buffer);

// Use dynamic to force residualization
if (input) {
  decoder;
} else {
  arr;
}
