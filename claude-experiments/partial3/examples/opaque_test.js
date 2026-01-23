// Test opaque handler functionality
const decoder = new TextDecoder();
const buffer = new ArrayBuffer(8);
const view = new DataView(buffer);
const arr = new Uint8Array(buffer);

// These should be replaced with placeholder variables when --builtins is used
decoder;
