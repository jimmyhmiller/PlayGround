// WebAssembly `?init` runtime helper.
//
// Ported from Vite's `src/node/plugins/wasm.ts` (MIT License, (c) 2019-present
// VoidZero Inc. and Vite contributors). The public URL is the emitted (or
// inlined `data:`) `.wasm` asset; the returned promise resolves to a live
// `WebAssembly.Instance`. Instantiate-streaming is used when the fetched
// response is served with `application/wasm`, otherwise the bytes are buffered
// and instantiated. A `data:` URL is base64-decoded and instantiated directly.
//
// Diffpack embeds this text verbatim into each synthesized `?init` module via
// `include_str!`, so the guest code lives in a real file rather than a Rust
// string literal.
async function __diffpackWasmInit(imports, url) {
  let result;
  if (url.startsWith("data:")) {
    const urlContent = url.replace(/^data:.*?base64,/, "");
    let bytes;
    if (typeof Buffer === "function" && typeof Buffer.from === "function") {
      bytes = Buffer.from(urlContent, "base64");
    } else if (typeof atob === "function") {
      const binaryString = atob(urlContent);
      const size = binaryString.length;
      bytes = new Uint8Array(size);
      for (let i = 0; i < size; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
    } else {
      throw new Error(
        "Failed to decode base64-encoded data URL, Buffer and atob are not supported",
      );
    }
    result = await WebAssembly.instantiate(bytes, imports);
  } else {
    const response = await fetch(url);
    const contentType = response.headers.get("Content-Type") || "";
    if (
      "instantiateStreaming" in WebAssembly &&
      contentType.startsWith("application/wasm")
    ) {
      result = await WebAssembly.instantiateStreaming(response, imports);
    } else {
      const buffer = await response.arrayBuffer();
      result = await WebAssembly.instantiate(buffer, imports);
    }
  }
  return result.instance;
}
