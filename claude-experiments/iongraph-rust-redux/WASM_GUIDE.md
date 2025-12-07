# WASM-Based Interactive Viewer

## Overview

IonGraph now supports two HTML generation modes:

1. **Static HTML** (`--html`): Pre-renders all passes as SVG. Large files (~18MB for mega-complex.json)
2. **WASM HTML** (`--wasm`): Renders graphs on-demand in the browser. Smaller files (~9MB for mega-complex.json, 51% reduction!)

## Quick Start

### 1. Build WASM Binary (One-time setup)

```bash
# Install wasm-pack (if not already installed)
cargo install wasm-pack

# Build WASM binary
wasm-pack build --target web --out-dir pkg
```

This creates:
- `pkg/iongraph_rust_redux_bg.wasm` - The WASM binary (256KB)
- `pkg/iongraph_rust_redux.js` - JavaScript glue code (11KB)

### 2. Generate WASM HTML

```bash
# Build the iongraph binary
cargo build --release --bin iongraph

# Generate WASM HTML
./target/release/iongraph --wasm ion-examples/mega-complex.json output.html
```

### 3. Open in Browser

Simply open the generated HTML file in any modern browser:

```bash
open output.html  # macOS
# or
xdg-open output.html  # Linux
# or
start output.html  # Windows
```

## Usage Comparison

### Static HTML (Pre-rendered)
```bash
./target/release/iongraph --html mega-complex.json output-static.html
```
- ✅ Instant display (no rendering time)
- ✅ Works offline
- ❌ Large file size (18MB for mega-complex.json)
- ❌ Slow to load over network

### WASM HTML (Client-side rendering)
```bash
./target/release/iongraph --wasm mega-complex.json output-wasm.html
```
- ✅ Smaller file size (9.2MB for mega-complex.json, 51% smaller!)
- ✅ Faster network transfer
- ✅ Works offline (WASM is embedded)
- ⚠️ First render takes ~100-200ms (then cached)
- ⚠️ Requires modern browser with WASM support

## File Size Comparison

| File | Passes | Static HTML | WASM HTML | Reduction |
|------|--------|-------------|-----------|-----------|
| array-access.json | 41 | 386KB | 541KB | -40% (overhead) |
| mega-complex.json | 526 | 18MB | 9.2MB | **51% smaller** |

**Key insight**: WASM is more efficient for large files with many passes. The 256KB WASM overhead is amortized across multiple passes.

## How It Works

### WASM HTML Structure

```html
<!DOCTYPE html>
<html>
<head>
  <title>IonGraph WASM Viewer</title>
  <style>/* CSS styles */</style>
</head>
<body>
  <div id="app">
    <div class="ig-sidebar">
      <select id="function-selector">...</select>
      <div id="pass-sidebar">...</div>
    </div>
    <div class="ig-viewport" id="viewport"></div>
  </div>

  <script>
    // Embedded IonJSON data
    const ION_DATA = {...};

    // Embedded WASM binary (base64)
    const WASM_BASE64 = "AGFzbQE...";

    // Decode base64 to bytes
    function base64ToBytes(base64) { ... }
  </script>

  <script type="module">
    // WASM JavaScript glue code
    // (from pkg/iongraph_rust_redux.js)

    // Initialize WASM
    const wasmBytes = base64ToBytes(WASM_BASE64);
    const { render_pass_svg, get_function_count, ... } = await wasm_bindgen(wasmBytes);

    // Interactive JavaScript
    async function renderPass(funcIdx, passIdx) {
      const svg = render_pass_svg(JSON.stringify(ION_DATA), funcIdx, passIdx);
      viewport.innerHTML = svg;
    }
  </script>
</body>
</html>
```

### Rendering Flow

1. **Page Load**: HTML loads instantly (~9MB)
2. **WASM Init**: Browser decodes and initializes WASM (~50ms)
3. **First Render**: User selects a pass, WASM generates SVG (~100ms)
4. **Cached**: Subsequent renders use cached SVG (instant)

## Features

### Interactive Navigation

- **Function Selector**: Dropdown to switch between functions
- **Pass Sidebar**: Click any pass to view its graph
- **Keyboard Shortcuts**:
  - `↑/↓`: Navigate functions
  - `←/→` or `r/f`: Navigate passes
- **Caching**: Rendered SVGs are cached for instant switching

### WASM API

The WASM module exposes these functions:

```javascript
// Render a specific pass to SVG string
render_pass_svg(ion_json: string, func_idx: number, pass_idx: number): string

// Get the number of functions
get_function_count(ion_json: string): number

// Get the number of passes for a function
get_pass_count(ion_json: string, func_idx: number): number

// Get function name
get_function_name(ion_json: string, func_idx: number): string

// Get pass name
get_pass_name(ion_json: string, func_idx: number, pass_idx: number): string
```

## Browser Compatibility

Works in all modern browsers with WebAssembly support:

- ✅ Chrome/Edge 57+
- ✅ Firefox 52+
- ✅ Safari 11+
- ✅ Opera 44+

## Troubleshooting

### Error: "WASM binary not found"

Make sure you've built the WASM binary first:

```bash
wasm-pack build --target web --out-dir pkg
```

### Error: "JSON parse error"

The IonJSON file might be malformed. Validate it with:

```bash
jq . your-file.json
```

### Slow rendering

First render of each pass takes ~100ms. Subsequent renders are instant (cached). This is expected behavior.

## Development

### Rebuild WASM after changes

```bash
# Rebuild WASM
wasm-pack build --target web --out-dir pkg

# Rebuild iongraph binary
cargo build --release --bin iongraph

# Generate new HTML
./target/release/iongraph --wasm your-file.json output.html
```

### Optimize WASM size

The WASM binary is already optimized with:
- `opt-level = "z"` - Optimize for size
- `lto = true` - Link-time optimization
- `wasm-opt -Oz` - Post-processing optimization (automatic with wasm-pack)

Current WASM size: **256KB** (compressed: ~80KB with gzip)

## Future Enhancements

Possible improvements:

1. **WebGPU rendering**: Use GPU for faster rendering of complex graphs
2. **Streaming WASM**: Load WASM progressively
3. **External WASM**: Serve WASM separately (avoid base64 overhead)
4. **Service Worker**: Cache WASM for offline use
5. **Progressive rendering**: Show partial graphs while rendering

## Comparison with TypeScript Version

The WASM implementation is **byte-for-byte identical** to the TypeScript SVG output:
- ✅ 157/157 test cases passed
- ✅ Pixel-perfect rendering
- ✅ Same layout algorithms
- ✅ Same arrow routing

See [TEST_RESULTS_FINAL.md](TEST_RESULTS_FINAL.md) for validation details.
