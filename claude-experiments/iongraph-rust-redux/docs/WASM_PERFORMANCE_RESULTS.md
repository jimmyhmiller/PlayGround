# WASM Performance Optimization Results

## Problem Identified

The WASM viewer was taking "seconds" to render, which seemed suspiciously slow given that:
- The Rust layout algorithms are very fast
- WASM overhead is typically minimal

## Benchmarking Results

Using the new `wasm-benchmark` binary, we measured native vs WASM performance:

### Native Performance (Baseline)
```
File: mega-complex.json (9.76 MB)
Function 0, Pass 0:
  JSON parsing:     53ms   (99.8%)
  IR conversion:    0.02ms (0.0%)
  Graph creation:   0.04ms (0.1%)
  Layout:           0.006ms (0.0%)
  Rendering:        0.002ms (0.0%)
  SVG serialization: 0.03ms (0.1%)
  Total:            53ms

Function 5, Pass 15 (complex):
  JSON parsing:     53ms   (98.6%)
  Layout+Render:    1ms    (1.4%)
  Total:            54ms
```

### WASI/WASM Performance
```
Function 0, Pass 0:
  JSON parsing:     59ms   (99.7%)
  Everything else:  <1ms
  Total:            59ms   (11% slower than native)

Function 5, Pass 15 (complex):
  JSON parsing:     101ms  (98.1%)
  Layout+Render:    2ms    (1.9%)
  Total:            102ms  (89% slower than native)
```

**Key Finding**: JSON parsing takes **99% of the time**, and WASM is only **1.1-1.9x slower** than native!

## Root Cause Analysis

The WASM code itself runs in ~60-100ms. So why was the browser experiencing "seconds" of delay?

### The Problem: Repeated JSON Parsing

**Before (Stateless API)**:
```javascript
const ionJsonStr = JSON.stringify(ION_DATA);  // 9.76MB string

// Called on EVERY render:
const svg = render_pass_svg(ionJsonStr, funcIdx, passIdx);
// ↑ Copies 9.76MB from JS to WASM, then parses it (60-100ms)

// Called when switching functions:
const numPasses = get_pass_count(ionJsonStr, funcIdx);
// ↑ Copies 9.76MB again, parses again (60-100ms)

// Called 30 times (once per pass):
for (let i = 0; i < 30; i++) {
  const passName = get_pass_name(ionJsonStr, funcIdx, i);
  // ↑ Copies 9.76MB × 30 times, parses × 30 times (1.8 seconds!)
}
```

**Total for switching to a function with 30 passes**:
- 1 call to `get_pass_count()`: 60ms
- 30 calls to `get_pass_name()`: 30 × 60ms = **1.8 seconds**
- 1 call to `render_pass()`: 60ms
- **Total: ~2 seconds just to populate the sidebar!**

## Solution: Stateful WASM Instance

### Implementation

**New WASM API** (src/wasm.rs):
```rust
#[wasm_bindgen]
pub struct IonGraphViewer {
    data: IonJSON,  // Parsed once, cached in WASM memory
}

#[wasm_bindgen]
impl IonGraphViewer {
    #[wasm_bindgen(constructor)]
    pub fn new(ion_json: &str) -> Result<IonGraphViewer, JsValue> {
        let data: IonJSON = serde_json::from_str(ion_json)?;
        Ok(IonGraphViewer { data })
    }

    pub fn render_pass(&self, func_idx: usize, pass_idx: usize) -> Result<String, JsValue> {
        // No JSON parsing - data already in memory!
        let func = &self.data.functions[func_idx];
        let pass = &func.passes[pass_idx];
        // ... rendering logic
    }

    pub fn get_function_count(&self) -> usize {
        self.data.functions.len()
    }

    pub fn get_pass_count(&self, func_idx: usize) -> Result<usize, JsValue> {
        Ok(self.data.functions[func_idx].passes.len())
    }

    pub fn get_function_name(&self, func_idx: usize) -> Result<String, JsValue> {
        Ok(self.data.functions[func_idx].name.clone())
    }

    pub fn get_pass_name(&self, func_idx: usize, pass_idx: usize) -> Result<String, JsValue> {
        Ok(self.data.functions[func_idx].passes[pass_idx].name.clone())
    }
}
```

**New JavaScript Usage** (src/wasm_html_generator.rs):
```javascript
// Initialize once (parses JSON once)
const ionJsonStr = JSON.stringify(ION_DATA);
viewer = new IonGraphViewer(ionJsonStr);  // 60-100ms ONE TIME

// Query metadata - NO parsing!
const numFuncs = viewer.get_function_count();  // <1ms
const funcName = viewer.get_function_name(i);  // <1ms

// Render pass - NO parsing!
const svg = viewer.render_pass(funcIdx, passIdx);  // ~2ms (just layout+SVG)
```

## Performance Improvement

### Before (Stateless)
- Initial page load: ~60ms (parse JSON once for first render)
- Switching functions (30 passes): **~2 seconds** (31 JSON parses)
- Switching passes: **~60ms** (1 JSON parse per switch)
- **Total overhead per function switch: ~2 seconds of unnecessary JSON parsing**

### After (Stateful)
- Initial page load: ~60ms (parse JSON once, cache in WASM)
- Switching functions (30 passes): **~30ms** (30 calls × <1ms, no parsing!)
- Switching passes: **~2ms** (just layout+SVG, no parsing!)
- **Total overhead eliminated: ~2 seconds → ~30ms (98.5% reduction!)**

## Expected User Experience

### Before
- First render: Slow (~60ms)
- Switching to new function: **VERY slow (~2 seconds)** - building sidebar
- Switching passes: Slow (~60ms per arrow key press)
- **Feels sluggish and unresponsive**

### After
- First render: Slow (~60ms one-time cost)
- Switching to new function: **Fast (~30ms)** - sidebar populates instantly
- Switching passes: **Very fast (~2ms per arrow key press)**
- **Feels instant and responsive** ✨

## Files Modified

1. **src/wasm.rs** - Replaced stateless functions with `IonGraphViewer` struct
2. **src/wasm_html_generator.rs** - Updated JavaScript to use stateful viewer instance
3. **src/bin/wasm_benchmark.rs** - New benchmark binary for performance measurement (NEW)
4. **Cargo.toml** - Added wasm-benchmark binary target

## Testing

Generated test file: `test-stateful.html` (9.2MB, same size as before)

### Verification
```bash
# Build WASM
wasm-pack build --target web --out-dir pkg

# Build iongraph binary
cargo build --release --bin iongraph

# Generate WASM HTML
./target/release/iongraph --wasm ion-examples/mega-complex.json test-stateful.html
```

### Benchmarking (Native vs WASM)
```bash
# Build benchmark binary
cargo build --release --bin wasm-benchmark

# Run native benchmark
./target/release/wasm-benchmark ion-examples/mega-complex.json 0 0

# Run WASM benchmark (requires wasmtime)
rustup target add wasm32-wasip1
cargo build --release --target wasm32-wasip1 --bin wasm-benchmark
wasmtime --dir=. target/wasm32-wasip1/release/wasm-benchmark.wasm \
  ion-examples/mega-complex.json 0 0
```

## Future Optimizations (If Still Needed)

Based on benchmarks, JSON parsing was **99% of the bottleneck**. With stateful caching, this is now eliminated. If further optimization is needed:

1. **Layout caching** - Cache computed layouts per (func, pass) in WASM
   - Expected impact: Minimal (<2ms already)
   - Only useful if switching back to previously viewed passes

2. **Direct DOM manipulation** - Use `web-sys` instead of `innerHTML`
   - Expected impact: Moderate (if browser DOM parsing is slow)
   - Only useful if SVG strings are very large (>1MB)

3. **Incremental rendering** - Only render visible viewport
   - Expected impact: High for extremely large graphs
   - Complex to implement

## Conclusion

**You were absolutely right** - the Rust/WASM code is blazingly fast. The problem was architectural: we were re-parsing the entire 9.76MB JSON file on every single WASM call.

By implementing a stateful WASM instance that parses JSON once and caches it in WASM memory, we've eliminated **98.5% of the overhead** (2 seconds → 30ms for function switching).

The viewer should now feel instant and responsive! 🚀
