# Mega-Complex Universal Format Conversions

This document lists the mega-complex.json functions that have been converted to universal format for testing and comparison.

## Converted Functions

### Function 11: "-e:21" (BuildSSA)
- **Source**: `examples/mega-complex.json` function 11, pass 0
- **Universal**: `examples/mega-complex-func11-pass0-universal.json`
- **SVG**: `examples/mega-complex-func11-pass0-universal.svg`
- **Blocks**: 20
- **Size**: 51 KB (JSON), 69 KB (SVG)
- **Dimensions**: 1109 x 4466 pixels

This function has nested loops with multiple inner loops at different depths.

### Function 6: "self-hosted:205" (BuildSSA)
- **Source**: `examples/mega-complex.json` function 6, pass 0
- **Universal**: `examples/mega-complex-func6-pass0-universal.json`
- **SVG**: `examples/mega-complex-func6-pass0-universal.svg`
- **Blocks**: 22 (largest function)
- **Size**: 44 KB (JSON), 64 KB (SVG)
- **Dimensions**: 1713 x 4016 pixels

This is the largest function in mega-complex.json with 22 basic blocks.

### Function 8: "self-hosted:183" (Split Critical Edges)
- **Source**: `examples/mega-complex.json` function 8, pass 5
- **Universal**: `examples/mega-complex-func8-pass5-universal.json`
- **SVG**: `examples/mega-complex-func8-pass5-universal.svg`
- **Blocks**: 21
- **Size**: 39 KB (JSON), 55 KB (SVG)
- **Dimensions**: 2003 x 3362 pixels

This function showcases the effect of the "Split Critical Edges" optimization pass.

### Function 5: "self-hosted:163" (Split Critical Edges)
- **Source**: `examples/mega-complex.json` function 5, pass 5
- **Universal**: `examples/mega-complex-func5-pass5-universal.json`
- **Blocks**: 17
- **Size**: 35 KB (JSON)

## Converting More Functions

To convert any function/pass from mega-complex.json to universal format:

```bash
# Syntax: convert_to_universal <input> --pass <func-idx> <pass-idx> [output]
./target/release/convert_to_universal examples/mega-complex.json --pass 11 0 output.json
```

### Finding Interesting Functions

mega-complex.json contains 15 functions, each with 35 compilation passes. To find the most complex functions:

```python
import json
with open('examples/mega-complex.json') as f:
    data = json.load(f)

for i, func in enumerate(data['functions']):
    max_blocks = max(len(p['mir']['blocks']) for p in func['passes'] if 'mir' in p and p['mir'])
    print(f"Function {i}: {func['name']} - max {max_blocks} blocks")
```

### Rendering Universal Format

```bash
# Render any universal format file
./target/release/render_universal examples/mega-complex-func11-pass0-universal.json output.svg
```

## Comparison with IonJSON Format

The universal format is a simplified, compiler-agnostic representation:

**Advantages:**
- Simpler schema (no compiler-specific details)
- Can represent IR from any compiler (Ion, LLVM, GCC, etc.)
- Easier to generate synthetic test cases
- Better for cross-compiler comparisons

**IonJSON preserves more details:**
- Instruction pointers (ptr)
- Input/use relationships
- Resume points
- Full type information

Both formats render identically in the visualization - the universal format is created by stripping compiler-specific metadata while preserving the graph structure.

## Statistics

| File | Blocks | JSON Size | SVG Size | Dimensions |
|------|--------|-----------|----------|------------|
| simple-universal.json | 4 | 1.5 KB | 4.3 KB | 481 x 448 |
| mega-complex-func5-pass5 | 17 | 35 KB | - | - |
| mega-complex-func11-pass0 | 20 | 51 KB | 69 KB | 1109 x 4466 |
| mega-complex-func8-pass5 | 21 | 39 KB | 55 KB | 2003 x 3362 |
| mega-complex-func6-pass0 | 22 | 44 KB | 64 KB | 1713 x 4016 |
| large-universal.json | 221 | 121 KB | 331 KB | 2174 x 29898 |

## Testing

All converted universal format files can be tested with:

```bash
./test_universal_direct.sh
```

Or individually:

```bash
./target/release/render_universal examples/mega-complex-func11-pass0-universal.json test.svg
```
