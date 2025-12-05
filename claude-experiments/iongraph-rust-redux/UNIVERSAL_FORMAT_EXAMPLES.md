# Universal Format Examples

This directory contains examples in the Universal CodeGraph format (`codegraph-v1`), a compiler-agnostic intermediate representation format.

## Examples

### Simple Example
- **File**: `examples/simple-universal.json`
- **Blocks**: 4
- **Description**: Simple loop structure with entry, loop header, backedge, and exit blocks
- **Size**: 76 lines
- **SVG Output**: 481x448 pixels

### Large Example
- **File**: `examples/large-universal.json`
- **Blocks**: 221
- **Description**: Very large complex graph with:
  - 3 levels of nested loops
  - 5 loop headers
  - 5 backedge blocks
  - Long edge chains requiring many dummy nodes
  - Complex control flow with branches and merges
  - Sequential code sections
- **Size**: 6,829 lines (121 KB)
- **SVG Output**: 2,174 x 29,898 pixels

## Rendering Universal Format

Use the `render_universal` binary to render universal format files directly:

```bash
# Build the binary
cargo build --release --bin render_universal

# Render simple example
./target/release/render_universal examples/simple-universal.json output.svg

# Render large example
./target/release/render_universal examples/large-universal.json output.svg
```

## Testing

Run the universal format test suite:

```bash
./test_universal_direct.sh
```

This tests that both small and large universal format files render correctly.

## Generating More Examples

Use the `generate_large_universal.py` script to create additional large examples:

```bash
python3 generate_large_universal.py
```

This will regenerate `examples/large-universal.json` with a fresh synthetic graph.

## Universal Format Structure

The universal format is defined in `src/compilers/universal/schema.rs`:

```json
{
  "format": "codegraph-v1",
  "compiler": "ion|llvm-mir|gcc-rtl|synthetic",
  "metadata": {
    "name": "functionName",
    "optimization_level": 2
  },
  "blocks": [
    {
      "id": "0",
      "attributes": ["entry", "loopheader", "backedge"],
      "loopDepth": 0,
      "predecessors": ["1", "2"],
      "successors": ["3"],
      "instructions": [
        {
          "opcode": "Add",
          "attributes": ["Commutative"],
          "type": "int32"
        }
      ]
    }
  ]
}
```

## Conversion from IonJSON

The `generate_svg` binary automatically converts IonJSON format to universal format internally:

```bash
./target/release/generate_svg ion-examples/mega-complex.json 0 0 output.svg
```

This conversion happens via `pass_to_universal()` in `src/compilers/universal/convert.rs`.

## Large Example Statistics

The generated large example (`large-universal.json`) includes:

- **Total blocks**: 221
- **Max loop depth**: 3 (triple-nested loops)
- **Loop headers**: 5
- **Backedge blocks**: 5
- **Control flow patterns**:
  - Outer loop with multiple inner loops
  - Nested loops at depths 1, 2, and 3
  - Long edge chains (50+ blocks)
  - Branch-merge patterns
  - Sequential code sections

This stresses the layout engine with:
- Complex loop nesting
- Long edges requiring many dummy nodes
- Multiple backedges at different loop levels
- Large vertical span (almost 30,000 pixels)
