# LLVM MIR Support in CodeGraph

## Overview

CodeGraph now supports visualizing LLVM Machine IR (MIR), demonstrating that the universal visualization engine works with multiple compilers.

## LLVM MIR Format

### JSON Schema

```json
{
  "name": "module_name",
  "target": "x86_64-unknown-linux-gnu",
  "functions": [
    {
      "name": "function_name",
      "attributes": ["nounwind", "readonly"],
      "blocks": [
        {
          "label": "entry",
          "attributes": ["loop.header"],
          "loopDepth": 1,
          "predecessors": ["pred1", "pred2"],
          "successors": ["succ1"],
          "instructions": [
            {
              "result": "%3",
              "opcode": "add",
              "type": "i32",
              "operands": ["%1", "%2"],
              "attributes": ["nsw", "nuw"]
            }
          ]
        }
      ]
    }
  ]
}
```

### Key Differences from Ion

| Feature | Ion | LLVM |
|---------|-----|------|
| **Block IDs** | Numeric (0, 1, 2) | Labels (entry, loop.header) |
| **Loop Headers** | `"loopheader"` | `"loop.header"` |
| **Backedges** | `"backedge"` | `"loop.latch"` |
| **Instruction Format** | `opcode type` | `%result = opcode type operands` |
| **Attributes** | Movable, Guard | nounwind, readonly, nsw, nuw |

## Semantic Attribute Mapping

CodeGraph automatically maps LLVM attributes to universal semantics:

```rust
"loop.header" → SemanticAttribute::LoopHeader
"loop.latch"  → SemanticAttribute::Backedge
```

This means the same layout algorithms work for both Ion and LLVM.

## Example: Simple Loop

### LLVM IR Source

```llvm
define void @simple_loop(i32 %n) nounwind {
entry:
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.body ]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %loop.body, label %exit

loop.body:
  %i.next = add nsw i32 %i, 1
  br label %loop.header

exit:
  ret void
}
```

### JSON Representation

See `examples/llvm/simple-loop.json` for the complete JSON representation.

### Universal Format

Convert to universal format:

```bash
cargo run --bin convert_llvm_to_universal examples/llvm/simple-loop.json output.json
```

Result (`output.json`):
```json
{
  "format": "codegraph-v1",
  "compiler": "llvm-mir",
  "blocks": [
    {
      "id": "entry",
      "successors": ["loop.header"],
      "instructions": [...]
    },
    {
      "id": "loop.header",
      "attributes": ["loop.header"],
      "loopDepth": 1,
      "instructions": [...]
    }
  ]
}
```

## Instruction Rendering

LLVM instructions are rendered differently from Ion:

### Ion Format
```
5  |  MLoadElement  |  int32
```

### LLVM Format
```
%3  =  add  i32  %1, %2
```

The `IRInstruction::render_row()` trait method allows each compiler to define its own rendering style.

## Theme Support

Use the LLVM theme for LLVM-specific attribute colors:

```bash
# Test LLVM theme
cargo run --bin test_themes config/llvm.toml
```

LLVM theme colors (`config/llvm.toml`):
- **nounwind**: `#4ec9b0` (cyan)
- **readonly**: `#569cd6` (blue)
- **nsw**: `#ce9178` (orange)
- **nuw**: `#dcdcaa` (yellow)

## Creating LLVM Examples

### Method 1: Use Built-in Example

```bash
# Generate example LLVM JSON
cargo run --bin generate_llvm_example examples/llvm/my-function.json
```

### Method 2: Write JSON Manually

```json
{
  "name": "my_module",
  "target": "x86_64-pc-linux-gnu",
  "functions": [
    {
      "name": "my_function",
      "attributes": [],
      "blocks": [
        {
          "label": "bb0",
          "attributes": [],
          "loopDepth": 0,
          "predecessors": [],
          "successors": ["bb1"],
          "instructions": [
            {
              "result": "%0",
              "opcode": "alloca",
              "type": "i32",
              "operands": [],
              "attributes": []
            }
          ]
        }
      ]
    }
  ]
}
```

### Method 3: Extract from LLVM

If you have access to LLVM, you can use this process (not yet automated):

1. **Compile to IR**:
   ```bash
   clang -S -emit-llvm -O2 input.c -o output.ll
   ```

2. **Extract MIR** (manually create JSON from IR)

3. **Add control flow annotations**:
   - Mark loop headers with `"loop.header"` attribute
   - Mark backedges with `"loop.latch"` attribute
   - Calculate `loopDepth` for each block

## Visualization

Once you have LLVM JSON, visualize it using CodeGraph:

```bash
# Convert to universal format
cargo run --bin convert_llvm_to_universal examples/llvm/simple-loop.json universal.json

# Future: Direct visualization support
# cargo run --bin visualize_llvm examples/llvm/simple-loop.json -o output.svg
```

## Supported LLVM Features

### ✅ Fully Supported

- Basic blocks with labels
- Control flow (predecessors/successors)
- Loop detection (via `loop.header` attribute)
- Backedges (via `loop.latch` attribute)
- Loop depth tracking
- SSA instructions with registers
- Instruction attributes (nsw, nuw, etc.)
- Type annotations

### ⚠️ Partial Support

- Function attributes (loaded but not visualized)
- Metadata (stored but not displayed)

### ❌ Not Yet Supported

- PHI node special rendering
- Memory dependency visualization
- Register allocation coloring
- Call graph integration

## LLVM-Specific Attributes

### Instruction Attributes

- **nsw**: No signed wrap
- **nuw**: No unsigned wrap
- **inbounds**: GEP inbounds flag
- **exact**: Exact division
- **nnan**: No NaN values
- **ninf**: No infinity values

### Function Attributes

- **nounwind**: Function doesn't throw
- **readonly**: Function doesn't write memory
- **noalias**: Pointer doesn't alias
- **alwaysinline**: Always inline
- **noinline**: Never inline

## Comparison with Other Compilers

| Feature | Ion | LLVM | Future: GCC RTL |
|---------|-----|------|-----------------|
| Format | JSON | JSON | JSON |
| Block IDs | Numbers | Labels | Numbers or Labels |
| Loop Markers | loopheader | loop.header | RTL_LOOP_HEADER |
| Backedges | backedge | loop.latch | RTL_BACKEDGE |
| Instructions | Opcode + Type | SSA Form | RTL Patterns |

## Implementation Details

### Files

- `src/compilers/llvm/schema.rs` - LLVM data structures
- `src/compilers/llvm/ir_impl.rs` - CompilerIR trait implementation
- `src/compilers/universal/convert.rs` - LLVM → Universal converter
- `config/llvm.toml` - LLVM theme colors

### Key Code

```rust
// LLVM implements CompilerIR trait
impl CompilerIR for LLVMIR {
    fn format_id() -> &'static str { "llvm-mir" }
    type Instruction = LLVMInstruction;
    type Block = LLVMBlockWithIndices;
    // ...
}

// LLVM attribute semantics
impl AttributeSemantics for LLVMIR {
    fn parse_attribute(attr: &str) -> SemanticAttribute {
        match attr {
            "loop.header" => SemanticAttribute::LoopHeader,
            "loop.latch" => SemanticAttribute::Backedge,
            _ => SemanticAttribute::Custom,
        }
    }
}
```

## Future Work

- **LLVM Integration Tool**: Script to extract MIR from LLVM
- **PHI Node Visualization**: Special rendering for PHI nodes
- **Use-Def Chains**: Visualize SSA dependencies
- **Register Coloring**: Color code virtual registers
- **Pass Progression**: Show optimization pass effects
- **Machine Code**: Support MachineInstr visualization

## Example Output

Running the example:

```bash
$ cargo run --bin generate_llvm_example
✓ LLVM MIR example written to: examples/llvm/simple-loop.json
  Function: simple_loop
  Blocks: 4
  Target: x86_64-unknown-linux-gnu

$ cargo run --bin convert_llvm_to_universal examples/llvm/simple-loop.json output.json
✓ Universal JSON written to: output.json
  Blocks: 4
```

## Conclusion

LLVM MIR support demonstrates that CodeGraph's universal architecture works:

1. **Same layout algorithms** - Loop detection, layering, edge routing
2. **Same theme system** - Just different attribute colors
3. **Different rendering** - Custom instruction format
4. **Clean separation** - LLVM code isolated in `src/compilers/llvm/`

Adding a new compiler requires:
- ~300 lines of schema definitions
- ~200 lines of trait implementations
- ~30 lines of theme configuration

**Total: < 600 lines of code to add full compiler support!**
