# CodeGraph Universal JSON Format

## Overview

The CodeGraph Universal JSON Format is a compiler-agnostic representation of control flow graphs. It allows any compiler to generate visualizations using CodeGraph without requiring compiler-specific code.

## Format Specification

### Version

Current version: `codegraph-v1`

### Schema

```json
{
  "format": "codegraph-v1",
  "compiler": "string",
  "metadata": { },
  "blocks": [ ]
}
```

### Required Fields

#### Top Level

- **`format`** (string, required): Must be `"codegraph-v1"` to identify this as universal format
- **`compiler`** (string, required): Compiler identifier (e.g., `"ion"`, `"llvm-mir"`, `"gcc-rtl"`)
- **`blocks`** (array, required): Array of basic blocks

#### Block Object

- **`id`** (string, required): Unique block identifier
- **`predecessors`** (array of strings, required): Array of predecessor block IDs
- **`successors`** (array of strings, required): Array of successor block IDs
- **`instructions`** (array, required): Array of instruction objects

#### Instruction Object

- **`opcode`** (string, required): Instruction name/opcode

### Optional Fields

#### Top Level

- **`metadata`** (object, optional): Arbitrary metadata about the compilation unit
  - Common fields: `name`, `optimization_level`, `source_file`

#### Block Object

- **`attributes`** (array of strings, optional): Block attributes
  - Common semantic attributes:
    - `"loopheader"` - Block is a loop header
    - `"backedge"` - Block is a backedge (jumps back to loop header)
    - `"entry"` - Function entry block
    - `"splitedge"` - Block created by edge splitting
- **`loopDepth`** (number, optional): Loop nesting depth (0 = not in loop)
- **`metadata`** (object, optional): Block-specific metadata

#### Instruction Object

- **`attributes`** (array of strings, optional): Instruction attributes
- **`type`** (string, optional): Type annotation (e.g., `"int32"`, `"i64*"`)
- **`profiling`** (object, optional): Profiling/sample count data
  - `sample_count` (number): Total execution samples
  - `hotness` (number, 0.0-1.0): Normalized hotness score
- **`metadata`** (object, optional): Instruction-specific metadata

## Examples

### Minimal Example

```json
{
  "format": "codegraph-v1",
  "compiler": "example",
  "blocks": [
    {
      "id": "entry",
      "predecessors": [],
      "successors": ["exit"],
      "instructions": [
        {
          "opcode": "return"
        }
      ]
    },
    {
      "id": "exit",
      "predecessors": ["entry"],
      "successors": [],
      "instructions": []
    }
  ]
}
```

### Loop Example

```json
{
  "format": "codegraph-v1",
  "compiler": "ion",
  "metadata": {
    "name": "simpleLoop",
    "optimization_level": 2
  },
  "blocks": [
    {
      "id": "0",
      "attributes": ["entry"],
      "loopDepth": 0,
      "predecessors": [],
      "successors": ["1"],
      "instructions": [
        {
          "opcode": "MParameter",
          "type": "int32"
        }
      ]
    },
    {
      "id": "1",
      "attributes": ["loopheader"],
      "loopDepth": 1,
      "predecessors": ["0", "2"],
      "successors": ["2", "3"],
      "instructions": [
        {
          "opcode": "MPhi",
          "type": "int32"
        },
        {
          "opcode": "MCompare",
          "attributes": ["Movable"],
          "type": "boolean"
        }
      ]
    },
    {
      "id": "2",
      "attributes": ["backedge"],
      "loopDepth": 1,
      "predecessors": ["1"],
      "successors": ["1"],
      "instructions": [
        {
          "opcode": "MAdd",
          "attributes": ["Movable"],
          "type": "int32"
        }
      ]
    },
    {
      "id": "3",
      "attributes": [],
      "loopDepth": 0,
      "predecessors": ["1"],
      "successors": [],
      "instructions": [
        {
          "opcode": "MReturn",
          "type": "int32"
        }
      ]
    }
  ]
}
```

### With Profiling Data

```json
{
  "format": "codegraph-v1",
  "compiler": "profiled-compiler",
  "blocks": [
    {
      "id": "hot_loop",
      "attributes": ["loopheader"],
      "loopDepth": 1,
      "predecessors": ["entry", "hot_loop"],
      "successors": ["hot_loop", "exit"],
      "instructions": [
        {
          "opcode": "add",
          "type": "i64",
          "profiling": {
            "sample_count": 125000,
            "hotness": 0.95
          }
        }
      ]
    }
  ]
}
```

## Semantic Attributes

The universal format supports multiple naming conventions for semantic attributes:

| Semantic Meaning | Supported Names |
|-----------------|-----------------|
| Loop Header | `loopheader`, `loop.header`, `loop_header` |
| Backedge | `backedge`, `loop.latch`, `loop_backedge` |
| Split Edge | `splitedge`, `split_edge` |
| Entry Block | `entry` |
| Unreachable | `unreachable` |

This allows different compilers to use their preferred naming conventions while maintaining semantic compatibility.

## Converting from Ion Format

Use the `convert_to_universal` tool to convert Ion JSON to universal format:

```bash
# Convert entire Ion JSON file
cargo run --release --bin convert_to_universal input.json output.json

# Convert single function/pass
cargo run --release --bin convert_to_universal input.json --pass 0 5 output.json
```

## Advantages

1. **Compiler Agnostic**: No compiler-specific code needed
2. **Simple Structure**: Flat block array instead of nested hierarchies
3. **Extensible**: Metadata fields allow compiler-specific extensions
4. **Human Readable**: Clean JSON format easy to generate and debug
5. **Multiple Naming Conventions**: Supports different attribute naming styles

## Migration from Ion Format

The Ion-specific format has a nested structure:

```
IonJSON → functions[] → passes[] → mir/lir → blocks[]
```

The universal format flattens this to:

```
UniversalIR → blocks[]
```

Compiler-specific fields (like Ion's `ptr`, `uses`, `inputs`) are preserved in the `metadata` object.
