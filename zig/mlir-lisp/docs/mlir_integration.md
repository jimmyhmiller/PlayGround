# MLIR Integration

This document describes how MLIR is integrated into the mlir-lisp compiler.

## Overview

The project uses MLIR (Multi-Level Intermediate Representation) from the LLVM project as its compilation backend. MLIR provides a flexible infrastructure for building domain-specific compilers.

## Directory Structure

```
src/mlir/
├── README.md    # Setup instructions
└── c.zig        # Zig wrappers around MLIR C API
```

## Build System Integration

The MLIR C library is integrated into the build system through `build.zig`:

### MLIR Path Configuration

The build system allows specifying the MLIR installation path:

```bash
zig build --mlir-path=/path/to/mlir
```

Default path: `/usr/local`

### Link Configuration

The build system automatically:
- Adds MLIR include paths (`{mlir-path}/include`)
- Adds MLIR library paths (`{mlir-path}/lib`)
- Links `MLIR-C` library
- Links `libc` and `libc++` (required by MLIR)

This is applied to:
- The main executable
- All test executables

## API Wrappers

The `src/mlir/c.zig` file provides Zig-friendly wrappers around the MLIR C API:

### Context Management
```zig
var ctx = try mlir.Context.create();
defer ctx.destroy();
```

### Module Management
```zig
const loc = mlir.Location.unknown(&ctx);
var mod = try mlir.Module.create(loc);
defer mod.destroy();
```

### Type System
```zig
const i32_type = mlir.Type.@"i32"(&ctx);
const f64_type = mlir.Type.@"f64"(&ctx);
const custom_type = try mlir.Type.parse(&ctx, "!my.type");
```

### Locations
```zig
const unknown = mlir.Location.unknown(&ctx);
const file_loc = mlir.Location.fileLineCol(&ctx, "test.mlir", 10, 5);
```

### Attributes
```zig
const int_attr = mlir.Attribute.integer(i32_type, 42);
const custom_attr = try mlir.Attribute.parse(&ctx, "#my.attr");
```

## Usage in Code

The MLIR module is exported from `src/root.zig`:

```zig
const mlir_lisp = @import("mlir_lisp");
const mlir = mlir_lisp.mlir;

// Use MLIR API
var ctx = try mlir.Context.create();
defer ctx.destroy();
```

## Testing

MLIR integration is tested in `test/mlir_test.zig`:

- Context creation and destruction
- Module creation
- Type system operations
- Location creation

Run tests with:
```bash
zig build test
```

## Installation Requirements

### macOS
```bash
brew install llvm
zig build --mlir-path=/opt/homebrew/opt/llvm  # Apple Silicon
zig build --mlir-path=/usr/local/opt/llvm     # Intel
```

### Linux
```bash
# Ubuntu/Debian
sudo apt-get install llvm-dev libmlir-dev

# Fedora
sudo dnf install llvm-devel mlir-devel

# Arch
sudo pacman -S llvm mlir
```

### From Source
See `src/mlir/README.md` for build-from-source instructions.

## Example: Main Program

The main program (`src/main.zig`) demonstrates MLIR usage:

```zig
pub fn main() !void {
    std.debug.print("MLIR-Lisp Compiler\n", .{});

    // Create an MLIR context
    var ctx = try mlir.Context.create();
    defer ctx.destroy();

    // Create a module
    const loc = mlir.Location.unknown(&ctx);
    var mod = try mlir.Module.create(loc);
    defer mod.destroy();

    // Create types
    const i32_type = mlir.Type.i32(&ctx);
    const f64_type = mlir.Type.f64(&ctx);

    std.debug.print("✓ MLIR integration is working!\n", .{});
}
```

## Future Work

Additional MLIR wrappers to be added:
- Operation builders
- Block and region management
- Pass management
- Dialect registration
- IR printing and parsing
- Verification

## Architecture

```
┌─────────────────┐
│   mlir-lisp     │
│   (S-expr)      │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│    Reader       │
│  (AST builder)  │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   MLIR Builder  │ ← Uses src/mlir/c.zig wrappers
│ (IR generation) │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   MLIR Passes   │
│  (optimization) │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   LLVM Codegen  │
└─────────────────┘
```

The MLIR wrappers in `src/mlir/c.zig` provide the foundation for the "MLIR Builder" stage, where our S-expression AST will be converted to MLIR operations.
