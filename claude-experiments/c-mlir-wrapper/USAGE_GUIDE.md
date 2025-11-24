# Usage Guide: MLIR Introspection C API

## Quick Answer: Do I need the MLIR C API?

**YES** - This library is an **extension** to the MLIR C API, not a replacement.

- ✅ Use the standard MLIR C API for everything: creating contexts, building IR, running passes, etc.
- ✅ Use **this library** to introspect dialects: enumerate operations, check if ops exist, etc.

Think of it as adding missing features to the MLIR C API, not replacing it.

## What This Library Adds

The standard MLIR C API doesn't let you:
- ❌ List all operations in a dialect
- ❌ Check if an operation exists (beyond string matching)
- ❌ Discover what dialects are loaded

This library adds those capabilities using the C++ API under the hood.

## Installation

### Option 1: Build and Install (Recommended)

```bash
cd mlir-introspection-c-api
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make
sudo make install
```

This installs:
- `/usr/local/include/mlir-introspection.h`
- `/usr/local/lib/libmlir-introspection.so` (or `.dylib` on macOS)

### Option 2: Use as Subdirectory

Add to your `CMakeLists.txt`:
```cmake
add_subdirectory(path/to/mlir-introspection-c-api)
target_link_libraries(your_target PRIVATE mlir-introspection)
```

### Option 3: Manual Integration

Copy files to your project:
```bash
cp include/mlir-introspection.h your-project/include/
cp build/libmlir-introspection.so your-project/lib/
```

## Usage from C

### Complete Example

```c
#include "mlir-introspection.h"
#include "mlir-c/IR.h"
#include "mlir-c/Dialect/Arith.h"
#include <stdio.h>

int main() {
    // 1. Create MLIR context (standard C API)
    MlirContext ctx = mlirContextCreate();

    // 2. Load a dialect (standard C API)
    MlirDialectHandle arith = mlirGetDialectHandle__arith__();
    mlirDialectHandleRegisterDialect(arith, ctx);
    mlirDialectHandleLoadDialect(arith, ctx);

    // 3. Use introspection (this library!)
    MlirStringRef arithNs = mlirStringRefCreateFromCString("arith");

    // List all operations
    printf("Operations in arith:\n");
    bool printOp(MlirStringRef name, void* data) {
        printf("  %.*s\n", (int)name.length, name.data);
        return true;
    }
    mlirEnumerateDialectOperations(ctx, arithNs, printOp, NULL);

    // Check if operation exists
    MlirStringRef addi = mlirStringRefCreateFromCString("arith.addi");
    if (mlirOperationBelongsToDialect(ctx, addi, arithNs)) {
        printf("arith.addi exists!\n");
    }

    // Cleanup (standard C API)
    mlirContextDestroy(ctx);
    return 0;
}
```

### Compile Your Program

```bash
# If installed system-wide
gcc your_program.c -o your_program \
    -lmlir-introspection \
    -lMLIRCAPIIR \
    -I/usr/local/include \
    -L/usr/local/lib

# If using homebrew MLIR
gcc your_program.c -o your_program \
    -lmlir-introspection \
    -I$(brew --prefix llvm)/include \
    -L$(brew --prefix llvm)/lib \
    -lMLIRCAPIIR
```

### Common Patterns

#### Pattern 1: Check if an operation exists before using it

```c
MlirStringRef opName = mlirStringRefCreateFromCString("arith.addi");
MlirStringRef dialectNs = mlirStringRefCreateFromCString("arith");

if (mlirOperationBelongsToDialect(ctx, opName, dialectNs)) {
    // Safe to use arith.addi
    // ... create operation ...
} else {
    fprintf(stderr, "Error: arith.addi not available\n");
}
```

#### Pattern 2: List all available operations

```c
typedef struct {
    char** ops;
    size_t count;
} OpList;

bool collectOp(MlirStringRef name, void* userData) {
    OpList* list = (OpList*)userData;
    list->ops[list->count++] = strndup(name.data, name.length);
    return true;
}

OpList list = { .ops = malloc(sizeof(char*) * 100), .count = 0 };
mlirEnumerateDialectOperations(ctx, dialectNs, collectOp, &list);

// Now you have an array of operation names
for (size_t i = 0; i < list.count; i++) {
    printf("%s\n", list.ops[i]);
    free(list.ops[i]);
}
free(list.ops);
```

#### Pattern 3: Parse operation names

```c
MlirStringRef fullName = mlirStringRefCreateFromCString("arith.addi");

MlirStringRef dialect = mlirOperationGetDialectNamespace(fullName);
MlirStringRef shortName = mlirOperationGetShortName(fullName);

printf("Dialect: %.*s, Op: %.*s\n",
       (int)dialect.length, dialect.data,
       (int)shortName.length, shortName.data);
```

## Usage from Zig

Zig has excellent C interop! Here's how to use it:

### build.zig

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "my-mlir-app",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Add MLIR include paths
    exe.addIncludePath(.{ .path = "/opt/homebrew/opt/llvm/include" });

    // Add introspection library include
    exe.addIncludePath(.{ .path = "/usr/local/include" });

    // Link libraries
    exe.addLibraryPath(.{ .path = "/opt/homebrew/opt/llvm/lib" });
    exe.addLibraryPath(.{ .path = "/usr/local/lib" });

    exe.linkSystemLibrary("mlir-introspection");
    exe.linkSystemLibrary("MLIRCAPIIR");
    exe.linkLibC();
    exe.linkLibCpp();

    b.installArtifact(exe);
}
```

### main.zig

```zig
const std = @import("std");
const c = @cImport({
    @cInclude("mlir-introspection.h");
    @cInclude("mlir-c/IR.h");
    @cInclude("mlir-c/Dialect/Arith.h");
});

fn printOp(opName: c.MlirStringRef, userData: ?*anyopaque) callconv(.C) bool {
    _ = userData;
    const name = opName.data[0..opName.length];
    std.debug.print("  {s}\n", .{name});
    return true;
}

pub fn main() !void {
    std.debug.print("=== MLIR from Zig ===\n", .{});

    // Create context
    const ctx = c.mlirContextCreate();
    defer c.mlirContextDestroy(ctx);

    // Load arith dialect
    const arith = c.mlirGetDialectHandle__arith__();
    c.mlirDialectHandleRegisterDialect(arith, ctx);
    c.mlirDialectHandleLoadDialect(arith, ctx);

    // Enumerate operations
    const arithNs = c.mlirStringRefCreateFromCString("arith");
    std.debug.print("Arith operations:\n", .{});

    _ = c.mlirEnumerateDialectOperations(ctx, arithNs, printOp, null);

    // Check if operation exists
    const addi = c.mlirStringRefCreateFromCString("arith.addi");
    if (c.mlirOperationBelongsToDialect(ctx, addi, arithNs)) {
        std.debug.print("arith.addi is available!\n", .{});
    }
}
```

### Build and Run

```bash
zig build
./zig-out/bin/my-mlir-app
```

## Common Questions

### Q: Do I need to learn C++ to use this?
**A:** No! This is a pure C API. You never touch C++ code.

### Q: What dialects can I introspect?
**A:** Any dialect loaded in your `MlirContext`. Standard ones include:
- `arith` - Arithmetic operations
- `func` - Functions
- `tensor` - Tensor operations
- `memref` - Memory references
- `scf` - Structured control flow
- `cf` - Control flow
- Your custom dialects!

### Q: How do I load more dialects?
**A:** Use the standard MLIR C API:

```c
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/Tensor.h"

MlirDialectHandle func = mlirGetDialectHandle__func__();
mlirDialectHandleRegisterDialect(func, ctx);
mlirDialectHandleLoadDialect(func, ctx);

MlirDialectHandle tensor = mlirGetDialectHandle__tensor__();
mlirDialectHandleRegisterDialect(tensor, ctx);
mlirDialectHandleLoadDialect(tensor, ctx);
```

### Q: Can I use this with my custom MLIR dialect?
**A:** Yes! Once your dialect is registered and loaded, you can introspect it:

```c
// After loading your custom dialect
MlirStringRef myDialect = mlirStringRefCreateFromCString("mydialect");
mlirEnumerateDialectOperations(ctx, myDialect, callback, NULL);
```

### Q: Is this thread-safe?
**A:** Thread safety follows MLIR C API rules:
- Safe if using different `MlirContext` per thread
- Not safe if sharing `MlirContext` without external locking
- Enable context threading: `mlirContextEnableMultithreading(ctx, true)`

### Q: What's the performance impact?
**A:** Very minimal:
- Enumeration iterates registered operations (fast)
- Lookup uses MLIR's internal hash table (O(1))
- No runtime overhead when not introspecting

## API Reference Quick Lookup

| Function | Purpose | Example |
|----------|---------|---------|
| `mlirEnumerateDialectOperations` | List all ops in dialect | See Pattern 2 |
| `mlirOperationBelongsToDialect` | Check if op exists | See Pattern 1 |
| `mlirDialectHasOperation` | Lookup by short name | `mlirDialectHasOperation(ctx, "arith", "addi")` |
| `mlirOperationGetDialectNamespace` | Parse dialect from name | See Pattern 3 |
| `mlirOperationGetShortName` | Get op without prefix | See Pattern 3 |
| `mlirEnumerateLoadedDialects` | List all loaded dialects | See example |
| `mlirDumpAllDialects` | Debug print everything | `mlirDumpAllDialects(ctx)` |

## Troubleshooting

### "undefined reference to mlirEnumerateDialectOperations"

Link the library:
```bash
gcc ... -lmlir-introspection
```

### "cannot find mlir-introspection.h"

Add include path:
```bash
gcc ... -I/usr/local/include
```

### "cannot open shared object file"

Add library path or install:
```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# Or on macOS:
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
```

### "dialect not loaded" / returns 0 operations

Make sure you load the dialect first:
```c
MlirDialectHandle h = mlirGetDialectHandle__arith__();
mlirDialectHandleRegisterDialect(h, ctx);
mlirDialectHandleLoadDialect(h, ctx);  // Don't forget this!
```

## Next Steps

1. Check out `examples/introspection-example.c` for a complete working example
2. See `include/mlir-introspection.h` for full API documentation
3. Read MLIR C API docs: https://mlir.llvm.org/docs/CAPI/

## Summary

- ✅ Works alongside MLIR C API (not a replacement)
- ✅ Pure C interface - no C++ needed
- ✅ Works from C, Zig, and any language with C FFI
- ✅ Simple to integrate: just link `-lmlir-introspection`
- ✅ Zero runtime overhead when not introspecting
