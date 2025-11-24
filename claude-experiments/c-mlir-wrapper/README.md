# MLIR Dialect Introspection - C API Extension

**Enumerate operations, check if ops exist, discover dialects - from C, Zig, or any language with C FFI.**

This library **extends** the MLIR C API with dialect introspection capabilities that exist in C++ but aren't exposed to C.

> **👋 New here?** Start with **[START_HERE.md](START_HERE.md)** for a quick orientation!

## The Problem

The MLIR C API doesn't let you:
- ❌ List all operations in a dialect
- ❌ Check if an operation exists (beyond string prefix matching)
- ❌ Discover what dialects are loaded at runtime

These capabilities exist in the C++ API but aren't exposed.

## The Solution

A lightweight C wrapper around MLIR's C++ introspection APIs:

- ✅ **Pure C interface** - Works with C, Zig, Rust, Python, etc.
- ✅ **Extends, doesn't replace** - Use alongside standard MLIR C API
- ✅ **Registry-based lookup** - Proper checking, not just string matching
- ✅ **Zero overhead** - Only pay when you introspect
- ✅ **Easy to integrate** - Single library, standard MLIR types

## Features

### ✅ Operation Introspection (Fully Supported)
- `mlirEnumerateDialectOperations()` - Enumerate all ops in a dialect
- `mlirOperationBelongsToDialect()` - Check if op is registered in dialect
- `mlirDialectHasOperation()` - Lookup operation by short name
- `mlirOperationGetDialectNamespace()` - Parse dialect from op name
- `mlirOperationGetShortName()` - Get op name without dialect prefix

### ⚠️ Type & Attribute Checking (Partial Support)
- `mlirTypeBelongsToDialect()` - Check if type instance belongs to dialect
- `mlirAttributeBelongsToDialect()` - Check if attribute instance belongs to dialect
- ❌ `mlirEnumerateDialectTypes()` - Not supported (MLIR C++ API limitation)
- ❌ `mlirEnumerateDialectAttributes()` - Not supported (MLIR C++ API limitation)

### ✅ Dialect Discovery (Fully Supported)
- `mlirEnumerateLoadedDialects()` - List all loaded dialects
- `mlirDumpAllDialects()` - Debug dump of all dialect info

> **Note:** Type and attribute enumeration is not supported because MLIR's C++ API doesn't expose this. See [CAPABILITIES.md](CAPABILITIES.md) for details.

## Quick Start

```bash
# Install
mkdir build && cd build
cmake ..
make
sudo make install

# Use in your C code
#include "mlir-introspection.h"

# Compile
gcc your_code.c -lmlir-introspection -lMLIRCAPIIR
```

**Want more detail?** → See [`QUICK_START.md`](QUICK_START.md) for a 5-minute tutorial

**Integrating into your project?** → See [`INTEGRATION.md`](INTEGRATION.md) for CMake/Make/Zig/etc.

**Complete API docs?** → See [`USAGE_GUIDE.md`](USAGE_GUIDE.md)

## Example

```c
#include "mlir-introspection.h"
#include "mlir-c/IR.h"
#include "mlir-c/Dialect/Arith.h"

bool printOp(MlirStringRef name, void* data) {
    printf("  %.*s\n", (int)name.length, name.data);
    return true;
}

int main() {
    MlirContext ctx = mlirContextCreate();

    // Load arith dialect (standard MLIR C API)
    MlirDialectHandle arith = mlirGetDialectHandle__arith__();
    mlirDialectHandleRegisterDialect(arith, ctx);
    mlirDialectHandleLoadDialect(arith, ctx);

    // Enumerate operations (this library!)
    MlirStringRef arithNs = mlirStringRefCreateFromCString("arith");
    mlirEnumerateDialectOperations(ctx, arithNs, printOp, NULL);

    // Check if operation exists
    MlirStringRef addi = mlirStringRefCreateFromCString("arith.addi");
    if (mlirOperationBelongsToDialect(ctx, addi, arithNs)) {
        printf("✓ arith.addi is available\n");
    }

    mlirContextDestroy(ctx);
}
```

**Output:**
```
  arith.addi
  arith.addf
  arith.subi
  ... (49 operations total)
✓ arith.addi is available
```

## API Structure

- `include/mlir-introspection.h` - Main introspection API header
- `src/mlir-introspection.cpp` - C++ implementation wrapping MLIR C++ API
- `examples/introspection-example.c` - Complete usage example
- `CMakeLists.txt` - CMake build configuration

## Limitations

### Type and Attribute Enumeration
The current implementation has limited support for enumerating types and attributes because MLIR's C++ API doesn't provide a generic registry for these across all dialects. Individual dialects may expose their own type/attribute lists, but there's no standard interface.

The following functions are available but may return 0 results:
- `mlirEnumerateDialectTypes()`
- `mlirEnumerateDialectAttributes()`

However, you can still check if a specific type/attribute belongs to a dialect:
- `mlirTypeBelongsToDialect()`
- `mlirAttributeBelongsToDialect()`

### Dialect-Specific Extensions
Some dialects may provide additional introspection capabilities through their own APIs. This library provides a generic foundation that can be extended.

## Implementation Notes

This library wraps the C++ MLIR APIs:
- `mlir::MLIRContext::getRegisteredOperations()`
- `mlir::RegisteredOperationName::lookup()`
- `mlir::MLIRContext::getLoadedDialects()`
- `mlir::Dialect::getNamespace()`

All C API types are unwrapped, processed using C++ APIs, then wrapped back to C types for return.

## License

This project follows the LLVM/MLIR license (Apache 2.0 with LLVM exceptions).

## Documentation

- 📖 **[START_HERE.md](START_HERE.md)** - New here? Start here!
- 📖 **[QUICK_START.md](QUICK_START.md)** - 5 minute tutorial
- 📖 **[CAPABILITIES.md](CAPABILITIES.md)** - What works and what doesn't
- 📖 **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Complete API reference with C and Zig examples
- 📖 **[INTEGRATION.md](INTEGRATION.md)** - How to integrate into your build system
- 📖 **[BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)** - Detailed build instructions
- 📖 **[TESTED.md](TESTED.md)** - Test results and verification

## Use Cases

- **Dynamic code generation** - Check what operations are available before generating IR
- **Language bindings** - Expose MLIR capabilities to higher-level languages
- **Error messages** - Suggest available operations when user makes a typo
- **Tooling** - Build MLIR explorers, debuggers, LSPs
- **Custom dialects** - Introspect your own dialects at runtime

## Supported Languages

Any language with C FFI:
- ✅ **C** (native)
- ✅ **Zig** (excellent C interop, see USAGE_GUIDE.md)
- ✅ **Rust** (via bindgen)
- ✅ **Python** (via ctypes/cffi)
- ✅ **Go** (via cgo)
- ✅ **Swift** (native C interop)
- ✅ **Julia** (via ccall)

## FAQ

**Q: Does this replace the MLIR C API?**
A: No! This extends it. Use the standard MLIR C API for everything else.

**Q: Can I enumerate types and attributes?**
A: No, only check if existing instances belong to a dialect. MLIR's C++ API doesn't expose type/attribute enumeration. See [CAPABILITIES.md](CAPABILITIES.md).

**Q: Is this header-only?**
A: No, it's a shared library. C doesn't have templates, so header-only isn't as beneficial.

**Q: What's the performance impact?**
A: Negligible. Uses MLIR's internal hash tables (O(1) lookup).

**Q: Thread-safe?**
A: Yes, follows MLIR C API threading rules (safe per context).

## References

- [MLIR C API Documentation](https://mlir.llvm.org/docs/CAPI/)
- [MLIR Dialect Documentation](https://mlir.llvm.org/docs/Dialects/)
- Original motivation: See `conversation.txt`
