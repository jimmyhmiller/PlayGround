# 👋 Start Here

Welcome to **MLIR Dialect Introspection C API**!

## What is this?

This library lets you **discover what operations are available** in MLIR dialects, and **check type/attribute ownership** - from C, Zig, or any language with C FFI.

The standard MLIR C API doesn't expose this. We fix that.

**What works:**
- ✅ Enumerate all operations in a dialect
- ✅ Check if operations exist
- ✅ Check if types/attributes belong to a dialect (when you have an instance)

**What doesn't work:**
- ❌ Enumerate all types in a dialect (MLIR C++ API limitation)
- ❌ Enumerate all attributes in a dialect (MLIR C++ API limitation)

See **[CAPABILITIES.md](CAPABILITIES.md)** for full details.

## I just want to use it!

→ **[QUICK_START.md](QUICK_START.md)** - 5 minute tutorial

Install:
```bash
mkdir build && cd build && cmake .. && make && sudo make install
```

Use:
```c
#include "mlir-introspection.h"
mlirEnumerateDialectOperations(ctx, dialectNs, callback, NULL);
```

## I want to integrate it into my project

→ **[INTEGRATION.md](INTEGRATION.md)** - CMake, Make, Zig, etc.

## I want to understand the full API

→ **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Complete reference with examples

## I'm having build issues

→ **[BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)** - Detailed build guide

## Does it actually work?

→ **[TESTED.md](TESTED.md)** - Yes! See test results.

## Key Questions Answered

### Does this replace the MLIR C API?

**No!** This **extends** it. You still use the normal MLIR C API for:
- Creating contexts
- Loading dialects
- Building IR
- Running passes
- Everything else

Use **this library** only for:
- Listing operations in a dialect
- Checking if an operation exists
- Discovering loaded dialects

### What languages work?

Anything with C FFI:
- C (obviously)
- Zig (excellent support, see USAGE_GUIDE.md)
- Rust, Python, Go, Swift, Julia, etc.

### Is this hard to use?

No! Three simple steps:

1. Install: `sudo make install`
2. Include: `#include "mlir-introspection.h"`
3. Link: `gcc ... -lmlir-introspection`

### Is it fast?

Yes. Uses MLIR's internal hash tables. O(1) lookups. Zero overhead when not introspecting.

## File Guide

```
.
├── README.md              ← Project overview
├── START_HERE.md          ← This file
├── QUICK_START.md         ← 5 minute tutorial
├── USAGE_GUIDE.md         ← Complete API docs
├── INTEGRATION.md         ← Build system integration
├── BUILD_INSTRUCTIONS.md  ← Detailed build guide
├── TESTED.md              ← Verification results
├── include/
│   └── mlir-introspection.h     ← Main API header
├── src/
│   └── mlir-introspection.cpp   ← Implementation
└── examples/
    └── introspection-example.c  ← Working example
```

## Still Have Questions?

1. Check the [README.md](README.md) FAQ section
2. Look at [examples/introspection-example.c](examples/introspection-example.c)
3. Read the header [include/mlir-introspection.h](include/mlir-introspection.h) - it's well documented

## Quick Example

```c
#include "mlir-introspection.h"
#include "mlir-c/IR.h"

int main() {
    MlirContext ctx = mlirContextCreate();

    // Load dialect (standard MLIR C API)
    MlirDialectHandle arith = mlirGetDialectHandle__arith__();
    mlirDialectHandleLoadDialect(arith, ctx);

    // Enumerate operations (this library!)
    MlirStringRef ns = mlirStringRefCreateFromCString("arith");
    mlirEnumerateDialectOperations(ctx, ns, printOp, NULL);

    mlirContextDestroy(ctx);
}
```

That's it! 🎉

## Next Steps

Choose your path:

- **Just want to try it?** → [QUICK_START.md](QUICK_START.md)
- **Integrating into a project?** → [INTEGRATION.md](INTEGRATION.md)
- **Need full API docs?** → [USAGE_GUIDE.md](USAGE_GUIDE.md)
- **Using Zig?** → [USAGE_GUIDE.md](USAGE_GUIDE.md#usage-from-zig)
- **Building from source?** → [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)
