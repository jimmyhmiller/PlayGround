# Quick Start - 5 Minute Tutorial

## Install (30 seconds)

```bash
cd mlir-introspection-c-api
mkdir build && cd build
cmake ..
make
sudo make install
```

## Your First Program (2 minutes)

Create `test.c`:

```c
#include "mlir-introspection.h"
#include "mlir-c/IR.h"
#include "mlir-c/Dialect/Arith.h"
#include <stdio.h>

bool printOp(MlirStringRef name, void* data) {
    printf("  %.*s\n", (int)name.length, name.data);
    return true;
}

int main() {
    // Create context
    MlirContext ctx = mlirContextCreate();

    // Load arith dialect
    MlirDialectHandle arith = mlirGetDialectHandle__arith__();
    mlirDialectHandleRegisterDialect(arith, ctx);
    mlirDialectHandleLoadDialect(arith, ctx);

    // List all operations
    printf("Arith operations:\n");
    MlirStringRef arithNs = mlirStringRefCreateFromCString("arith");
    mlirEnumerateDialectOperations(ctx, arithNs, printOp, NULL);

    // Check if operation exists
    MlirStringRef addi = mlirStringRefCreateFromCString("arith.addi");
    if (mlirOperationBelongsToDialect(ctx, addi, arithNs)) {
        printf("\n✓ arith.addi is available\n");
    }

    mlirContextDestroy(ctx);
    return 0;
}
```

## Compile and Run (30 seconds)

```bash
gcc test.c -o test -lmlir-introspection -lMLIRCAPIIR
./test
```

You should see:
```
Arith operations:
  arith.addi
  arith.addf
  arith.subi
  ... (and 46 more)

✓ arith.addi is available
```

## That's it! 🎉

You now have dialect introspection working.

## Next Steps

- Read `USAGE_GUIDE.md` for complete API documentation
- Check `examples/introspection-example.c` for more patterns
- See `INTEGRATION.md` for integrating into your build system

## Common Use Cases

### Check if an operation exists before creating it

```c
MlirStringRef op = mlirStringRefCreateFromCString("arith.addi");
MlirStringRef ns = mlirStringRefCreateFromCString("arith");

if (mlirOperationBelongsToDialect(ctx, op, ns)) {
    // Safe to create the operation
} else {
    fprintf(stderr, "Operation not available!\n");
}
```

### List all loaded dialects

```c
bool printDialect(MlirStringRef ns, void* data) {
    printf("Loaded: %.*s\n", (int)ns.length, ns.data);
    return true;
}

mlirEnumerateLoadedDialects(ctx, printDialect, NULL);
```

### Parse operation names

```c
MlirStringRef full = mlirStringRefCreateFromCString("arith.addi");
MlirStringRef dialect = mlirOperationGetDialectNamespace(full);
MlirStringRef name = mlirOperationGetShortName(full);

// dialect = "arith"
// name = "addi"
```

## From Zig? Even Easier!

```zig
const c = @cImport({
    @cInclude("mlir-introspection.h");
    @cInclude("mlir-c/IR.h");
});

pub fn main() !void {
    const ctx = c.mlirContextCreate();
    defer c.mlirContextDestroy(ctx);

    // ... same as C ...
}
```

## Questions?

- **Not working?** → Check `TROUBLESHOOTING.md`
- **Need more examples?** → See `examples/`
- **Integrating into your project?** → Read `INTEGRATION.md`
- **Zig-specific help?** → See `USAGE_GUIDE.md` Zig section
