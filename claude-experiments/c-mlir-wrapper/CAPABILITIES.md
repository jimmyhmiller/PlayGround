# What Works and What Doesn't

## TL;DR

| Feature | Enumerate All | Check Instance Belongs to Dialect |
|---------|---------------|-----------------------------------|
| **Operations** | ✅ Yes | ✅ Yes |
| **Types** | ❌ No | ✅ Yes |
| **Attributes** | ❌ No | ✅ Yes |
| **Dialects** | ✅ Yes (loaded ones) | N/A |

## Detailed Capabilities

### ✅ Operations - Fully Supported

**What works:**
```c
// Enumerate all operations in a dialect
mlirEnumerateDialectOperations(ctx, dialectNs, callback, userData);

// Check if an operation exists by full name
mlirOperationBelongsToDialect(ctx, opName, dialectNs);

// Check if an operation exists by short name
mlirDialectHasOperation(ctx, dialectNs, shortName);

// Parse operation names
MlirStringRef ns = mlirOperationGetDialectNamespace(fullName);
MlirStringRef name = mlirOperationGetShortName(fullName);
```

**Example:**
```c
MlirStringRef arithNs = mlirStringRefCreateFromCString("arith");

// List all 49 operations in arith dialect
mlirEnumerateDialectOperations(ctx, arithNs, printOp, NULL);

// Check if specific operation exists
MlirStringRef addi = mlirStringRefCreateFromCString("arith.addi");
bool exists = mlirOperationBelongsToDialect(ctx, addi, arithNs); // true
```

**Why it works:**
MLIR's C++ API exposes `MLIRContext::getRegisteredOperations()` which gives us access to all registered operations.

---

### ⚠️ Types - Partial Support

**What works:**
```c
// Check if a type instance belongs to a dialect
MlirType someType = ...; // You have a type from somewhere
bool belongs = mlirTypeBelongsToDialect(someType, dialectNs);
```

**What DOESN'T work:**
```c
// ❌ This returns 0 - cannot enumerate types
mlirEnumerateDialectTypes(ctx, dialectNs, callback, userData); // Returns 0
```

**Example of what works:**
```c
// Given you have a type instance
MlirType i32 = mlirIntegerTypeGet(ctx, 32);
MlirStringRef builtinNs = mlirStringRefCreateFromCString("builtin");

// Check which dialect it belongs to
bool isBuiltin = mlirTypeBelongsToDialect(i32, builtinNs); // true
```

**Why enumeration doesn't work:**
- MLIR's C++ API does **not** expose a `Dialect::getRegisteredTypes()` method
- Types are registered but not stored in an enumerable registry
- Each dialect would need to expose its own type list
- This is a limitation of MLIR itself, not this library

**Workaround:**
If you need to know what types a dialect has, you must:
1. Read the dialect's documentation
2. Or parse the dialect's ODS (TableGen) files
3. Or maintain your own hardcoded list

---

### ⚠️ Attributes - Partial Support

**What works:**
```c
// Check if an attribute instance belongs to a dialect
MlirAttribute someAttr = ...; // You have an attribute from somewhere
bool belongs = mlirAttributeBelongsToDialect(someAttr, dialectNs);
```

**What DOESN'T work:**
```c
// ❌ This returns 0 - cannot enumerate attributes
mlirEnumerateDialectAttributes(ctx, dialectNs, callback, userData); // Returns 0
```

**Example of what works:**
```c
// Given you have an attribute instance
MlirAttribute intAttr = mlirIntegerAttrGet(i32Type, 42);
MlirStringRef builtinNs = mlirStringRefCreateFromCString("builtin");

// Check which dialect it belongs to
bool isBuiltin = mlirAttributeBelongsToDialect(intAttr, builtinNs); // true
```

**Why enumeration doesn't work:**
Same reason as types - MLIR doesn't expose `Dialect::getRegisteredAttributes()`.

---

### ✅ Dialects - Fully Supported

**What works:**
```c
// List all loaded dialects
mlirEnumerateLoadedDialects(ctx, callback, userData);

// Debug dump
mlirDumpAllDialects(ctx);
```

**Example:**
```c
bool printDialect(MlirStringRef ns, void* data) {
    printf("Loaded: %.*s\n", (int)ns.length, ns.data);
    return true;
}

mlirEnumerateLoadedDialects(ctx, printDialect, NULL);
// Output:
// Loaded: builtin
// Loaded: arith
// Loaded: func
```

---

## Practical Usage Patterns

### Pattern 1: Validate Operations Before Use ✅

```c
// Check if operation exists before creating it
MlirStringRef op = mlirStringRefCreateFromCString("arith.addi");
MlirStringRef ns = mlirStringRefCreateFromCString("arith");

if (mlirOperationBelongsToDialect(ctx, op, ns)) {
    // Safe to create arith.addi operation
    MlirOperationState state = mlirOperationStateGet(op, loc);
    // ... build operation ...
} else {
    fprintf(stderr, "Error: arith.addi not available\n");
}
```

### Pattern 2: Generate Error Messages with Suggestions ✅

```c
// User typed "arith.add" instead of "arith.addi"
MlirStringRef userOp = mlirStringRefCreateFromCString("arith.add");
if (!mlirOperationBelongsToDialect(ctx, userOp, arithNs)) {
    printf("Error: 'arith.add' not found. Did you mean:\n");
    mlirEnumerateDialectOperations(ctx, arithNs, printSimilar, "add");
}
```

### Pattern 3: Check Type Ownership (when you have a type) ✅

```c
// You have a type from parsing or construction
MlirType someType = mlirTypeParseGet(ctx, typeStr);

// Check which dialect owns it
MlirStringRef arithNs = mlirStringRefCreateFromCString("arith");
MlirStringRef tensorNs = mlirStringRefCreateFromCString("tensor");

if (mlirTypeBelongsToDialect(someType, arithNs)) {
    printf("This is an arith type\n");
} else if (mlirTypeBelongsToDialect(someType, tensorNs)) {
    printf("This is a tensor type\n");
}
```

### Pattern 4: What You CAN'T Do ❌

```c
// ❌ Cannot enumerate all types in a dialect
// This will return 0:
size_t typeCount = mlirEnumerateDialectTypes(ctx, dialectNs, callback, NULL);

// ❌ Cannot discover what types are available
// You need to know type names from documentation:
MlirType tensor = mlirRankedTensorTypeGet(...); // Must know this exists

// ❌ Cannot enumerate all attributes in a dialect
// This will return 0:
size_t attrCount = mlirEnumerateDialectAttributes(ctx, dialectNs, callback, NULL);
```

---

## Why This Limitation Exists

### In MLIR's C++ API

Operations have special treatment because they're the primary IR construct:
```cpp
// ✅ Operations are enumerable
MLIRContext::getRegisteredOperations() → ArrayRef<RegisteredOperationName>

// ❌ Types are not enumerable at the dialect level
// No Dialect::getRegisteredTypes() exists

// ❌ Attributes are not enumerable at the dialect level
// No Dialect::getRegisteredAttributes() exists
```

### Why?

1. **Operations** are central to MLIR - they're what you build IR with
2. **Types and Attributes** are often parameterized and infinite (e.g., `tensor<NxM>` for any N, M)
3. Individual dialects *can* expose their own type/attribute lists, but there's no standard interface
4. ODS (TableGen) definitions have this info at build time, but it's not exposed at runtime

---

## Recommendations

### For Operations ✅
Use this library! It works perfectly.

### For Types ⚠️
- If you just need to check ownership of a type you already have → use `mlirTypeBelongsToDialect()`
- If you need to enumerate all types → read dialect documentation or parse ODS files
- Consider maintaining a static list of known types if needed

### For Attributes ⚠️
- Same as types - use `mlirAttributeBelongsToDialect()` for checking
- Maintain static knowledge of available attributes

---

## Summary

This library exposes **everything that MLIR's C++ API makes available**:

| You Can | You Cannot |
|---------|------------|
| ✅ List all operations | ❌ List all types |
| ✅ Check if operation exists | ❌ List all attributes |
| ✅ Check if type belongs to dialect | ❌ Discover types at runtime |
| ✅ Check if attribute belongs to dialect | ❌ Discover attributes at runtime |
| ✅ List loaded dialects | |

The limitations exist in MLIR itself, not in this wrapper.

**Bottom line:** This library is most useful for **operation introspection**, which is also the most common use case.
