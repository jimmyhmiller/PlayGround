#ifndef MLIR_INTROSPECTION_H
#define MLIR_INTROSPECTION_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dialect Introspection API
//
// This API exposes dialect reflection capabilities from the C++ API that
// are not available in the standard MLIR C API.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operation Introspection
//===----------------------------------------------------------------------===//

/// Callback for enumerating operations in a dialect.
/// @param opName The full operation name (e.g., "arith.addi")
/// @param userData User-provided data passed through from enumeration call
/// @return true to continue enumeration, false to stop
typedef bool (*MlirOperationEnumeratorCallback)(MlirStringRef opName, void* userData);

/// Enumerate all operations registered in a dialect.
/// @param ctx The MLIR context
/// @param dialectNamespace The dialect namespace (e.g., "arith", "func")
/// @param callback Function called for each operation
/// @param userData User data passed to callback
/// @return Number of operations enumerated
size_t mlirEnumerateDialectOperations(
    MlirContext ctx,
    MlirStringRef dialectNamespace,
    MlirOperationEnumeratorCallback callback,
    void* userData
);

/// Check if an operation name belongs to a specific dialect.
/// This does a proper registry lookup, not just prefix matching.
/// @param ctx The MLIR context
/// @param opName The full operation name (e.g., "arith.addi")
/// @param dialectNamespace The dialect namespace to check against
/// @return true if the operation is registered in the dialect
bool mlirOperationBelongsToDialect(
    MlirContext ctx,
    MlirStringRef opName,
    MlirStringRef dialectNamespace
);

/// Lookup operation information in a dialect's registry.
/// @param ctx The MLIR context
/// @param dialectNamespace The dialect namespace
/// @param opName The operation name within the dialect (e.g., "addi" for arith dialect)
/// @return true if the operation exists in this dialect
bool mlirDialectHasOperation(
    MlirContext ctx,
    MlirStringRef dialectNamespace,
    MlirStringRef opName
);

/// Get the dialect namespace for a fully-qualified operation name.
/// @param opName Fully qualified op name (e.g., "arith.addi")
/// @return The dialect namespace portion, or empty string if not parseable
MlirStringRef mlirOperationGetDialectNamespace(MlirStringRef opName);

/// Get the operation name portion (without dialect prefix).
/// @param opName Fully qualified op name (e.g., "arith.addi")
/// @return The operation name portion (e.g., "addi"), or empty string if not parseable
MlirStringRef mlirOperationGetShortName(MlirStringRef opName);

//===----------------------------------------------------------------------===//
// Type Introspection
//===----------------------------------------------------------------------===//

/// Callback for enumerating types in a dialect.
typedef bool (*MlirTypeEnumeratorCallback)(MlirStringRef typeName, void* userData);

/// Enumerate all types registered in a dialect.
/// @param ctx The MLIR context
/// @param dialectNamespace The dialect namespace
/// @param callback Function called for each type
/// @param userData User data passed to callback
/// @return Number of types enumerated
size_t mlirEnumerateDialectTypes(
    MlirContext ctx,
    MlirStringRef dialectNamespace,
    MlirTypeEnumeratorCallback callback,
    void* userData
);

/// Check if a type belongs to a specific dialect.
/// @param type The type to check
/// @param dialectNamespace The dialect namespace to check against
/// @return true if the type belongs to the dialect
bool mlirTypeBelongsToDialect(
    MlirType type,
    MlirStringRef dialectNamespace
);

//===----------------------------------------------------------------------===//
// Attribute Introspection
//===----------------------------------------------------------------------===//

/// Callback for enumerating attributes in a dialect.
typedef bool (*MlirAttributeEnumeratorCallback)(MlirStringRef attrName, void* userData);

/// Enumerate all attributes registered in a dialect.
/// @param ctx The MLIR context
/// @param dialectNamespace The dialect namespace
/// @param callback Function called for each attribute
/// @param userData User data passed to callback
/// @return Number of attributes enumerated
size_t mlirEnumerateDialectAttributes(
    MlirContext ctx,
    MlirStringRef dialectNamespace,
    MlirAttributeEnumeratorCallback callback,
    void* userData
);

/// Check if an attribute belongs to a specific dialect.
/// @param attr The attribute to check
/// @param dialectNamespace The dialect namespace to check against
/// @return true if the attribute belongs to the dialect
bool mlirAttributeBelongsToDialect(
    MlirAttribute attr,
    MlirStringRef dialectNamespace
);

//===----------------------------------------------------------------------===//
// Dialect Enumeration
//===----------------------------------------------------------------------===//

/// Callback for enumerating loaded dialects.
typedef bool (*MlirDialectEnumeratorCallback)(MlirStringRef dialectNamespace, void* userData);

/// Enumerate all loaded dialects in a context.
/// @param ctx The MLIR context
/// @param callback Function called for each loaded dialect
/// @param userData User data passed to callback
/// @return Number of dialects enumerated
size_t mlirEnumerateLoadedDialects(
    MlirContext ctx,
    MlirDialectEnumeratorCallback callback,
    void* userData
);

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Print all operations in a dialect to stdout (for debugging).
/// @param ctx The MLIR context
/// @param dialectNamespace The dialect namespace
void mlirDumpDialectOperations(MlirContext ctx, MlirStringRef dialectNamespace);

/// Print all types in a dialect to stdout (for debugging).
/// @param ctx The MLIR context
/// @param dialectNamespace The dialect namespace
void mlirDumpDialectTypes(MlirContext ctx, MlirStringRef dialectNamespace);

/// Print all attributes in a dialect to stdout (for debugging).
/// @param ctx The MLIR context
/// @param dialectNamespace The dialect namespace
void mlirDumpDialectAttributes(MlirContext ctx, MlirStringRef dialectNamespace);

/// Print complete information about all loaded dialects.
/// @param ctx The MLIR context
void mlirDumpAllDialects(MlirContext ctx);

#ifdef __cplusplus
}
#endif

#endif // MLIR_INTROSPECTION_H
