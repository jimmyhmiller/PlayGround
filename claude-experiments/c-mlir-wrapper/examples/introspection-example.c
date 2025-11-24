#include "mlir-introspection.h"
#include "mlir-c/IR.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/Dialect/Arith.h"
#include "mlir-c/Dialect/Func.h"
#include <stdio.h>
#include <string.h>

// Callback to print each operation found
bool printOperation(MlirStringRef opName, void* userData) {
    printf("  - %.*s\n", (int)opName.length, opName.data);
    return true; // Continue enumeration
}

// Callback to count operations
bool countOperation(MlirStringRef opName, void* userData) {
    size_t* count = (size_t*)userData;
    (*count)++;
    return true;
}

// Callback to print each dialect
bool printDialect(MlirStringRef dialectNs, void* userData) {
    printf("\nDialect: %.*s\n", (int)dialectNs.length, dialectNs.data);

    MlirContext ctx = *(MlirContext*)userData;

    // Enumerate operations in this dialect
    printf("Operations:\n");
    mlirEnumerateDialectOperations(ctx, dialectNs, printOperation, NULL);

    return true;
}

int main(int argc, char** argv) {
    printf("=== MLIR Dialect Introspection Example ===\n\n");

    // Create MLIR context
    MlirContext ctx = mlirContextCreate();
    mlirContextSetAllowUnregisteredDialects(ctx, false);

    // Load some common dialects
    printf("Loading dialects...\n");
    MlirDialectHandle arithHandle = mlirGetDialectHandle__arith__();
    mlirDialectHandleRegisterDialect(arithHandle, ctx);
    mlirDialectHandleLoadDialect(arithHandle, ctx);

    MlirDialectHandle funcHandle = mlirGetDialectHandle__func__();
    mlirDialectHandleRegisterDialect(funcHandle, ctx);
    mlirDialectHandleLoadDialect(funcHandle, ctx);

    printf("Dialects loaded.\n\n");

    // Enumerate all loaded dialects
    printf("=== Enumerating all loaded dialects ===\n");
    size_t dialectCount = mlirEnumerateLoadedDialects(ctx, printDialect, &ctx);
    printf("\nTotal dialects loaded: %zu\n\n", dialectCount);

    // Test specific dialect queries
    printf("=== Testing Arith Dialect ===\n");
    MlirStringRef arithNs = mlirStringRefCreateFromCString("arith");

    // Enumerate arith operations
    printf("Arith operations:\n");
    size_t arithOpCount = 0;
    mlirEnumerateDialectOperations(ctx, arithNs, countOperation, &arithOpCount);
    printf("Found %zu operations in arith dialect\n\n", arithOpCount);

    // Test operation name parsing
    printf("=== Testing operation name parsing ===\n");
    MlirStringRef testOpName = mlirStringRefCreateFromCString("arith.addi");

    MlirStringRef dialect = mlirOperationGetDialectNamespace(testOpName);
    MlirStringRef shortName = mlirOperationGetShortName(testOpName);

    printf("Full name: %.*s\n", (int)testOpName.length, testOpName.data);
    printf("Dialect: %.*s\n", (int)dialect.length, dialect.data);
    printf("Short name: %.*s\n\n", (int)shortName.length, shortName.data);

    // Test operation lookup
    printf("=== Testing operation lookup ===\n");
    MlirStringRef addiOp = mlirStringRefCreateFromCString("arith.addi");
    bool hasAddi = mlirOperationBelongsToDialect(ctx, addiOp, arithNs);
    printf("Does 'arith.addi' belong to arith dialect? %s\n", hasAddi ? "YES" : "NO");

    MlirStringRef fakeOp = mlirStringRefCreateFromCString("arith.notreal");
    bool hasFake = mlirOperationBelongsToDialect(ctx, fakeOp, arithNs);
    printf("Does 'arith.notreal' belong to arith dialect? %s\n\n", hasFake ? "YES" : "NO");

    // Test short name lookup
    printf("=== Testing short name lookup ===\n");
    MlirStringRef addiShort = mlirStringRefCreateFromCString("addi");
    bool dialectHasAddi = mlirDialectHasOperation(ctx, arithNs, addiShort);
    printf("Does arith dialect have 'addi' operation? %s\n", dialectHasAddi ? "YES" : "NO");

    MlirStringRef fakeShort = mlirStringRefCreateFromCString("notreal");
    bool dialectHasFake = mlirDialectHasOperation(ctx, arithNs, fakeShort);
    printf("Does arith dialect have 'notreal' operation? %s\n\n", dialectHasFake ? "YES" : "NO");

    // Dump all information
    printf("=== Complete dump of all dialects ===\n");
    mlirDumpAllDialects(ctx);

    // Cleanup
    mlirContextDestroy(ctx);

    printf("\n=== Example completed successfully ===\n");
    return 0;
}
