#include "mlir-introspection.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/StringRef.h"
#include <iostream>

using namespace mlir;

// Note: unwrap() and wrap() for MlirContext, MlirStringRef, etc. are already
// provided by mlir/CAPI/IR.h and mlir/CAPI/Support.h, so we use those directly

//===----------------------------------------------------------------------===//
// Operation Introspection
//===----------------------------------------------------------------------===//

size_t mlirEnumerateDialectOperations(
    MlirContext ctx,
    MlirStringRef dialectNamespace,
    MlirOperationEnumeratorCallback callback,
    void* userData
) {
    MLIRContext* context = unwrap(ctx);
    llvm::StringRef ns = unwrap(dialectNamespace);

    Dialect* dialect = context->getLoadedDialect(ns);
    if (!dialect) {
        return 0;
    }

    size_t count = 0;

    // Enumerate all registered operations in this dialect
    for (const auto& registeredOp : context->getRegisteredOperations()) {
        llvm::StringRef opName = registeredOp.getStringRef();

        // Check if this operation belongs to our dialect by parsing the name
        size_t dotPos = opName.find('.');
        if (dotPos != llvm::StringRef::npos) {
            llvm::StringRef opDialect = opName.take_front(dotPos);
            if (opDialect == ns) {
                count++;
                MlirStringRef wrappedName = wrap(opName);
                if (!callback(wrappedName, userData)) {
                    break;
                }
            }
        }
    }

    return count;
}

bool mlirOperationBelongsToDialect(
    MlirContext ctx,
    MlirStringRef opName,
    MlirStringRef dialectNamespace
) {
    MLIRContext* context = unwrap(ctx);
    llvm::StringRef name = unwrap(opName);
    llvm::StringRef ns = unwrap(dialectNamespace);

    // Check if the operation is registered
    std::optional<RegisteredOperationName> registeredOp =
        RegisteredOperationName::lookup(name, context);

    if (!registeredOp) {
        return false;
    }

    // Check if it belongs to the requested dialect
    Dialect& dialect = registeredOp->getDialect();
    return dialect.getNamespace() == ns;
}

bool mlirDialectHasOperation(
    MlirContext ctx,
    MlirStringRef dialectNamespace,
    MlirStringRef opName
) {
    MLIRContext* context = unwrap(ctx);
    llvm::StringRef ns = unwrap(dialectNamespace);
    llvm::StringRef name = unwrap(opName);

    // Build the full operation name
    std::string fullName = ns.str() + "." + name.str();

    // Check if it's registered
    std::optional<RegisteredOperationName> registeredOp =
        RegisteredOperationName::lookup(fullName, context);

    return registeredOp.has_value();
}

MlirStringRef mlirOperationGetDialectNamespace(MlirStringRef opName) {
    llvm::StringRef name = unwrap(opName);
    size_t dotPos = name.find('.');

    if (dotPos == llvm::StringRef::npos) {
        return mlirStringRefCreateFromCString("");
    }

    llvm::StringRef dialectNs = name.take_front(dotPos);
    return wrap(dialectNs);
}

MlirStringRef mlirOperationGetShortName(MlirStringRef opName) {
    llvm::StringRef name = unwrap(opName);
    size_t dotPos = name.find('.');

    if (dotPos == llvm::StringRef::npos) {
        return opName;
    }

    llvm::StringRef shortName = name.drop_front(dotPos + 1);
    return wrap(shortName);
}

//===----------------------------------------------------------------------===//
// Type Introspection
//===----------------------------------------------------------------------===//

size_t mlirEnumerateDialectTypes(
    MlirContext ctx,
    MlirStringRef dialectNamespace,
    MlirTypeEnumeratorCallback callback,
    void* userData
) {
    MLIRContext* context = unwrap(ctx);
    llvm::StringRef ns = unwrap(dialectNamespace);

    Dialect* dialect = context->getLoadedDialect(ns);
    if (!dialect) {
        return 0;
    }

    size_t count = 0;

    // Note: MLIR doesn't provide a direct way to enumerate all registered types
    // This is a limitation of the C++ API as well
    // We would need to use the dialect's internal registry if exposed

    // For now, we return 0 as this requires dialect-specific knowledge
    // Individual dialects would need to expose their type lists

    return count;
}

bool mlirTypeBelongsToDialect(
    MlirType type,
    MlirStringRef dialectNamespace
) {
    Type cppType = unwrap(type);
    llvm::StringRef ns = unwrap(dialectNamespace);

    Dialect& dialect = cppType.getDialect();
    return dialect.getNamespace() == ns;
}

//===----------------------------------------------------------------------===//
// Attribute Introspection
//===----------------------------------------------------------------------===//

size_t mlirEnumerateDialectAttributes(
    MlirContext ctx,
    MlirStringRef dialectNamespace,
    MlirAttributeEnumeratorCallback callback,
    void* userData
) {
    MLIRContext* context = unwrap(ctx);
    llvm::StringRef ns = unwrap(dialectNamespace);

    Dialect* dialect = context->getLoadedDialect(ns);
    if (!dialect) {
        return 0;
    }

    size_t count = 0;

    // Similar to types, attribute enumeration is not directly exposed
    // This would require dialect-specific knowledge

    return count;
}

bool mlirAttributeBelongsToDialect(
    MlirAttribute attr,
    MlirStringRef dialectNamespace
) {
    Attribute cppAttr = unwrap(attr);
    llvm::StringRef ns = unwrap(dialectNamespace);

    Dialect& dialect = cppAttr.getDialect();
    return dialect.getNamespace() == ns;
}

//===----------------------------------------------------------------------===//
// Dialect Enumeration
//===----------------------------------------------------------------------===//

size_t mlirEnumerateLoadedDialects(
    MlirContext ctx,
    MlirDialectEnumeratorCallback callback,
    void* userData
) {
    MLIRContext* context = unwrap(ctx);

    size_t count = 0;
    for (Dialect* dialect : context->getLoadedDialects()) {
        count++;
        MlirStringRef ns = wrap(dialect->getNamespace());
        if (!callback(ns, userData)) {
            break;
        }
    }

    return count;
}

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

void mlirDumpDialectOperations(MlirContext ctx, MlirStringRef dialectNamespace) {
    llvm::StringRef ns = unwrap(dialectNamespace);

    std::cout << "Operations in dialect '" << ns.str() << "':\n";

    auto printOp = [](MlirStringRef opName, void* userData) -> bool {
        llvm::StringRef name = unwrap(opName);
        std::cout << "  " << name.str() << "\n";
        return true;
    };

    size_t count = mlirEnumerateDialectOperations(ctx, dialectNamespace, printOp, nullptr);
    std::cout << "Total: " << count << " operations\n";
}

void mlirDumpDialectTypes(MlirContext ctx, MlirStringRef dialectNamespace) {
    llvm::StringRef ns = unwrap(dialectNamespace);
    std::cout << "Types in dialect '" << ns.str() << "': (enumeration not yet implemented)\n";
}

void mlirDumpDialectAttributes(MlirContext ctx, MlirStringRef dialectNamespace) {
    llvm::StringRef ns = unwrap(dialectNamespace);
    std::cout << "Attributes in dialect '" << ns.str() << "': (enumeration not yet implemented)\n";
}

void mlirDumpAllDialects(MlirContext ctx) {
    std::cout << "=== Loaded Dialects ===\n";

    auto printDialect = [](MlirStringRef dialectNs, void* userData) -> bool {
        MlirContext ctx = *static_cast<MlirContext*>(userData);
        std::cout << "\n";
        mlirDumpDialectOperations(ctx, dialectNs);
        return true;
    };

    mlirEnumerateLoadedDialects(ctx, printDialect, &ctx);
}
