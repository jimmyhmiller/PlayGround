// Thin C wrapper to create an MLIR ExecutionEngine with native target CPU
#include "mlir-c/ExecutionEngine.h"
#include "mlir/CAPI/ExecutionEngine.h"
#include "mlir/CAPI/IR.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Support/TargetSelect.h"

extern "C" MlirExecutionEngine mlirExecutionEngineCreateNative(
    MlirModule op, int optLevel, int numPaths,
    const MlirStringRef *sharedLibPaths, bool enableObjectDump) {

    // Get the native target triple and CPU
    auto triple = llvm::sys::getDefaultTargetTriple();
    auto cpu = llvm::sys::getHostCPUName();

    // TODO: This would need access to MLIR internals to create a TargetMachine
    // For now, just use the standard create path
    // The real fix is to set target-cpu on each function as an attribute
    return mlirExecutionEngineCreate(op, optLevel, numPaths, sharedLibPaths, enableObjectDump);
}
