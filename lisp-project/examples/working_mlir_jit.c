#include <stdio.h>
#include <stdint.h>

#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/ExecutionEngine.h"
#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/RegisterEverything.h"

typedef struct {
    void (*die)(uint8_t*);
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static void die(uint8_t*);
static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->die = &die;
    ns->main_fn = &main_fn;
}

static void die(uint8_t* msg) {
    printf("%s\n", msg);
    fflush(stderr);
    exit(1);
}
static int32_t main_fn() {
    return ({ MlirContext ctx = mlirContextCreate(); ({ MlirDialectRegistry registry = mlirDialectRegistryCreate(); mlirRegisterAllDialects(registry); mlirContextAppendDialectRegistry(ctx, registry); mlirDialectRegistryDestroy(registry); mlirContextLoadAllAvailableDialects(ctx); mlirRegisterAllPasses(); mlirRegisterAllLLVMTranslations(ctx); ({ uint8_t* mlirText = (uint8_t*)"module {\n  func.func @add(%arg0: i32, %arg1: i32) -> i32 {\n    %0 = arith.addi %arg0, %arg1 : i32\n    return %0 : i32\n  }\n}\n"; ({ MlirModule mod = mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(mlirText)); ({ if ((mlirModuleIsNull(mod) == 1)) { g_user.die("Failed to parse MLIR module"); } else { } }); ({ MlirPassManager pm = mlirPassManagerCreate(ctx); ({ MlirOpPassManager opm = mlirPassManagerGetAsOpPassManager(pm); ({ uint8_t* pipeline = (uint8_t*)"builtin.module(func.func(convert-arith-to-llvm),convert-func-to-llvm,reconcile-unrealized-casts)"; ({ if ((mlirLogicalResultIsFailure(mlirParsePassPipeline(opm, mlirStringRefCreateFromCString(pipeline), NULL, NULL)) == 1)) { g_user.die("Failed to parse pass pipeline"); } else { } }); ({ MlirOperation moduleOp = mlirModuleGetOperation(mod); ({ if ((mlirLogicalResultIsFailure(mlirPassManagerRunOnOp(pm, moduleOp)) == 1)) { g_user.die("Pass pipeline failed"); } else { } }); mlirPassManagerDestroy(pm); printf("=== Lowered MLIR ===\n"); mlirOperationDump(moduleOp); printf("===================\n"); ({ MlirExecutionEngine engine = mlirExecutionEngineCreate(mod, 3, 0, NULL, 0); ({ if ((mlirExecutionEngineIsNull(engine) == 1)) { g_user.die("Failed to create execution engine"); } else { } }); ({ void* fnPtr = (void*)mlirExecutionEngineLookup(engine, mlirStringRefCreateFromCString("add")); ({ if ((fnPtr == NULL)) { g_user.die("Failed to lookup function"); } else { } }); ({ int32_t (*add_fn)(int32_t, int32_t) = (int32_t (*)(int32_t, int32_t))((int32_t (*)(int32_t, int32_t))fnPtr); ({ int32_t a = 40; ({ int32_t b = 2; ({ int32_t result = (*add_fn)(a, b); printf("add(%d, %d) = %d\n", a, b, result); mlirExecutionEngineDestroy(engine); mlirContextDestroy(ctx); 0; }); }); }); }); }); }); }); }); }); }); }); }); }); });
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
