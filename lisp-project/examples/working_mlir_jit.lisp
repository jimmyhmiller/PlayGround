;; working_mlir_jit.lisp
;; A working MLIR JIT example - direct conversion from working_mlir_jit.c

;; Compiler flags for MLIR
(compiler-flag "-I/opt/homebrew/opt/llvm/include")
(compiler-flag "-L/opt/homebrew/opt/llvm/lib")

;; Link MLIR libraries
(link-library "MLIRCAPIIR")
(link-library "MLIRCAPIExecutionEngine")
(link-library "MLIRCAPIRegisterEverything")
(link-library "MLIRCAPIConversion")
(link-library "MLIRCAPITransforms")
(link-library "MLIRExecutionEngine")
(link-library "MLIRExecutionEngineUtils")
(link-library "MLIR")
(link-library "LLVM")
(link-library "c++")

;; Include all required headers
(include-header "stdint.h")
(include-header "stdio.h")
(include-header "stdlib.h")
(include-header "mlir-c/BuiltinAttributes.h")
(include-header "mlir-c/BuiltinTypes.h")
(include-header "mlir-c/ExecutionEngine.h")
(include-header "mlir-c/IR.h")
(include-header "mlir-c/Pass.h")
(include-header "mlir-c/RegisterEverything.h")

;; Declare opaque MLIR types (these are opaque structs passed by value in the C API)
(declare-type MlirContext)
(declare-type MlirDialectRegistry)
(declare-type MlirLocation)
(declare-type MlirModule)
(declare-type MlirPassManager)
(declare-type MlirOpPassManager)
(declare-type MlirOperation)
(declare-type MlirExecutionEngine)
(declare-type MlirStringRef)
(declare-type MlirLogicalResult)

;; Declare all MLIR C API functions
;; Note: MLIR types are passed by value, not as pointers!
(declare-fn mlirContextCreate [] -> MlirContext)
(declare-fn mlirContextDestroy [ctx MlirContext] -> Nil)
(declare-fn mlirContextAppendDialectRegistry [ctx MlirContext registry MlirDialectRegistry] -> Nil)
(declare-fn mlirContextLoadAllAvailableDialects [ctx MlirContext] -> Nil)

(declare-fn mlirDialectRegistryCreate [] -> MlirDialectRegistry)
(declare-fn mlirDialectRegistryDestroy [registry MlirDialectRegistry] -> Nil)
(declare-fn mlirRegisterAllDialects [registry MlirDialectRegistry] -> Nil)
(declare-fn mlirRegisterAllPasses [] -> Nil)
(declare-fn mlirRegisterAllLLVMTranslations [ctx MlirContext] -> Nil)

(declare-fn mlirStringRefCreateFromCString [str (Pointer U8)] -> MlirStringRef)
(declare-fn mlirLocationUnknownGet [ctx MlirContext] -> MlirLocation)

(declare-fn mlirModuleCreateParse [ctx MlirContext mlirText MlirStringRef] -> MlirModule)
(declare-fn mlirModuleIsNull [mod MlirModule] -> I8)
(declare-fn mlirModuleGetOperation [mod MlirModule] -> MlirOperation)

(declare-fn mlirPassManagerCreate [ctx MlirContext] -> MlirPassManager)
(declare-fn mlirPassManagerDestroy [pm MlirPassManager] -> Nil)
(declare-fn mlirPassManagerGetAsOpPassManager [pm MlirPassManager] -> MlirOpPassManager)
(declare-fn mlirPassManagerRunOnOp [pm MlirPassManager op MlirOperation] -> MlirLogicalResult)

(declare-fn mlirParsePassPipeline [opm MlirOpPassManager pipeline MlirStringRef callback (Pointer Nil) userData (Pointer Nil)] -> MlirLogicalResult)
(declare-fn mlirLogicalResultIsFailure [res MlirLogicalResult] -> I8)

(declare-fn mlirOperationDump [op MlirOperation] -> Nil)

(declare-fn mlirExecutionEngineCreate [mod MlirModule optLevel I32 numPaths I32 sharedLibPaths (Pointer Nil) enableObjectDump I8] -> MlirExecutionEngine)
(declare-fn mlirExecutionEngineIsNull [engine MlirExecutionEngine] -> I8)
(declare-fn mlirExecutionEngineDestroy [engine MlirExecutionEngine] -> Nil)
(declare-fn mlirExecutionEngineLookup [engine MlirExecutionEngine name MlirStringRef] -> (Pointer Nil))

;; Declare standard C functionsp
(declare-fn fflush [stream (Pointer Nil)] -> I32)
(declare-fn exit [code I32] -> Nil)
(declare-var stderr (Pointer Nil))

;; Helper: die function
(def die (: (-> [(Pointer U8)] Nil))
  (fn [msg]
    (printf (c-str "%s\n") msg)
    (fflush stderr)
    (exit 1)))

;; Main function
(def main-fn (: (-> [] I32))
  (fn []
    ;; Create MLIR context
    (let [ctx (: MlirContext) (mlirContextCreate)]

      ;; Register all dialects and passes
      (let [registry (: MlirDialectRegistry) (mlirDialectRegistryCreate)]
        (mlirRegisterAllDialects registry)
        (mlirContextAppendDialectRegistry ctx registry)
        (mlirDialectRegistryDestroy registry)
        (mlirContextLoadAllAvailableDialects ctx)

        ;; Register all passes (needed for pass pipeline parsing)
        (mlirRegisterAllPasses)

        ;; Register LLVM translations (needed for JIT execution)
        (mlirRegisterAllLLVMTranslations ctx)

        ;; Parse MLIR from string
        (let [mlirText (: (Pointer U8))
              (c-str "module {\n  func.func @add(%arg0: i32, %arg1: i32) -> i32 {\n    %0 = arith.addi %arg0, %arg1 : i32\n    return %0 : i32\n  }\n}\n")]
          (let [mod (: MlirModule)
                (mlirModuleCreateParse ctx (mlirStringRefCreateFromCString mlirText))]

            ;; Check if parsing failed
            (if (= (mlirModuleIsNull mod) 1)
              (die (c-str "Failed to parse MLIR module"))
              nil)

            ;; Create pass manager
            (let [pm (: MlirPassManager) (mlirPassManagerCreate ctx)]
              (let [opm (: MlirOpPassManager) (mlirPassManagerGetAsOpPassManager pm)]

                ;; Parse and run the lowering pipeline
                (let [pipeline (: (Pointer U8))
                      (c-str "builtin.module(func.func(convert-arith-to-llvm),convert-func-to-llvm,reconcile-unrealized-casts)")]

                  (if (= (mlirLogicalResultIsFailure
                          (mlirParsePassPipeline opm (mlirStringRefCreateFromCString pipeline) pointer-null pointer-null)) 1)
                    (die (c-str "Failed to parse pass pipeline"))
                    nil)

                  ;; Run passes on the module
                  (let [moduleOp (: MlirOperation) (mlirModuleGetOperation mod)]
                    (if (= (mlirLogicalResultIsFailure (mlirPassManagerRunOnOp pm moduleOp)) 1)
                      (die (c-str "Pass pipeline failed"))
                      nil)

                    (mlirPassManagerDestroy pm)

                    ;; Dump the lowered IR for debugging
                    (printf (c-str "=== Lowered MLIR ===\n"))
                    (mlirOperationDump moduleOp)
                    (printf (c-str "===================\n"))

                    ;; Create execution engine
                    (let [engine (: MlirExecutionEngine)
                          (mlirExecutionEngineCreate mod 3 0 pointer-null 0)]

                      (if (= (mlirExecutionEngineIsNull engine) 1)
                        (die (c-str "Failed to create execution engine"))
                        nil)

                      ;; Look up the function
                      (let [fnPtr (: (Pointer Nil))
                            (mlirExecutionEngineLookup engine (mlirStringRefCreateFromCString (c-str "add")))]

                        (if (pointer-equal? fnPtr pointer-null)
                          (die (c-str "Failed to lookup function"))
                          nil)

                        ;; Cast to function pointer and call it
                        ;; typedef int32_t (*AddFn)(int32_t, int32_t);
                        (let [add-fn (: (Pointer (-> [I32 I32] I32)))
                              (cast (Pointer (-> [I32 I32] I32)) fnPtr)]
                          (let [a (: I32) 40]
                            (let [b (: I32) 2]
                              (let [result (: I32) (add-fn a b)]
                                (printf (c-str "add(%d, %d) = %d\n") a b result)
                                (mlirExecutionEngineDestroy engine)
                                (mlirContextDestroy ctx)
                                0))))))))))))))))


(main-fn)
