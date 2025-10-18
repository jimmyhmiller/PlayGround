;; mlir_jit_proper_builder.lisp
;; MLIR JIT example using PROPER programmatic IR building with MlirOperationState
;; This demonstrates the low-level C API for building MLIR operations without text parsing

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

;; Include headers
(include-header "stdint.h")
(include-header "stdio.h")
(include-header "stdlib.h")
(include-header "mlir-c/BuiltinAttributes.h")
(include-header "mlir-c/BuiltinTypes.h")
(include-header "mlir-c/ExecutionEngine.h")
(include-header "mlir-c/IR.h")
(include-header "mlir-c/Pass.h")
(include-header "mlir-c/RegisterEverything.h")

;; Declare opaque MLIR types
(declare-type MlirContext)
(declare-type MlirDialectRegistry)
(declare-type MlirLocation)
(declare-type MlirModule)
(declare-type MlirOperation)
(declare-type MlirRegion)
(declare-type MlirBlock)
(declare-type MlirValue)
(declare-type MlirType)
(declare-type MlirAttribute)
(declare-type MlirIdentifier)
(declare-type MlirPassManager)
(declare-type MlirOpPassManager)
(declare-type MlirExecutionEngine)
(declare-type MlirStringRef)
(declare-type MlirLogicalResult)
(declare-type MlirNamedAttribute)

;; Declare MlirOperationState as opaque type (defined in header)
(declare-type MlirOperationState)

;; Declare MLIR C API functions
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

(declare-fn mlirModuleCreateEmpty [loc MlirLocation] -> MlirModule)
(declare-fn mlirModuleGetOperation [mod MlirModule] -> MlirOperation)

(declare-fn mlirOperationGetRegion [op MlirOperation index I64] -> MlirRegion)
(declare-fn mlirOperationGetResult [op MlirOperation index I64] -> MlirValue)
(declare-fn mlirOperationIsNull [op MlirOperation] -> I8)
(declare-fn mlirOperationDump [op MlirOperation] -> Nil)

(declare-fn mlirOperationStateGet [name MlirStringRef loc MlirLocation] -> MlirOperationState)
(declare-fn mlirOperationStateAddOperands [state (Pointer MlirOperationState) n I64 operands (Pointer MlirValue)] -> Nil)
(declare-fn mlirOperationStateAddResults [state (Pointer MlirOperationState) n I64 results (Pointer MlirType)] -> Nil)
(declare-fn mlirOperationStateAddOwnedRegions [state (Pointer MlirOperationState) n I64 regions (Pointer MlirRegion)] -> Nil)
(declare-fn mlirOperationStateAddAttributes [state (Pointer MlirOperationState) n I64 attrs (Pointer MlirNamedAttribute)] -> Nil)
(declare-fn mlirOperationCreate [state (Pointer MlirOperationState)] -> MlirOperation)

(declare-fn mlirRegionCreate [] -> MlirRegion)
(declare-fn mlirRegionAppendOwnedBlock [region MlirRegion block MlirBlock] -> Nil)
(declare-fn mlirRegionGetFirstBlock [region MlirRegion] -> MlirBlock)

(declare-fn mlirBlockCreate [nArgs I64 argTypes (Pointer MlirType) locs (Pointer MlirLocation)] -> MlirBlock)
(declare-fn mlirBlockAppendOwnedOperation [block MlirBlock op MlirOperation] -> Nil)
(declare-fn mlirBlockGetArgument [block MlirBlock pos I64] -> MlirValue)

(declare-fn mlirIntegerTypeGet [ctx MlirContext bitwidth U32] -> MlirType)
(declare-fn mlirFunctionTypeGet [ctx MlirContext numInputs I64 inputs (Pointer MlirType) numResults I64 results (Pointer MlirType)] -> MlirType)

(declare-fn mlirIdentifierGet [ctx MlirContext str MlirStringRef] -> MlirIdentifier)
(declare-fn mlirNamedAttributeGet [name MlirIdentifier attr MlirAttribute] -> MlirNamedAttribute)
(declare-fn mlirStringAttrGet [ctx MlirContext str MlirStringRef] -> MlirAttribute)
(declare-fn mlirTypeAttrGet [type MlirType] -> MlirAttribute)

(declare-fn mlirPassManagerCreate [ctx MlirContext] -> MlirPassManager)
(declare-fn mlirPassManagerDestroy [pm MlirPassManager] -> Nil)
(declare-fn mlirPassManagerGetAsOpPassManager [pm MlirPassManager] -> MlirOpPassManager)
(declare-fn mlirPassManagerRunOnOp [pm MlirPassManager op MlirOperation] -> MlirLogicalResult)
(declare-fn mlirParsePassPipeline [opm MlirOpPassManager pipeline MlirStringRef callback (Pointer Nil) userData (Pointer Nil)] -> MlirLogicalResult)
(declare-fn mlirLogicalResultIsFailure [res MlirLogicalResult] -> I8)

(declare-fn mlirExecutionEngineCreate [mod MlirModule optLevel I32 numPaths I32 sharedLibPaths (Pointer Nil) enableObjectDump I8] -> MlirExecutionEngine)
(declare-fn mlirExecutionEngineIsNull [engine MlirExecutionEngine] -> I8)
(declare-fn mlirExecutionEngineDestroy [engine MlirExecutionEngine] -> Nil)
(declare-fn mlirExecutionEngineLookup [engine MlirExecutionEngine name MlirStringRef] -> (Pointer Nil))

(declare-fn printf [fmt (Pointer U8)] -> I32)
(declare-fn fprintf [stream (Pointer Nil) fmt (Pointer U8)] -> I32)
(declare-fn fflush [stream (Pointer Nil)] -> I32)
(declare-fn exit [code I32] -> Nil)
(declare-var stderr (Pointer Nil))

;; Helper: die function
(def die (: (-> [(Pointer U8)] Nil))
  (fn [msg]
    (printf (c-str "ERROR: %s\n") msg)
    (fflush stderr)
    (exit 1)))

;; Main function
(def main-fn (: (-> [] I32))
  (fn []
    ;; ===== Setup =====
    (let [ctx (: MlirContext) (mlirContextCreate)
          registry (: MlirDialectRegistry) (mlirDialectRegistryCreate)]
      (mlirRegisterAllDialects registry) (mlirContextAppendDialectRegistry ctx registry) (mlirDialectRegistryDestroy registry) (mlirContextLoadAllAvailableDialects ctx) (mlirRegisterAllPasses) (mlirRegisterAllLLVMTranslations ctx) (let [loc (: MlirLocation) (mlirLocationUnknownGet ctx)]

        ;; Create empty module
        (let [mod (: MlirModule) (mlirModuleCreateEmpty loc)
              moduleOp (: MlirOperation) (mlirModuleGetOperation mod)
              moduleBody (: MlirRegion) (mlirOperationGetRegion moduleOp 0)
              moduleBlock (: MlirBlock) (mlirRegionGetFirstBlock moduleBody)]
          ;; ===== Build function type: (i32, i32) -> i32 =====
          (let [i32Type (: MlirType) (mlirIntegerTypeGet ctx 32)]

            ;; Create input types array
            (let [inputTypes (: (Array MlirType 2)) (array MlirType 2)]
              (array-set! inputTypes 0 i32Type)
              (array-set! inputTypes 1 i32Type)

              (let [funcType (: MlirType)
                    (mlirFunctionTypeGet ctx 2 (array-ptr inputTypes 0) 1 (address-of i32Type))]

                ;; ===== Create function body region and block =====
                (let [funcBodyRegion (: MlirRegion) (mlirRegionCreate)]

                  ;; Create entry block with 2 i32 arguments
                  (let [argTypes (: (Array MlirType 2)) (array MlirType 2)]
                    (array-set! argTypes 0 i32Type)
                    (array-set! argTypes 1 i32Type)

                    (let [argLocs (: (Array MlirLocation 2)) (array MlirLocation 2)]
                      (array-set! argLocs 0 loc)
                      (array-set! argLocs 1 loc)

                      (let [entryBlock (: MlirBlock)
                            (mlirBlockCreate 2 (array-ptr argTypes 0) (array-ptr argLocs 0))]
                        (mlirRegionAppendOwnedBlock funcBodyRegion entryBlock)

                        ;; Get block arguments
                        (let [arg0 (: MlirValue) (mlirBlockGetArgument entryBlock 0)
                              arg1 (: MlirValue) (mlirBlockGetArgument entryBlock 1)]
                          ;; ===== Build arith.addi operation =====
                          (let [addiState (: MlirOperationState)
                                (mlirOperationStateGet
                                  (mlirStringRefCreateFromCString (c-str "arith.addi"))
                                  loc)]

                            ;; Add operands
                            (let [addiOperands (: (Array MlirValue 2)) (array MlirValue 2)]
                              (array-set! addiOperands 0 arg0)
                              (array-set! addiOperands 1 arg1)
                              (mlirOperationStateAddOperands
                                (address-of addiState)
                                2
                                (array-ptr addiOperands 0))

                              ;; Add result type
                              (let [addiResultTypes (: (Array MlirType 1)) (array MlirType 1)]
                                (array-set! addiResultTypes 0 i32Type)
                                (mlirOperationStateAddResults
                                  (address-of addiState)
                                  1
                                  (array-ptr addiResultTypes 0))

                                ;; Create the addi operation
                                (let [addiOp (: MlirOperation) (mlirOperationCreate (address-of addiState))]
                                  (if (= (mlirOperationIsNull addiOp) 1)
                                    (die (c-str "Failed to create arith.addi operation"))
                                    nil)

                                  ;; Insert into block
                                  (mlirBlockAppendOwnedOperation entryBlock addiOp)

                                  ;; Get result value
                                  (let [addiResult (: MlirValue) (mlirOperationGetResult addiOp 0)]

                                    ;; ===== Build func.return operation =====
                                    (let [returnState (: MlirOperationState)
                                          (mlirOperationStateGet
                                            (mlirStringRefCreateFromCString (c-str "func.return"))
                                            loc)]

                                      ;; Add operand
                                      (let [returnOperands (: (Array MlirValue 1)) (array MlirValue 1)]
                                        (array-set! returnOperands 0 addiResult)
                                        (mlirOperationStateAddOperands
                                          (address-of returnState)
                                          1
                                          (array-ptr returnOperands 0))

                                        (let [returnOp (: MlirOperation) (mlirOperationCreate (address-of returnState))]
                                          (if (= (mlirOperationIsNull returnOp) 1)
                                            (die (c-str "Failed to create func.return operation"))
                                            nil)

                                          (mlirBlockAppendOwnedOperation entryBlock returnOp)

                                          ;; ===== Build func.func operation =====
                                          (let [funcState (: MlirOperationState)
                                                (mlirOperationStateGet
                                                  (mlirStringRefCreateFromCString (c-str "func.func"))
                                                  loc)]

                                            ;; Add the function body region
                                            (let [funcRegions (: (Array MlirRegion 1)) (array MlirRegion 1)]
                                              (array-set! funcRegions 0 funcBodyRegion)
                                              (mlirOperationStateAddOwnedRegions
                                                (address-of funcState)
                                                1
                                                (array-ptr funcRegions 0))

                                              ;; Add attributes: sym_name and function_type
                                              (let [symNameId (: MlirIdentifier) (mlirIdentifierGet ctx (mlirStringRefCreateFromCString (c-str "sym_name")))
                                                    symNameAttr (: MlirAttribute) (mlirStringAttrGet ctx (mlirStringRefCreateFromCString (c-str "add")))
                                                    symNameNamed (: MlirNamedAttribute) (mlirNamedAttributeGet symNameId symNameAttr)
                                                    funcTypeId (: MlirIdentifier) (mlirIdentifierGet ctx (mlirStringRefCreateFromCString (c-str "function_type")))
                                                    funcTypeAttr (: MlirAttribute) (mlirTypeAttrGet funcType)
                                                    funcTypeNamed (: MlirNamedAttribute) (mlirNamedAttributeGet funcTypeId funcTypeAttr)]
                                                ;; Create array of attributes
                                                (let [funcAttrs (: (Array MlirNamedAttribute 2)) (array MlirNamedAttribute 2)]
                                                  (array-set! funcAttrs 0 symNameNamed)
                                                  (array-set! funcAttrs 1 funcTypeNamed)
                                                  (mlirOperationStateAddAttributes
                                                    (address-of funcState)
                                                    2
                                                    (array-ptr funcAttrs 0))

                                                  ;; Create the func.func operation
                                                  (let [funcOp (: MlirOperation) (mlirOperationCreate (address-of funcState))]
                                                    (if (= (mlirOperationIsNull funcOp) 1)
                                                      (die (c-str "Failed to create func.func operation"))
                                                      nil)

                                                    ;; Insert function into module
                                                    (mlirBlockAppendOwnedOperation moduleBlock funcOp)

                                                    (printf (c-str "=== Built MLIR (before lowering) ===\n"))
                                                    (mlirOperationDump moduleOp)
                                                    (printf (c-str "====================================\n\n"))

                                                    ;; ===== Run lowering passes =====
                                                    (let [pm (: MlirPassManager) (mlirPassManagerCreate ctx)
                                                          opm (: MlirOpPassManager) (mlirPassManagerGetAsOpPassManager pm)
                                                          pipeline (: (Pointer U8)) (c-str "builtin.module(func.func(convert-arith-to-llvm),convert-func-to-llvm,reconcile-unrealized-casts)")]
                                                      (if (= (mlirLogicalResultIsFailure
                                                        (mlirParsePassPipeline opm (mlirStringRefCreateFromCString pipeline) pointer-null pointer-null)) 1)
                                                        (die (c-str "Failed to parse pass pipeline"))
                                                        nil) (if (= (mlirLogicalResultIsFailure (mlirPassManagerRunOnOp pm moduleOp)) 1)
                                                          (die (c-str "Pass pipeline failed"))
                                                          nil) (mlirPassManagerDestroy pm) (printf (c-str "=== Lowered MLIR ===\n")) (mlirOperationDump moduleOp) (printf (c-str "====================\n\n")) ;; ===== JIT and execute =====
                                                      (let [engine (: MlirExecutionEngine)
                                                            (mlirExecutionEngineCreate mod 3 0 pointer-null 0)]

                                                        (if (= (mlirExecutionEngineIsNull engine) 1)
                                                          (die (c-str "Failed to create execution engine"))
                                                          nil)

                                                        (let [fnPtr (: (Pointer Nil))
                                                              (mlirExecutionEngineLookup engine (mlirStringRefCreateFromCString (c-str "add")))]

                                                          (if (pointer-equal? fnPtr pointer-null)
                                                            (die (c-str "Failed to lookup function"))
                                                            nil)

                                                          ;; Cast and call function
                                                          (let [add-fn (: (Pointer (-> [I32 I32] I32))) (cast (Pointer (-> [I32 I32] I32)) fnPtr)
                                                                a (: I32) 40
                                                                b (: I32) 2
                                                                result (: I32) (add-fn a b)]
                                                            (printf (c-str "add(%d, %d) = %d\n") a b result) (mlirExecutionEngineDestroy engine) (mlirContextDestroy ctx) 0))))))))))))))))))))))))))))))

;; Call main function
(main-fn)