;; mlir_builder.lisp
;; MLIR Builder - Convert OpNode/BlockNode structures to MLIR C API operations

(ns mlir-builder)

(require [types :as types])
(require [mlir-ast :as ast])

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
(include-header "string.h")
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
(declare-type MlirOperationState)

;; Declare MLIR C API functions
(declare-fn printf [fmt (Pointer U8)] -> I32)
(declare-fn fprintf [stream (Pointer Nil) fmt (Pointer U8)] -> I32)
(declare-fn fflush [stream (Pointer Nil)] -> I32)
(declare-fn exit [code I32] -> Nil)
(declare-fn strcmp [s1 (Pointer U8) s2 (Pointer U8)] -> I32)
(declare-fn atoi [str (Pointer U8)] -> I32)
(declare-fn malloc [size I32] -> (Pointer U8))
(declare-var stderr (Pointer Nil))

;; Context management
(declare-fn mlirContextCreate [] -> MlirContext)
(declare-fn mlirContextDestroy [ctx MlirContext] -> Nil)
(declare-fn mlirContextAppendDialectRegistry [ctx MlirContext registry MlirDialectRegistry] -> Nil)
(declare-fn mlirContextLoadAllAvailableDialects [ctx MlirContext] -> Nil)

;; Dialect registry
(declare-fn mlirDialectRegistryCreate [] -> MlirDialectRegistry)
(declare-fn mlirDialectRegistryDestroy [registry MlirDialectRegistry] -> Nil)
(declare-fn mlirRegisterAllDialects [registry MlirDialectRegistry] -> Nil)
(declare-fn mlirRegisterAllPasses [] -> Nil)
(declare-fn mlirRegisterAllLLVMTranslations [ctx MlirContext] -> Nil)

;; String and location
(declare-fn mlirStringRefCreateFromCString [str (Pointer U8)] -> MlirStringRef)
(declare-fn mlirLocationUnknownGet [ctx MlirContext] -> MlirLocation)

;; Module
(declare-fn mlirModuleCreateEmpty [loc MlirLocation] -> MlirModule)
(declare-fn mlirModuleGetOperation [mod MlirModule] -> MlirOperation)

;; Operation
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

;; Region
(declare-fn mlirRegionCreate [] -> MlirRegion)
(declare-fn mlirRegionAppendOwnedBlock [region MlirRegion block MlirBlock] -> Nil)
(declare-fn mlirRegionGetFirstBlock [region MlirRegion] -> MlirBlock)

;; Block
(declare-fn mlirBlockCreate [nArgs I64 argTypes (Pointer MlirType) locs (Pointer MlirLocation)] -> MlirBlock)
(declare-fn mlirBlockAppendOwnedOperation [block MlirBlock op MlirOperation] -> Nil)
(declare-fn mlirBlockGetArgument [block MlirBlock pos I64] -> MlirValue)

;; Types
(declare-fn mlirIntegerTypeGet [ctx MlirContext bitwidth U32] -> MlirType)
(declare-fn mlirFunctionTypeGet [ctx MlirContext numInputs I64 inputs (Pointer MlirType) numResults I64 results (Pointer MlirType)] -> MlirType)

;; Attributes
(declare-fn mlirIdentifierGet [ctx MlirContext str MlirStringRef] -> MlirIdentifier)
(declare-fn mlirNamedAttributeGet [name MlirIdentifier attr MlirAttribute] -> MlirNamedAttribute)
(declare-fn mlirStringAttrGet [ctx MlirContext str MlirStringRef] -> MlirAttribute)
(declare-fn mlirTypeAttrGet [type MlirType] -> MlirAttribute)
(declare-fn mlirIntegerAttrGet [type MlirType value I64] -> MlirAttribute)

;; Pass management
(declare-fn mlirPassManagerCreate [ctx MlirContext] -> MlirPassManager)
(declare-fn mlirPassManagerDestroy [pm MlirPassManager] -> Nil)
(declare-fn mlirPassManagerGetAsOpPassManager [pm MlirPassManager] -> MlirOpPassManager)
(declare-fn mlirPassManagerRunOnOp [pm MlirPassManager op MlirOperation] -> MlirLogicalResult)
(declare-fn mlirParsePassPipeline [opm MlirOpPassManager pipeline MlirStringRef callback (Pointer Nil) userData (Pointer Nil)] -> MlirLogicalResult)
(declare-fn mlirLogicalResultIsFailure [res MlirLogicalResult] -> I8)

;; Execution engine
(declare-fn mlirExecutionEngineCreate [mod MlirModule optLevel I32 numPaths I32 sharedLibPaths (Pointer Nil) enableObjectDump I8] -> MlirExecutionEngine)
(declare-fn mlirExecutionEngineIsNull [engine MlirExecutionEngine] -> I8)
(declare-fn mlirExecutionEngineDestroy [engine MlirExecutionEngine] -> Nil)
(declare-fn mlirExecutionEngineLookup [engine MlirExecutionEngine name MlirStringRef] -> (Pointer Nil))

;; Builder context - holds all the MLIR state we need
(def MLIRBuilderContext (: Type)
  (Struct
    [ctx MlirContext]
    [loc MlirLocation]
    [mod MlirModule]
    [i32Type MlirType]
    [i64Type MlirType]))

;; Helper: die function
(def die (: (-> [(Pointer U8)] Nil))
  (fn [msg]
    (printf (c-str "ERROR: %s\n") msg)
    (fflush stderr)
    (exit 1)))

;; Initialize MLIR builder context
(def mlir-builder-init (: (-> [] (Pointer MLIRBuilderContext)))
  (fn []
    (let [ctx (: MlirContext) (mlirContextCreate)
          registry (: MlirDialectRegistry) (mlirDialectRegistryCreate)]
      (mlirRegisterAllDialects registry)
      (mlirContextAppendDialectRegistry ctx registry)
      (mlirDialectRegistryDestroy registry)
      (mlirContextLoadAllAvailableDialects ctx)
      (mlirRegisterAllPasses)
      (mlirRegisterAllLLVMTranslations ctx)

      (let [loc (: MlirLocation) (mlirLocationUnknownGet ctx)
            mod (: MlirModule) (mlirModuleCreateEmpty loc)
            i32Type (: MlirType) (mlirIntegerTypeGet ctx 32)
            i64Type (: MlirType) (mlirIntegerTypeGet ctx 64)
            builder (: (Pointer MLIRBuilderContext)) (cast (Pointer MLIRBuilderContext) (malloc 40))]
        (pointer-field-write! builder ctx ctx)
        (pointer-field-write! builder loc loc)
        (pointer-field-write! builder mod mod)
        (pointer-field-write! builder i32Type i32Type)
        (pointer-field-write! builder i64Type i64Type)
        builder))))

;; Destroy builder context
(def mlir-builder-destroy (: (-> [(Pointer MLIRBuilderContext)] I32))
  (fn [builder]
    (let [ctx (: MlirContext) (pointer-field-read builder ctx)]
      (mlirContextDestroy ctx)
      0)))

;; SSA Value Tracker - maps SSA indices to MlirValue handles
;; Using fixed-size array for simplicity
(def MAX_VALUES (: I32) 256)

(def ValueTracker (: Type)
  (Struct
    [values (Array MlirValue 256)]  ; Fixed array of values
    [count I32]))                    ; Number of values

;; Create a new value tracker
(def value-tracker-create (: (-> [] (Pointer ValueTracker)))
  (fn []
    (let [tracker (: (Pointer ValueTracker)) (cast (Pointer ValueTracker) (malloc 2056))]
      (pointer-field-write! tracker count 0)
      tracker)))

;; Register a value in the tracker, returns its index
(def value-tracker-register (: (-> [(Pointer ValueTracker) MlirValue] I32))
  (fn [tracker value]
    (let [count (: I32) (pointer-field-read tracker count)]
      ;; Get pointer to the array field
      (let [values-ptr (: (Pointer MlirValue)) (array-ptr (pointer-field-read tracker values) 0)]
        ;; Write value at current count index
        (let [target-ptr (: (Pointer MlirValue)) (cast (Pointer MlirValue) (+ (cast I64 values-ptr) (* (cast I64 count) 8)))]
          (pointer-write! target-ptr value)
          (pointer-field-write! tracker count (+ count 1))
          count)))))

;; Lookup a value by index
(def value-tracker-lookup (: (-> [(Pointer ValueTracker) I32] MlirValue))
  (fn [tracker idx]
    (let [values-ptr (: (Pointer MlirValue)) (array-ptr (pointer-field-read tracker values) 0)
          target-ptr (: (Pointer MlirValue)) (cast (Pointer MlirValue) (+ (cast I64 values-ptr) (* (cast I64 idx) 8)))]
      (dereference target-ptr))))

;; Parse type string to MlirType (e.g., "i32" -> integer type)
;; NOTE: Workaround for compiler bug with nested if in return position
;; Uses separate helper function to avoid nesting
(def parse-type-string-helper (: (-> [(Pointer MLIRBuilderContext) (Pointer U8)] MlirType))
  (fn [builder type-str]
    (let [cmp-i64 (: I32) (strcmp type-str (c-str "i64"))]
      (if (== cmp-i64 0)
        (pointer-field-read builder i64Type)
        (let [_ (: I32) (printf (c-str "Unknown type: %s, defaulting to i32\n") type-str)]
          (pointer-field-read builder i32Type))))))

(def parse-type-string (: (-> [(Pointer MLIRBuilderContext) (Pointer U8)] MlirType))
  (fn [builder type-str]
    (let [cmp-i32 (: I32) (strcmp type-str (c-str "i32"))]
      (if (== cmp-i32 0)
        (pointer-field-read builder i32Type)
        (parse-type-string-helper builder type-str)))))

;; Parse integer value from string like "42 : i32"
(def parse-int-attr-value (: (-> [(Pointer U8)] I64))
  (fn [value-str]
    ;; For now, simple atoi - will need more complex parsing later
    (cast I64 (atoi value-str))))

;; Parse integer value attribute like (1 i32) or (42 i64)
;; value-list is a List (cons cell) of the form (int-value type-symbol)
(def parse-integer-value-attr (: (-> [(Pointer MLIRBuilderContext) (Pointer types/Value)] MlirAttribute))
  (fn [builder value-list]
    (printf (c-str "    DEBUG: Entering parse-integer-value-attr\n"))
    ;; Use car/cdr to access list elements
    (let [first-elem (: (Pointer types/Value)) (types/car value-list)]
      (printf (c-str "    DEBUG: Got first elem: %p\n") (cast (Pointer U8) first-elem))
      (let [rest (: (Pointer types/Value)) (types/cdr value-list)]
        (printf (c-str "    DEBUG: Got rest: %p\n") (cast (Pointer U8) rest))
        (let [rest-tag (: types/ValueTag) (pointer-field-read rest tag)]
          (printf (c-str "    DEBUG: rest tag = %d\n") (cast I32 rest-tag))
          (if (= rest-tag types/ValueTag/List)
            (let [second-elem (: (Pointer types/Value)) (types/car rest)
                  int-str (: (Pointer U8)) (pointer-field-read first-elem str_val)
                  int-val (: I64) (cast I64 (atoi int-str))
                  type-str (: (Pointer U8)) (pointer-field-read second-elem str_val)
                  mlir-type (: MlirType) (parse-type-string builder type-str)]
              (printf (c-str "    Parsed integer attr: %lld : %s\n") int-val type-str)
              (mlirIntegerAttrGet mlir-type int-val))
            (let [ctx (: MlirContext) (pointer-field-read builder ctx)]
              (printf (c-str "    ERROR: Invalid integer value format, rest-tag=%d\n") (cast I32 rest-tag))
              (mlirStringAttrGet ctx (mlirStringRefCreateFromCString (c-str "ERROR"))))))))))

;; Parse function type attribute like (-> [i32] [i32])
(def parse-function-type-attr (: (-> [(Pointer MLIRBuilderContext) (Pointer types/Value)] MlirAttribute))
  (fn [builder value-list]
    (let [ctx (: MlirContext) (pointer-field-read builder ctx)]
      ;; TODO: Properly parse function type - for now create placeholder TypeAttr
      (printf (c-str "    WARNING: function_type parsing not fully implemented\n"))
      ;; For now, create a simple function type with i32 -> i32
      (let [i32-type (: MlirType) (pointer-field-read builder i32Type)
            inputs (: (Array MlirType 1)) (array MlirType 1)
            results (: (Array MlirType 1)) (array MlirType 1)]
        (array-set! inputs 0 i32-type)
        (array-set! results 0 i32-type)
        (let [fn-type (: MlirType) (mlirFunctionTypeGet ctx 1 (array-ptr inputs 0) 1 (array-ptr results 0))]
          (mlirTypeAttrGet fn-type))))))

;; Create attribute from key-value pair where value is a Value (can be string or list)
(def create-attribute-from-value (: (-> [(Pointer MLIRBuilderContext) (Pointer U8) (Pointer types/Value)] MlirAttribute))
  (fn [builder key value-val]
    (let [ctx (: MlirContext) (pointer-field-read builder ctx)
          value-tag (: types/ValueTag) (pointer-field-read value-val tag)]
      (printf (c-str "  Creating attribute: %s (tag=%d)\n") key (cast I32 value-tag))

      ;; Check if value is a string or a list
      (if (= value-tag types/ValueTag/String)
        ;; String attribute
        (let [value-str (: (Pointer U8)) (pointer-field-read value-val str_val)]
          (if (== (strcmp key (c-str "predicate")) 0)
            ;; Predicate attribute for arith.cmpi - convert string to integer enum
            (let [pred-val (: I64)
                  (if (== (strcmp value-str (c-str "sle")) 0) 3
                    (if (== (strcmp value-str (c-str "eq")) 0) 0
                      (if (== (strcmp value-str (c-str "ne")) 0) 1
                        (if (== (strcmp value-str (c-str "slt")) 0) 2
                          (if (== (strcmp value-str (c-str "sgt")) 0) 4
                            (if (== (strcmp value-str (c-str "sge")) 0) 5
                              (if (== (strcmp value-str (c-str "ult")) 0) 6
                                (if (== (strcmp value-str (c-str "ule")) 0) 7
                                  (if (== (strcmp value-str (c-str "ugt")) 0) 8
                                    (if (== (strcmp value-str (c-str "uge")) 0) 9 0))))))))))
                  i64-type (: MlirType) (pointer-field-read builder i64Type)]
              (mlirIntegerAttrGet i64-type pred-val))
            ;; For all other string attributes
            (mlirStringAttrGet ctx (mlirStringRefCreateFromCString value-str))))

        ;; Check for List (cons cell structure from parenthesized expressions)
        (if (= value-tag types/ValueTag/List)
          ;; List attribute - check what kind
          (if (== (strcmp key (c-str "function_type")) 0)
            (parse-function-type-attr builder value-val)
            (if (== (strcmp key (c-str "value")) 0)
              (parse-integer-value-attr builder value-val)
              (if (== (strcmp key (c-str "callee")) 0)
                ;; callee is a symbol reference - extract first element from list using car
                (let [first-elem (: (Pointer types/Value)) (types/car value-val)
                      str-val (: (Pointer U8)) (pointer-field-read first-elem str_val)]
                  (mlirStringAttrGet ctx (mlirStringRefCreateFromCString str-val)))
                ;; Unknown list attribute
                (let [_ (: I32) (printf (c-str "    WARNING: Unknown list attribute type\n"))]
                  (mlirStringAttrGet ctx (mlirStringRefCreateFromCString (c-str "TODO")))))))

          ;; Unknown value type
          (let [_ (: I32) (printf (c-str "    ERROR: Unknown attribute value type\n"))]
            (mlirStringAttrGet ctx (mlirStringRefCreateFromCString (c-str "ERROR")))))))))

;; Create named attribute from a Value
(def create-named-attribute-from-value (: (-> [(Pointer MLIRBuilderContext) (Pointer U8) (Pointer types/Value)] MlirNamedAttribute))
  (fn [builder key value-val]
    (let [ctx (: MlirContext) (pointer-field-read builder ctx)
          name-id (: MlirIdentifier) (mlirIdentifierGet ctx (mlirStringRefCreateFromCString key))
          attr (: MlirAttribute) (create-attribute-from-value builder key value-val)]
      (mlirNamedAttributeGet name-id attr))))

;; Note: Forward declarations not needed - two-pass type checking handles mutual recursion automatically

;; Build an MLIR region from a region vector (vector of blocks)
(def build-mlir-region (: (-> [(Pointer MLIRBuilderContext) (Pointer types/Value) (Pointer ValueTracker)] MlirRegion))
  (fn [builder region-vec tracker]
    (let [region (: MlirRegion) (mlirRegionCreate)
          tag (: types/ValueTag) (pointer-field-read region-vec tag)]
      ;; region-vec should be a vector of blocks
      (if (= tag types/ValueTag/Vector)
        (let [vec-ptr (: (Pointer U8)) (pointer-field-read region-vec vec_val)
              vector-struct (: (Pointer types/Vector)) (cast (Pointer types/Vector) vec-ptr)
              count (: I32) (pointer-field-read vector-struct count)
              data (: (Pointer U8)) (pointer-field-read vector-struct data)
              idx (: I32) 0]
          ;; Iterate through blocks in this region
          (while (< idx count)
            (let [elem-offset (: I64) (* (cast I64 idx) 8)
                  elem-ptr-loc (: (Pointer U8)) (cast (Pointer U8) (+ (cast I64 data) elem-offset))
                  elem-ptr-ptr (: (Pointer (Pointer types/Value))) (cast (Pointer (Pointer types/Value)) elem-ptr-loc)
                  block-form (: (Pointer types/Value)) (dereference elem-ptr-ptr)
                  mlir-block (: MlirBlock) (build-mlir-block builder block-form tracker)]
              (mlirRegionAppendOwnedBlock region mlir-block)
              (set! idx (+ idx 1))))
          region)
        region))))

;; Build an MLIR block from a block form
(def build-mlir-block (: (-> [(Pointer MLIRBuilderContext) (Pointer types/Value) (Pointer ValueTracker)] MlirBlock))
  (fn [builder block-form tracker]
    (let [block-node (: (Pointer ast/BlockNode)) (ast/parse-block block-form)]
      (if (!= (cast I64 block-node) 0)
        ;; Parse block args - for now, just create empty block
        ;; TODO: Handle block arguments properly
        (let [loc (: MlirLocation) (pointer-field-read builder loc)
              block (: MlirBlock) (mlirBlockCreate 0 (cast (Pointer MlirType) 0) (cast (Pointer MlirLocation) 0))
              operations (: (Pointer types/Value)) (pointer-field-read block-node operations)
              ops-tag (: types/ValueTag) (pointer-field-read operations tag)]
          ;; operations should be a vector
          (if (= ops-tag types/ValueTag/Vector)
            (let [vec-ptr (: (Pointer U8)) (pointer-field-read operations vec_val)
                  vector-struct (: (Pointer types/Vector)) (cast (Pointer types/Vector) vec-ptr)
                  count (: I32) (pointer-field-read vector-struct count)
                  data (: (Pointer U8)) (pointer-field-read vector-struct data)
                  idx (: I32) 0]
              ;; Iterate through operations in this block
              (while (< idx count)
                (let [elem-offset (: I64) (* (cast I64 idx) 8)
                      elem-ptr-loc (: (Pointer U8)) (cast (Pointer U8) (+ (cast I64 data) elem-offset))
                      elem-ptr-ptr (: (Pointer (Pointer types/Value))) (cast (Pointer (Pointer types/Value)) elem-ptr-loc)
                      op-form (: (Pointer types/Value)) (dereference elem-ptr-ptr)
                      op-node (: (Pointer ast/OpNode)) (ast/parse-op op-form)]
                  (if (!= (cast I64 op-node) 0)
                    (let [mlir-op (: MlirOperation) (build-mlir-operation builder op-node tracker block)]
                      (set! idx (+ idx 1)))
                    (set! idx (+ idx 1)))))
              block)
            block))
        (mlirBlockCreate 0 (cast (Pointer MlirType) 0) (cast (Pointer MlirLocation) 0))))))

;; Helper: Process result types vector and add to operation state
(def add-result-types-to-state (: (-> [(Pointer MLIRBuilderContext) (Pointer MlirOperationState) (Pointer types/Value)] I32))
  (fn [builder state-ptr result-types-val]
    (let [tag (: types/ValueTag) (pointer-field-read result-types-val tag)]
      (if (= tag types/ValueTag/Vector)
        (let [vec-ptr (: (Pointer U8)) (pointer-field-read result-types-val vec_val)
              vector-struct (: (Pointer types/Vector)) (cast (Pointer types/Vector) vec-ptr)
              count (: I32) (pointer-field-read vector-struct count)]
          (if (> count 0)
            (let [data (: (Pointer U8)) (pointer-field-read vector-struct data)
                  result-types-array (: (Array MlirType 8)) (array MlirType 8)
                  idx (: I32) 0]
              ;; Iterate through result types and parse them
              (while (< idx count)
                (let [elem-offset (: I64) (* (cast I64 idx) 8)
                      elem-ptr-loc (: (Pointer U8)) (cast (Pointer U8) (+ (cast I64 data) elem-offset))
                      elem-ptr-ptr (: (Pointer (Pointer types/Value))) (cast (Pointer (Pointer types/Value)) elem-ptr-loc)
                      elem (: (Pointer types/Value)) (dereference elem-ptr-ptr)
                      elem-tag (: types/ValueTag) (pointer-field-read elem tag)]
                  (if (= elem-tag types/ValueTag/String)
                    (let [type-str (: (Pointer U8)) (pointer-field-read elem str_val)
                          mlir-type (: MlirType) (parse-type-string builder type-str)]
                      (array-set! result-types-array idx mlir-type)
                      (set! idx (+ idx 1)))
                    (set! idx (+ idx 1)))))
              ;; Add result types to state
              (mlirOperationStateAddResults state-ptr (cast I64 count) (array-ptr result-types-array 0))
              count)
            0))
        0))))

;; Helper: Process operands vector and add to operation state
(def add-operands-to-state (: (-> [(Pointer MLIRBuilderContext) (Pointer MlirOperationState) (Pointer types/Value) (Pointer ValueTracker)] I32))
  (fn [builder state-ptr operands-val tracker]
    (let [tag (: types/ValueTag) (pointer-field-read operands-val tag)]
      (if (= tag types/ValueTag/Vector)
        (let [vec-ptr (: (Pointer U8)) (pointer-field-read operands-val vec_val)
              vector-struct (: (Pointer types/Vector)) (cast (Pointer types/Vector) vec-ptr)
              count (: I32) (pointer-field-read vector-struct count)]
          (if (> count 0)
            (let [data (: (Pointer U8)) (pointer-field-read vector-struct data)
                  operands-array (: (Array MlirValue 8)) (array MlirValue 8)
                  idx (: I32) 0]
              ;; Iterate through operands (which are SSA value indices as strings)
              (while (< idx count)
                (let [elem-offset (: I64) (* (cast I64 idx) 8)
                      elem-ptr-loc (: (Pointer U8)) (cast (Pointer U8) (+ (cast I64 data) elem-offset))
                      elem-ptr-ptr (: (Pointer (Pointer types/Value))) (cast (Pointer (Pointer types/Value)) elem-ptr-loc)
                      elem (: (Pointer types/Value)) (dereference elem-ptr-ptr)
                      elem-tag (: types/ValueTag) (pointer-field-read elem tag)]
                  (if (= elem-tag types/ValueTag/String)
                    (let [operand-str (: (Pointer U8)) (pointer-field-read elem str_val)
                          operand-idx (: I32) (atoi operand-str)
                          operand-val (: MlirValue) (value-tracker-lookup tracker operand-idx)]
                      (array-set! operands-array idx operand-val)
                      (set! idx (+ idx 1)))
                    (set! idx (+ idx 1)))))
              ;; Add operands to state
              (mlirOperationStateAddOperands state-ptr (cast I64 count) (array-ptr operands-array 0))
              count)
            0))
        0))))

;; Helper: Process attributes map and add to operation state
(def add-attributes-to-state (: (-> [(Pointer MLIRBuilderContext) (Pointer MlirOperationState) (Pointer types/Value)] I32))
  (fn [builder state-ptr attributes-val]
    (let [tag (: types/ValueTag) (pointer-field-read attributes-val tag)]
      (if (= tag types/ValueTag/Map)
        ;; Map data is stored in vec_val as a Vector of [key, value, key, value, ...]
        (let [vec-ptr (: (Pointer U8)) (pointer-field-read attributes-val vec_val)
              vector-struct (: (Pointer types/Vector)) (cast (Pointer types/Vector) vec-ptr)
              vec-count (: I32) (pointer-field-read vector-struct count)
              attr-count (: I32) (cast I32 (/ (cast F64 vec-count) 2.0))  ; Each attribute is 2 elements
              (if (> attr-count 0)
                (let [data (: (Pointer U8)) (pointer-field-read vector-struct data)
                      attrs-array (: (Array MlirNamedAttribute 16)) (array MlirNamedAttribute 16)
                      idx (: I32) 0]
                  ;; Iterate through map entries (key-value pairs)
                  (while (< idx attr-count)
                    (let [key-idx (: I32) (* idx 2)
                          val-idx (: I32) (+ (* idx 2) 1)

                          key-offset (: I64) (* (cast I64 key-idx) 8)
                          key-ptr-loc (: (Pointer U8)) (cast (Pointer U8) (+ (cast I64 data) key-offset))
                          key-ptr-ptr (: (Pointer (Pointer types/Value))) (cast (Pointer (Pointer types/Value)) key-ptr-loc)
                          key-val (: (Pointer types/Value)) (dereference key-ptr-ptr)

                          val-offset (: I64) (* (cast I64 val-idx) 8)
                          val-ptr-loc (: (Pointer U8)) (cast (Pointer U8) (+ (cast I64 data) val-offset))
                          val-ptr-ptr (: (Pointer (Pointer types/Value))) (cast (Pointer (Pointer types/Value)) val-ptr-loc)
                          val-val (: (Pointer types/Value)) (dereference val-ptr-ptr)]

                      (let [key-str (: (Pointer U8)) (pointer-field-read key-val str_val)
                            named-attr (: MlirNamedAttribute) (create-named-attribute-from-value builder key-str val-val)]
                        (array-set! attrs-array idx named-attr)
                        (set! idx (+ idx 1)))))
                  ;; Add attributes to state
                  (mlirOperationStateAddAttributes state-ptr (cast I64 attr-count) (array-ptr attrs-array 0))
                  attr-count)
                0)])
        0))))

;; Helper: Process regions vector and add to operation state
(def add-regions-to-state (: (-> [(Pointer MLIRBuilderContext) (Pointer MlirOperationState) (Pointer types/Value) (Pointer ValueTracker)] I32))
  (fn [builder state-ptr regions-val tracker]
    (let [tag (: types/ValueTag) (pointer-field-read regions-val tag)]
      (if (= tag types/ValueTag/Vector)
        (let [vec-ptr (: (Pointer U8)) (pointer-field-read regions-val vec_val)
              vector-struct (: (Pointer types/Vector)) (cast (Pointer types/Vector) vec-ptr)
              count (: I32) (pointer-field-read vector-struct count)]
          (if (> count 0)
            (let [data (: (Pointer U8)) (pointer-field-read vector-struct data)
                  regions-array (: (Array MlirRegion 4)) (array MlirRegion 4)
                  idx (: I32) 0]
              ;; Iterate through regions (each is a vector of blocks)
              (while (< idx count)
                (let [elem-offset (: I64) (* (cast I64 idx) 8)
                      elem-ptr-loc (: (Pointer U8)) (cast (Pointer U8) (+ (cast I64 data) elem-offset))
                      elem-ptr-ptr (: (Pointer (Pointer types/Value))) (cast (Pointer (Pointer types/Value)) elem-ptr-loc)
                      region-vec (: (Pointer types/Value)) (dereference elem-ptr-ptr)
                      mlir-region (: MlirRegion) (build-mlir-region builder region-vec tracker)]
                  (array-set! regions-array idx mlir-region)
                  (set! idx (+ idx 1))))
              ;; Add regions to state
              (mlirOperationStateAddOwnedRegions state-ptr (cast I64 count) (array-ptr regions-array 0))
              count)
            0))
        0))))

;; Build an MLIR operation from an OpNode
(def build-mlir-operation (: (-> [(Pointer MLIRBuilderContext) (Pointer ast/OpNode) (Pointer ValueTracker) MlirBlock] MlirOperation))
  (fn [builder op-node tracker parent-block]
    (let [ctx (: MlirContext) (pointer-field-read builder ctx)
          loc (: MlirLocation) (pointer-field-read builder loc)
          name-str (: (Pointer U8)) (pointer-field-read op-node name)
          name-ref (: MlirStringRef) (mlirStringRefCreateFromCString name-str)
          state (: MlirOperationState) (mlirOperationStateGet name-ref loc)
          state-ptr (: (Pointer MlirOperationState)) (allocate MlirOperationState state)]

      (printf (c-str "Building operation: %s\n") name-str)

      ;; Add result types
      (let [result-types (: (Pointer types/Value)) (pointer-field-read op-node result-types)
            _ (: I32) (add-result-types-to-state builder state-ptr result-types)]

        ;; Add operands
        (let [operands (: (Pointer types/Value)) (pointer-field-read op-node operands)
              _ (: I32) (add-operands-to-state builder state-ptr operands tracker)]

          ;; Add attributes
          (let [attributes (: (Pointer types/Value)) (pointer-field-read op-node attributes)
                _ (: I32) (add-attributes-to-state builder state-ptr attributes)]

            ;; Add regions
            (let [regions (: (Pointer types/Value)) (pointer-field-read op-node regions)
                  _ (: I32) (add-regions-to-state builder state-ptr regions tracker)]

              ;; Create the operation
              (let [op (: MlirOperation) (mlirOperationCreate state-ptr)]
                (mlirBlockAppendOwnedOperation parent-block op)

                ;; Register result values in tracker
                (let [result-types-val (: (Pointer types/Value)) (pointer-field-read op-node result-types)
                      result-tag (: types/ValueTag) (pointer-field-read result-types-val tag)]
                  (if (= result-tag types/ValueTag/Vector)
                    (let [vec-ptr (: (Pointer U8)) (pointer-field-read result-types-val vec_val)
                          vector-struct (: (Pointer types/Vector)) (cast (Pointer types/Vector) vec-ptr)
                          result-count (: I32) (pointer-field-read vector-struct count)
                          idx (: I32) 0]
                      (while (< idx result-count)
                        (let [result-val (: MlirValue) (mlirOperationGetResult op (cast I64 idx))
                              _ (: I32) (value-tracker-register tracker result-val)]
                          (set! idx (+ idx 1))))
                      op)
                    op))))))))))

;; Compile a top-level OpNode to MLIR
(def compile-op-to-mlir (: (-> [(Pointer MLIRBuilderContext) (Pointer types/Value)] I32))
  (fn [builder op-form]
    (printf (c-str "=== Starting MLIR Compilation ===\n"))

    ;; Parse the top-level op
    (let [op-node (: (Pointer ast/OpNode)) (ast/parse-op op-form)]
      (if (!= (cast I64 op-node) 0)
        (let [mod (: MlirModule) (pointer-field-read builder mod)
              mod-op (: MlirOperation) (mlirModuleGetOperation mod)
              mod-region (: MlirRegion) (mlirOperationGetRegion mod-op 0)
              mod-block (: MlirBlock) (mlirRegionGetFirstBlock mod-region)
              tracker (: (Pointer ValueTracker)) (value-tracker-create)
              _ (: MlirOperation) (build-mlir-operation builder op-node tracker mod-block)]

          (printf (c-str "=== Dumping MLIR Module ===\n"))
          (mlirOperationDump mod-op)
          (printf (c-str "\n"))
          0)
        (do
          (printf (c-str "ERROR: Failed to parse op\n"))
          1)))))

;; Require additional modules for file parsing
(require [tokenizer :as tokenizer])
(require [parser :as parser])

;; C library functions for file I/O
(declare-fn fopen [filename (Pointer U8) mode (Pointer U8)] -> (Pointer U8))
(declare-fn fclose [file (Pointer U8)] -> I32)
(declare-fn fseek [file (Pointer U8) offset I32 whence I32] -> I32)
(declare-fn ftell [file (Pointer U8)] -> I32)
(declare-fn fread [ptr (Pointer U8) size I32 count I32 file (Pointer U8)] -> I32)
(declare-fn rewind [file (Pointer U8)] -> Nil)

;; Read entire file into a string
(def read-file (: (-> [(Pointer U8)] (Pointer U8)))
  (fn [filename]
    (let [file (: (Pointer U8)) (fopen filename (c-str "r"))]
      (if (= (cast I64 file) 0)
        (let [_ (: I32) (printf (c-str "Error: Could not open file %s\n") filename)]
          (cast (Pointer U8) 0))
        (let [;; Seek to end to get file size
              _ (: I32) (fseek file 0 2)  ; SEEK_END = 2
              size (: I32) (ftell file)
              _ (: Nil) (rewind file)

              ;; Allocate buffer (size + 1 for null terminator)
              buffer (: (Pointer U8)) (malloc (+ size 1))

              ;; Read file contents
              read-count (: I32) (fread buffer 1 size file)
              _ (: I32) (fclose file)]

          ;; Null terminate the buffer
          (let [null-pos (: I64) (+ (cast I64 buffer) (cast I64 size))
                null-ptr (: (Pointer U8)) (cast (Pointer U8) null-pos)]
            (pointer-write! null-ptr (cast U8 0))
            buffer))))))

;; Tokenize file content
(def tokenize-file (: (-> [(Pointer U8)] (Pointer types/Token)))
  (fn [content]
    (let [tok (: (Pointer tokenizer/Tokenizer)) (tokenizer/make-tokenizer content)
          max-tokens (: I32) 1000
          token-size (: I32) 24
          tokens (: (Pointer types/Token)) (cast (Pointer types/Token) (malloc (* max-tokens token-size)))
          count (: I32) 0]

      (while (< count max-tokens)
        (let [token (: types/Token) (tokenizer/next-token tok)
              token-type (: types/TokenType) (. token type)
              token-offset (: I64) (* (cast I64 count) (cast I64 token-size))
              token-ptr (: (Pointer types/Token)) (cast (Pointer types/Token) (+ (cast I64 tokens) token-offset))]

          (pointer-field-write! token-ptr type (. token type))
          (pointer-field-write! token-ptr text (. token text))
          (pointer-field-write! token-ptr length (. token length))
          (set! count (+ count 1))

          (if (= token-type types/TokenType/EOF)
            (set! count max-tokens)
            (set! count count))))

      tokens)))

;; Count tokens until EOF
(def count-tokens (: (-> [(Pointer types/Token)] I32))
  (fn [tokens]
    (let [token-count (: I32) 0
          found-eof (: I32) 0]
      (while (and (< token-count 1000) (= found-eof 0))
        (let [token-offset (: I64) (* (cast I64 token-count) 24)
              token-ptr (: (Pointer types/Token)) (cast (Pointer types/Token) (+ (cast I64 tokens) token-offset))
              token-type (: types/TokenType) (pointer-field-read token-ptr type)]
          (set! token-count (+ token-count 1))
          (if (= token-type types/TokenType/EOF)
            (set! found-eof 1)
            (set! found-eof 0))))
      token-count)))

;; Compile a file to MLIR
(def compile-file (: (-> [(Pointer MLIRBuilderContext) (Pointer U8)] I32))
  (fn [builder filename]
    (printf (c-str "=== Compiling: %s ===\n") filename)

    (let [content (: (Pointer U8)) (read-file filename)]
      (if (= (cast I64 content) 0)
        1
        (let [tokens (: (Pointer types/Token)) (tokenize-file content)
              token-count (: I32) (count-tokens tokens)
              _ (: I32) (printf (c-str "Found %d tokens\n") token-count)
              p (: (Pointer parser/Parser)) (parser/make-parser tokens token-count)]

          ;; Parse the single top-level op form
          (let [result (: (Pointer types/Value)) (parser/parse-value p)]
            (compile-op-to-mlir builder result)))))))

;; Main test
(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "=== MLIR Builder - File Compilation Test ===\n\n"))
    (let [builder (: (Pointer MLIRBuilderContext)) (mlir-builder-init)]
      (printf (c-str "Builder initialized successfully\n\n"))

      ;; Compile the fib test file
      (let [result (: I32) (compile-file builder (c-str "tests/fib.lisp"))]
        (mlir-builder-destroy builder)
        (printf (c-str "\nDone!\n"))
        result))))

(main-fn)