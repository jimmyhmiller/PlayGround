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
(declare-fn mlirFlatSymbolRefAttrGet [ctx MlirContext symbol MlirStringRef] -> MlirAttribute)
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

;; Linked list entry for SSA value tracking (name -> MlirValue)
(def ValueMapEntry (: Type)
  (Struct
    [name (Pointer U8)]
    [value MlirValue]
    [next (Pointer ValueMapEntry)]))

;; Tracker holding the head of the linked list
(def ValueTracker (: Type)
  (Struct
    [head (Pointer ValueMapEntry)]))

;; Allocate and initialize an empty tracker
(def value-tracker-create (: (-> [] (Pointer ValueTracker)))
  (fn []
    (let [tracker-bytes (: (Pointer U8)) (malloc 8)
          tracker (: (Pointer ValueTracker)) (cast (Pointer ValueTracker) tracker-bytes)]
      (pointer-field-write! tracker head (cast (Pointer ValueMapEntry) 0))
      tracker)))

;; Internal helper to find an entry by SSA name
(def value-tracker-find-entry (: (-> [(Pointer ValueMapEntry) (Pointer U8)] (Pointer ValueMapEntry)))
  (fn [entry name]
    (if (= (cast I64 entry) 0)
      (cast (Pointer ValueMapEntry) 0)
      (let [entry-name (: (Pointer U8)) (pointer-field-read entry name)
            cmp (: I32) (strcmp entry-name name)]
        (if (= cmp 0)
          entry
          (value-tracker-find-entry (pointer-field-read entry next) name))))))

;; Register (or update) an SSA value by name
(def value-tracker-register (: (-> [(Pointer ValueTracker) (Pointer U8) MlirValue] I32))
  (fn [tracker name value]
    (let [head (: (Pointer ValueMapEntry)) (pointer-field-read tracker head)
          existing (: (Pointer ValueMapEntry)) (value-tracker-find-entry head name)]
      (if (!= (cast I64 existing) 0)
        (do
          (pointer-field-write! existing value value)
          0)
        (let [entry-bytes (: (Pointer U8)) (malloc 32)
              new-entry (: (Pointer ValueMapEntry)) (cast (Pointer ValueMapEntry) entry-bytes)]
          (pointer-field-write! new-entry name name)
          (pointer-field-write! new-entry value value)
          (pointer-field-write! new-entry next head)
          (pointer-field-write! tracker head new-entry)
          0)))))

;; Helper used when lookups fail; prints message and exits while satisfying type checker
(def value-tracker-missing (: (-> [(Pointer U8)] MlirValue))
  (fn [name]
    (let [_ (: I32) (printf (c-str "ERROR: Unknown SSA value '%s'\n") name)]
      (die (c-str "SSA lookup failed"))
      (let [tmp (: (Pointer MlirValue)) (cast (Pointer MlirValue) (malloc 16))]
        (dereference tmp)))))

;; Look up an SSA value by name, exiting if the name is unknown
(def value-tracker-lookup (: (-> [(Pointer ValueTracker) (Pointer U8)] MlirValue))
  (fn [tracker name]
    (let [head (: (Pointer ValueMapEntry)) (pointer-field-read tracker head)
          entry (: (Pointer ValueMapEntry)) (value-tracker-find-entry head name)]
      (if (= (cast I64 entry) 0)
        (value-tracker-missing name)
        (pointer-field-read entry value)))))

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


;; Parse type string to MlirType (e.g., "i32" -> integer type)
;; NOTE: Workaround for compiler bug with nested if in return position
;; Uses separate helper function to avoid nesting
(def parse-type-string-helper (: (-> [(Pointer MLIRBuilderContext) (Pointer U8)] MlirType))
  (fn [builder type-str]
    (let [cmp-i64 (: I32) (strcmp type-str (c-str "i64"))]
      (if (== cmp-i64 0)
        (pointer-field-read builder i64Type)
        (let [cmp-i1 (: I32) (strcmp type-str (c-str "i1"))]
          (if (== cmp-i1 0)
            (let [ctx (: MlirContext) (pointer-field-read builder ctx)]
              (mlirIntegerTypeGet ctx 1))
            (let [_ (: I32) (printf (c-str "Unknown type: %s, defaulting to i32\n") type-str)]
              (pointer-field-read builder i32Type))))))))

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
    ;; Use car/cdr to access list elements
    (let [first-elem (: (Pointer types/Value)) (types/car value-list)]
      (let [rest (: (Pointer types/Value)) (types/cdr value-list)]
        (let [rest-tag (: types/ValueTag) (pointer-field-read rest tag)]
          (if (= rest-tag types/ValueTag/List)
            (let [second-elem (: (Pointer types/Value)) (types/car rest)
                  ;; First element should be a Number - read num_val directly
                  int-val (: I64) (pointer-field-read first-elem num_val)
                  ;; Second element should be a Symbol - read str_val
                  type-str (: (Pointer U8)) (pointer-field-read second-elem str_val)
                  mlir-type (: MlirType) (parse-type-string builder type-str)]
              (mlirIntegerAttrGet mlir-type int-val))
            (let [ctx (: MlirContext) (pointer-field-read builder ctx)]
              (printf (c-str "    ERROR: Invalid integer value format, rest-tag=%d\n") (cast I32 rest-tag))
              (mlirStringAttrGet ctx (mlirStringRefCreateFromCString (c-str "ERROR"))))))))))

;; Parse function type attribute like (-> [i32] [i32])
(def parse-function-type-attr (: (-> [(Pointer MLIRBuilderContext) (Pointer types/Value)] MlirAttribute))
  (fn [builder value-list]
    (let [ctx (: MlirContext) (pointer-field-read builder ctx)
          input-types (: (Array MlirType 8)) (array MlirType 8)
          result-types (: (Array MlirType 8)) (array MlirType 8)
          input-count (: I32) 0
          result-count (: I32) 0
          args-list (: (Pointer types/Value)) (types/cdr value-list)
          inputs-val (: (Pointer types/Value)) (if (= (cast I64 args-list) 0)
                                                (cast (Pointer types/Value) 0)
                                                (types/car args-list))
          outputs-list (: (Pointer types/Value)) (if (= (cast I64 args-list) 0)
                                                  (cast (Pointer types/Value) 0)
                                                  (types/cdr args-list))
          outputs-val (: (Pointer types/Value)) (if (= (cast I64 outputs-list) 0)
                                                 (cast (Pointer types/Value) 0)
                                                 (types/car outputs-list))]

      ;; Parse input types vector
      (if (!= (cast I64 inputs-val) 0)
        (let [inputs-tag (: types/ValueTag) (pointer-field-read inputs-val tag)]
          (if (= inputs-tag types/ValueTag/Vector)
            (let [vec-struct (: (Pointer types/Vector)) (cast (Pointer types/Vector) (pointer-field-read inputs-val vec_val))
                  vec-count (: I32) (pointer-field-read vec-struct count)
                  limit (: I32) (if (> vec-count 8) 8 vec-count)
                  idx (: I32) 0]
              (set! input-count limit)
              (while (< idx limit)
                (let [elem (: (Pointer types/Value)) (vector-element inputs-val idx)
                      elem-tag (: types/ValueTag) (pointer-field-read elem tag)
                      type-str (: (Pointer U8))
                        (if (= elem-tag types/ValueTag/Symbol)
                          (pointer-field-read elem str_val)
                          (if (= elem-tag types/ValueTag/String)
                            (pointer-field-read elem str_val)
                            (c-str "i32")))
                      mlir-type (: MlirType) (parse-type-string builder type-str)]
                  (array-set! input-types idx mlir-type)
                  (set! idx (+ idx 1))
                  0))
              0)
            0))
        0)

      ;; Parse result types vector
      (if (!= (cast I64 outputs-val) 0)
        (let [outputs-tag (: types/ValueTag) (pointer-field-read outputs-val tag)]
          (if (= outputs-tag types/ValueTag/Vector)
            (let [vec-struct (: (Pointer types/Vector)) (cast (Pointer types/Vector) (pointer-field-read outputs-val vec_val))
                  vec-count (: I32) (pointer-field-read vec-struct count)
                  limit (: I32) (if (> vec-count 8) 8 vec-count)
                  idx (: I32) 0]
              (set! result-count limit)
              (while (< idx limit)
                (let [elem (: (Pointer types/Value)) (vector-element outputs-val idx)
                      elem-tag (: types/ValueTag) (pointer-field-read elem tag)
                      type-str (: (Pointer U8))
                        (if (= elem-tag types/ValueTag/Symbol)
                          (pointer-field-read elem str_val)
                          (if (= elem-tag types/ValueTag/String)
                            (pointer-field-read elem str_val)
                            (c-str "i32")))
                      mlir-type (: MlirType) (parse-type-string builder type-str)]
                  (array-set! result-types idx mlir-type)
                  (set! idx (+ idx 1))
                  0))
              0)
            0))
        0)

      (let [input-ptr (: (Pointer MlirType)) (if (> input-count 0)
                                               (array-ptr input-types 0)
                                               (cast (Pointer MlirType) 0))
            result-ptr (: (Pointer MlirType)) (if (> result-count 0)
                                                (array-ptr result-types 0)
                                                (cast (Pointer MlirType) 0))
            fn-type (: MlirType) (mlirFunctionTypeGet ctx (cast I64 input-count) input-ptr (cast I64 result-count) result-ptr)]
        (mlirTypeAttrGet fn-type)))))

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
          (if (== (strcmp key (c-str "callee")) 0)
            (let [first-char (: U8) (dereference value-str)
                  symbol-name (: (Pointer U8)) (if (= first-char (cast U8 64))
                                                 (cast (Pointer U8) (+ (cast I64 value-str) 1))
                                                 value-str)
                  sym-ref (: MlirStringRef) (mlirStringRefCreateFromCString symbol-name)]
              (mlirFlatSymbolRefAttrGet ctx sym-ref))
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
              (mlirStringAttrGet ctx (mlirStringRefCreateFromCString value-str)))))

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

;; Helper to fetch an element from a vector Value
(def vector-element (: (-> [(Pointer types/Value) I32] (Pointer types/Value)))
  (fn [vec-val idx]
    (let [vec-ptr (: (Pointer U8)) (pointer-field-read vec-val vec_val)
          vector-struct (: (Pointer types/Vector)) (cast (Pointer types/Vector) vec-ptr)
          data (: (Pointer U8)) (pointer-field-read vector-struct data)
          elem-offset (: I64) (* (cast I64 idx) 8)
          elem-ptr-loc (: (Pointer U8)) (cast (Pointer U8) (+ (cast I64 data) elem-offset))
          elem-ptr-ptr (: (Pointer (Pointer types/Value))) (cast (Pointer (Pointer types/Value)) elem-ptr-loc)]
      (dereference elem-ptr-ptr))))

;; Helper: create block with arguments and register them
(def create-block-with-args (: (-> [(Pointer MLIRBuilderContext) (Pointer ValueTracker) MlirLocation (Pointer types/Value)] MlirBlock))
  (fn [builder tracker loc args-val]
    (let [tag (: types/ValueTag) (pointer-field-read args-val tag)]
      (if (= tag types/ValueTag/Vector)
        (let [vec-ptr (: (Pointer U8)) (pointer-field-read args-val vec_val)
              vector-struct (: (Pointer types/Vector)) (cast (Pointer types/Vector) vec-ptr)
              count (: I32) (pointer-field-read vector-struct count)]
          (if (> count 0)
            (let [arg-types (: (Array MlirType 16)) (array MlirType 16)
                  arg-locs (: (Array MlirLocation 16)) (array MlirLocation 16)
                  idx (: I32) 0]
              (while (< idx count)
                (let [pair (: (Pointer types/Value)) (vector-element args-val idx)
                      pair-tag (: types/ValueTag) (pointer-field-read pair tag)
                      mlir-type (: MlirType)
                      (if (= pair-tag types/ValueTag/Vector)
                        (let [type-elem (: (Pointer types/Value)) (vector-element pair 1)
                              type-str (: (Pointer U8)) (pointer-field-read type-elem str_val)]
                          (parse-type-string builder type-str))
                        (pointer-field-read builder i32Type))]
                  (array-set! arg-types idx mlir-type)
                  (array-set! arg-locs idx loc)
                  (set! idx (+ idx 1))))
              (let [block (: MlirBlock) (mlirBlockCreate (cast I64 count) (array-ptr arg-types 0) (array-ptr arg-locs 0))
                    reg-idx (: I32) 0]
                (while (< reg-idx count)
                  (let [pair (: (Pointer types/Value)) (vector-element args-val reg-idx)
                        pair-tag (: types/ValueTag) (pointer-field-read pair tag)]
                    (if (= pair-tag types/ValueTag/Vector)
                      (let [name-elem (: (Pointer types/Value)) (vector-element pair 0)
                            name-str (: (Pointer U8)) (pointer-field-read name-elem str_val)
                            arg-val (: MlirValue) (mlirBlockGetArgument block (cast I64 reg-idx))]
                        (value-tracker-register tracker name-str arg-val))
                      0)
                    (set! reg-idx (+ reg-idx 1))))
                block))
            (mlirBlockCreate 0 (cast (Pointer MlirType) 0) (cast (Pointer MlirLocation) 0))))
        (mlirBlockCreate 0 (cast (Pointer MlirType) 0) (cast (Pointer MlirLocation) 0))))))

;; Build an MLIR block from a block form
(def build-mlir-block (: (-> [(Pointer MLIRBuilderContext) (Pointer types/Value) (Pointer ValueTracker)] MlirBlock))
  (fn [builder block-form tracker]
    (let [block-node (: (Pointer ast/BlockNode)) (ast/parse-block block-form)]
      (if (!= (cast I64 block-node) 0)
        (let [loc (: MlirLocation) (pointer-field-read builder loc)
              args-val (: (Pointer types/Value)) (pointer-field-read block-node args)
              block (: MlirBlock) (create-block-with-args builder tracker loc args-val)
              operations (: (Pointer types/Value)) (pointer-field-read block-node operations)
              ops-tag (: types/ValueTag) (pointer-field-read operations tag)]
          (if (= ops-tag types/ValueTag/Vector)
            (let [vec-ptr (: (Pointer U8)) (pointer-field-read operations vec_val)
                  vector-struct (: (Pointer types/Vector)) (cast (Pointer types/Vector) vec-ptr)
                  count (: I32) (pointer-field-read vector-struct count)
                  idx (: I32) 0]
              (while (< idx count)
                (let [op-form (: (Pointer types/Value)) (vector-element operations idx)
                      op-node (: (Pointer ast/OpNode)) (ast/parse-op op-form)]
                  (if (!= (cast I64 op-node) 0)
                    (let [_ (: MlirOperation) (build-mlir-operation builder op-node tracker block)]
                      0)
                    0)
                  (set! idx (+ idx 1))
                  0))
              0)
            0)
          block)
        (mlirBlockCreate 0 (cast (Pointer MlirType) 0) (cast (Pointer MlirLocation) 0))))))

;; Extract the MLIR type string from a result entry
(def result-type-string (: (-> [(Pointer types/Value)] (Pointer U8)))
  (fn [elem]
    (let [elem-tag (: types/ValueTag) (pointer-field-read elem tag)]
      (if (= elem-tag types/ValueTag/Vector)
        (let [type-elem (: (Pointer types/Value)) (vector-element elem 1)]
          (pointer-field-read type-elem str_val))
        (if (= elem-tag types/ValueTag/String)
          (pointer-field-read elem str_val)
          (if (= elem-tag types/ValueTag/Symbol)
            (pointer-field-read elem str_val)
            (cast (Pointer U8) 0)))))))

;; Helper: Process result types vector and add to operation state
(def add-result-types-to-state (: (-> [(Pointer MLIRBuilderContext) (Pointer MlirOperationState) (Pointer types/Value)] I32))
  (fn [builder state-ptr result-types-val]
    (let [tag (: types/ValueTag) (pointer-field-read result-types-val tag)]
      (if (= tag types/ValueTag/Vector)
        (let [vec-ptr (: (Pointer U8)) (pointer-field-read result-types-val vec_val)
              vector-struct (: (Pointer types/Vector)) (cast (Pointer types/Vector) vec-ptr)
              raw-count (: I32) (pointer-field-read vector-struct count)]
          (if (> raw-count 0)
            (let [first-elem (: (Pointer types/Value)) (vector-element result-types-val 0)
                  first-tag (: types/ValueTag) (pointer-field-read first-elem tag)
                  result-count (: I32) (if (= first-tag types/ValueTag/Vector)
                                         raw-count
                                         (cast I32 (/ (cast F64 raw-count) 2.0)))]
              (if (> result-count 0)
                (let [result-types-array (: (Array MlirType 8)) (array MlirType 8)
                      idx (: I32) 0]
                  (while (< idx result-count)
                    (let [type-val (: (Pointer types/Value))
                                (if (= first-tag types/ValueTag/Vector)
                                  (let [elem (: (Pointer types/Value)) (vector-element result-types-val idx)]
                                    (vector-element elem 1))
                                  (let [type-idx (: I32) (+ (* idx 2) 1)]
                                    (vector-element result-types-val type-idx)))
                          type-str (: (Pointer U8)) (result-type-string type-val)
                          mlir-type (: MlirType) (if (!= (cast I64 type-str) 0)
                                                   (parse-type-string builder type-str)
                                                   (pointer-field-read builder i32Type))]
                      (array-set! result-types-array idx mlir-type)
                      (set! idx (+ idx 1))))
                  (mlirOperationStateAddResults state-ptr (cast I64 result-count) (array-ptr result-types-array 0))
                  result-count)
                0))
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
            (let [operands-array (: (Array MlirValue 8)) (array MlirValue 8)
                  idx (: I32) 0]
              (while (< idx count)
                (let [elem (: (Pointer types/Value)) (vector-element operands-val idx)
                      elem-tag (: types/ValueTag) (pointer-field-read elem tag)
                      operand-name (: (Pointer U8))
                        (if (= elem-tag types/ValueTag/String)
                          (pointer-field-read elem str_val)
                          (if (= elem-tag types/ValueTag/Symbol)
                            (pointer-field-read elem str_val)
                            (cast (Pointer U8) 0)))]
                  (if (!= (cast I64 operand-name) 0)
                    (let [operand-val (: MlirValue) (value-tracker-lookup tracker operand-name)]
                      (array-set! operands-array idx operand-val)
                      0)
                    0)
                  (set! idx (+ idx 1))))
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
              attr-count (: I32) (cast I32 (/ (cast F64 vec-count) 2.0))]  ; Each attribute is 2 elements
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
            0))
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
                  idx (: I32) 0
                  out-count (: I32) 0]
              ;; Iterate through regions (each is a vector of blocks)
              (while (< idx count)
                (let [elem-offset (: I64) (* (cast I64 idx) 8)
                      elem-ptr-loc (: (Pointer U8)) (cast (Pointer U8) (+ (cast I64 data) elem-offset))
                      elem-ptr-ptr (: (Pointer (Pointer types/Value))) (cast (Pointer (Pointer types/Value)) elem-ptr-loc)
                      region-vec (: (Pointer types/Value)) (dereference elem-ptr-ptr)
                      region-tag (: types/ValueTag) (pointer-field-read region-vec tag)]
                  (if (= region-tag types/ValueTag/Vector)
                    (let [mlir-region (: MlirRegion) (build-mlir-region builder region-vec tracker)]
                      (array-set! regions-array out-count mlir-region)
                      (set! out-count (+ out-count 1))
                      0)
                    0)
                  (set! idx (+ idx 1))))
              ;; Add regions to state (only the ones we actually created)
              (if (> out-count 0)
                (let [_ (: Nil) (mlirOperationStateAddOwnedRegions state-ptr (cast I64 out-count) (array-ptr regions-array 0))]
                  0)
                0)
              out-count)
            0))
        0))))

;; Lower the current module to the LLVM dialect using a fixed pass pipeline
(def run-lowering-passes (: (-> [(Pointer MLIRBuilderContext)] I32))
  (fn [builder]
    (let [ctx (: MlirContext) (pointer-field-read builder ctx)
          mod (: MlirModule) (pointer-field-read builder mod)
          first-pipeline (: (Pointer U8))
            (c-str "builtin.module(func.func(convert-scf-to-cf))")
          second-pipeline (: (Pointer U8))
            (c-str "builtin.module(func.func(convert-arith-to-llvm),convert-cf-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)")]

      ;; First run: convert structured control flow to cf
      (let [pm1 (: MlirPassManager) (mlirPassManagerCreate ctx)
            opm1 (: MlirOpPassManager) (mlirPassManagerGetAsOpPassManager pm1)
            parse1 (: MlirLogicalResult)
              (mlirParsePassPipeline opm1 (mlirStringRefCreateFromCString first-pipeline)
                                     (cast (Pointer Nil) 0) (cast (Pointer Nil) 0))]
        (if (= (mlirLogicalResultIsFailure parse1) 1)
          (do
            (printf (c-str "ERROR: Failed to parse SCF lowering pipeline\n"))
            (mlirPassManagerDestroy pm1)
            1)
          (let [run1 (: MlirLogicalResult) (mlirPassManagerRunOnOp pm1 (mlirModuleGetOperation mod))]
            (mlirPassManagerDestroy pm1)
            (if (= (mlirLogicalResultIsFailure run1) 1)
              (do
                (printf (c-str "ERROR: SCF lowering pipeline failed\n"))
                1)
              ;; Second run: lower remaining dialects to LLVM
              (let [pm2 (: MlirPassManager) (mlirPassManagerCreate ctx)
                    opm2 (: MlirOpPassManager) (mlirPassManagerGetAsOpPassManager pm2)
                    parse2 (: MlirLogicalResult)
                      (mlirParsePassPipeline opm2 (mlirStringRefCreateFromCString second-pipeline)
                                              (cast (Pointer Nil) 0) (cast (Pointer Nil) 0))]
                (if (= (mlirLogicalResultIsFailure parse2) 1)
                  (do
                    (printf (c-str "ERROR: Failed to parse LLVM lowering pipeline\n"))
                    (mlirPassManagerDestroy pm2)
                    1)
                  (let [run2 (: MlirLogicalResult) (mlirPassManagerRunOnOp pm2 (mlirModuleGetOperation mod))]
                    (mlirPassManagerDestroy pm2)
                    (if (= (mlirLogicalResultIsFailure run2) 1)
                      (do
                        (printf (c-str "ERROR: LLVM lowering pipeline failed\n"))
                        1)
                      0)))))))))))

;; JIT compile and run a single i32 -> i32 function from the current module
(def jit-run-function-i32 (: (-> [(Pointer MLIRBuilderContext) (Pointer U8) I32] I32))
  (fn [builder fn-name arg0]
    (let [mod (: MlirModule) (pointer-field-read builder mod)
          engine (: MlirExecutionEngine) (mlirExecutionEngineCreate mod 3 0 (cast (Pointer Nil) 0) 0)]
      (if (= (mlirExecutionEngineIsNull engine) 1)
        (do
          (printf (c-str "ERROR: Failed to create execution engine\n"))
          (cast I32 -1))
        (let [fn-ptr (: (Pointer Nil))
              (mlirExecutionEngineLookup engine (mlirStringRefCreateFromCString fn-name))]
          (if (= (cast I64 fn-ptr) 0)
            (do
              (printf (c-str "ERROR: Failed to lookup function %s\n") fn-name)
              (mlirExecutionEngineDestroy engine)
              (cast I32 -1))
            (let [callable (: (Pointer (-> [I32] I32))) (cast (Pointer (-> [I32] I32)) fn-ptr)
                  result (: I32) (callable arg0)]
              (mlirExecutionEngineDestroy engine)
              result)))))))

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
                          raw-count (: I32) (pointer-field-read vector-struct count)]
                      (if (> raw-count 0)
                        (let [first-elem (: (Pointer types/Value)) (vector-element result-types-val 0)
                              first-tag (: types/ValueTag) (pointer-field-read first-elem tag)
                              result-count (: I32) (if (= first-tag types/ValueTag/Vector)
                                                     raw-count
                                                     (cast I32 (/ (cast F64 raw-count) 2.0)))
                              idx (: I32) 0]
                          (while (< idx result-count)
                            (let [result-val (: MlirValue) (mlirOperationGetResult op (cast I64 idx))
                                  name-val (: (Pointer types/Value))
                                    (if (= first-tag types/ValueTag/Vector)
                                      (let [elem (: (Pointer types/Value)) (vector-element result-types-val idx)]
                                        (vector-element elem 0))
                                      (let [name-idx (: I32) (* idx 2)]
                                        (vector-element result-types-val name-idx)))
                                  name-str (: (Pointer U8)) (pointer-field-read name-val str_val)
                                  _ (: I32) (value-tracker-register tracker name-str result-val)]
                              (set! idx (+ idx 1))))
                          op)
                        op))
                    op))))))))))

;; Compile a top-level OpNode to MLIR
(def compile-op-to-mlir (: (-> [(Pointer MLIRBuilderContext) (Pointer types/Value)] I32))
  (fn [builder op-form]
    ;; Parse the top-level op
    (let [op-node (: (Pointer ast/OpNode)) (ast/parse-op op-form)]
      (if (!= (cast I64 op-node) 0)
        (let [mod (: MlirModule) (pointer-field-read builder mod)
              mod-op (: MlirOperation) (mlirModuleGetOperation mod)
              mod-region (: MlirRegion) (mlirOperationGetRegion mod-op 0)
              mod-block (: MlirBlock) (mlirRegionGetFirstBlock mod-region)
              tracker (: (Pointer ValueTracker)) (value-tracker-create)
              _ (: MlirOperation) (build-mlir-operation builder op-node tracker mod-block)]

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
    (let [builder (: (Pointer MLIRBuilderContext)) (mlir-builder-init)]
      ;; Compile the fib test file
      (let [compile-res (: I32) (compile-file builder (c-str "tests/fib.lisp"))]
        (if (= compile-res 0)
          (let [lower-res (: I32) (run-lowering-passes builder)]
            (if (= lower-res 0)
              (let [jit-res (: I32) (jit-run-function-i32 builder (c-str "fib") 10)]
                (mlir-builder-destroy builder)
                (printf (c-str "fib(10) = %d\n") jit-res)
                (if (= jit-res (cast I32 -1))
                  (cast I32 1)
                  0))
              (do
                (mlir-builder-destroy builder)
                lower-res)))
          (do
            (mlir-builder-destroy builder)
            compile-res))))))

(main-fn)
