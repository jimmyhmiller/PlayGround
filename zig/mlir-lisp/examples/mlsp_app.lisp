;; ========================================
;; MLSP DIALECT: High-Level Macro Building
;; ========================================
;;
;; Application file with IRDL dialect definition and macro implementation

;; ========================================
;; PART 1: IRDL Dialect Definition
;; ========================================

(operation
  (name builtin.module)
  (attributes {:metadata unit})
  (regions
    (region
      (block
        (operation
          (name irdl.dialect)
          (attributes {:sym_name @mlsp})
          (regions
            (region
              (block
                ;; mlsp.identifier - Create identifier atom
                (operation
                  (name irdl.operation)
                  (attributes {:sym_name @identifier})
                  (regions
                    (region
                      (block
                        (operation
                          (name irdl.is)
                          (result-bindings [%ptr_type])
                          (result-types !irdl.attribute)
                          (attributes {:expected !llvm.ptr}))
                        (operation
                          (name irdl.operands)
                          (attributes {:names ["value"] :variadicity #irdl<variadicity_array[ single]>})
                          (operand-uses %ptr_type))
                        (operation
                          (name irdl.results)
                          (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>})
                          (operand-uses %ptr_type))))))

                ;; mlsp.identifier_const - Create identifier directly from global string
                (operation
                  (name irdl.operation)
                  (attributes {:sym_name @identifier_const})
                  (regions
                    (region
                      (block
                        ;; Attribute: global symbol reference
                        (operation
                          (name irdl.any)
                          (result-bindings [%any_attr])
                          (result-types !irdl.attribute))
                        (operation
                          (name irdl.attributes)
                          (attributes {:attributeValueNames ["global"]})
                          (operand-uses %any_attr))
                        ;; Result: !llvm.ptr (CValueLayout*)
                        (operation
                          (name irdl.is)
                          (result-bindings [%ptr_type])
                          (result-types !irdl.attribute)
                          (attributes {:expected !llvm.ptr}))
                        (operation
                          (name irdl.results)
                          (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>})
                          (operand-uses %ptr_type))))))

                ;; mlsp.list - Create list from elements
                (operation
                  (name irdl.operation)
                  (attributes {:sym_name @list})
                  (regions
                    (region
                      (block
                        (operation
                          (name irdl.is)
                          (result-bindings [%ptr_type])
                          (result-types !irdl.attribute)
                          (attributes {:expected !llvm.ptr}))
                        (operation
                          (name irdl.operands)
                          (attributes {:names ["elements"] :variadicity #irdl<variadicity_array[ variadic]>})
                          (operand-uses %ptr_type))
                        (operation
                          (name irdl.results)
                          (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>})
                          (operand-uses %ptr_type))))))

                ;; mlsp.get_element - Extract list element by index
                (operation
                  (name irdl.operation)
                  (attributes {:sym_name @get_element})
                  (regions
                    (region
                      (block
                        (operation
                          (name irdl.is)
                          (result-bindings [%ptr_type])
                          (result-types !irdl.attribute)
                          (attributes {:expected !llvm.ptr}))
                        (operation
                          (name irdl.is)
                          (result-bindings [%i64_type])
                          (result-types !irdl.attribute)
                          (attributes {:expected i64}))
                        (operation
                          (name irdl.operands)
                          (attributes {:names ["list" "index"] :variadicity #irdl<variadicity_array[ single,  single]>})
                          (operand-uses %ptr_type %i64_type))
                        (operation
                          (name irdl.results)
                          (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>})
                          (operand-uses %ptr_type))))))

                ;; mlsp.string_const - Reference global string
                (operation
                  (name irdl.operation)
                  (attributes {:sym_name @string_const})
                  (regions
                    (region
                      (block
                        (operation
                          (name irdl.any)
                          (result-bindings [%any_attr])
                          (result-types !irdl.attribute))
                        (operation
                          (name irdl.attributes)
                          (attributes {:attributeValueNames ["global"]})
                          (operand-uses %any_attr))
                        (operation
                          (name irdl.is)
                          (result-bindings [%ptr_type])
                          (result-types !irdl.attribute)
                          (attributes {:expected !llvm.ptr}))
                        (operation
                          (name irdl.results)
                          (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>})
                          (operand-uses %ptr_type))))))))))))))



(operation
    (name builtin.module)
    (attributes {:metadata true})
    (regions
      (region
        (block
          (arguments [])
          (operation
            (name transform.named_sequence)
            (attributes {:function_type (!function (inputs !transform.any_op) (results)) :sym_name @__transform_main})
            (regions
              (region
                (block [^bb0]
                  (arguments [(: %arg0 !transform.any_op)])
                  (operation
                    (name transform.with_pdl_patterns)
                    (operand-uses %arg0)
                    (regions
                      (region
                        (block [^bb0]
                          (arguments [(: %arg1 !transform.any_op)])
                          (operation
                            (name pdl.pattern)
                            (attributes {:benefit (: 1 i16) :sym_name @mlsp_string_const})
                            (regions
                              (region
                                (block
                                  (arguments [])
                                  (operation
                                    (name pdl.type)
                                    (result-bindings [%42])
                                    (result-types !pdl.type)
                                    (attributes {:constantType !llvm.ptr}))
                                  (operation
                                    (name pdl.attribute)
                                    (result-bindings [%43])
                                    (result-types !pdl.attribute))
                                  (operation
                                    (name pdl.operation)
                                    (result-bindings [%44])
                                    (result-types !pdl.operation)
                                    (operand-uses %43 %42)
                                    (attributes {:attributeValueNames ["global"] :opName "mlsp.string_const" :operandSegmentSizes array<i32: 0, 1, 1>}))
                                  (operation
                                    (name pdl.rewrite)
                                    (operand-uses %44)
                                    (attributes {:operandSegmentSizes array<i32: 1, 0>})
                                    (regions
                                      (region
                                        (block
                                          (arguments [])
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%45])
                                            (result-types !pdl.operation)
                                            (operand-uses %43 %42)
                                            (attributes {:attributeValueNames ["global_name"] :opName "llvm.mlir.addressof" :operandSegmentSizes array<i32: 0, 1, 1>}))
                                          (operation
                                            (name pdl.replace)
                                            (operand-uses %44 %45)
                                            (attributes {:operandSegmentSizes array<i32: 1, 1, 0>}))))))))))
                          (operation
                            (name pdl.pattern)
                            (attributes {:benefit (: 1 i16) :sym_name @mlsp_get_element})
                            (regions
                              (region
                                (block
                                  (arguments [])
                                  (operation
                                    (name pdl.type)
                                    (result-bindings [%36])
                                    (result-types !pdl.type)
                                    (attributes {:constantType !llvm.ptr}))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%37])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%38])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operation)
                                    (result-bindings [%39])
                                    (result-types !pdl.operation)
                                    (operand-uses %37 %38 %36)
                                    (attributes {:attributeValueNames [] :opName "mlsp.get_element" :operandSegmentSizes array<i32: 2, 0, 1>}))
                                  (operation
                                    (name pdl.rewrite)
                                    (operand-uses %39)
                                    (attributes {:operandSegmentSizes array<i32: 1, 0>})
                                    (regions
                                      (region
                                        (block
                                          (arguments [])
                                          (operation
                                            (name pdl.attribute)
                                            (result-bindings [%40])
                                            (result-types !pdl.attribute)
                                            (attributes {:value @get_list_element}))
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%41])
                                            (result-types !pdl.operation)
                                            (operand-uses %37 %38 %40 %36)
                                            (attributes {:attributeValueNames ["callee"] :opName "func.call" :operandSegmentSizes array<i32: 2, 1, 1>}))
                                          (operation
                                            (name pdl.replace)
                                            (operand-uses %39 %41)
                                            (attributes {:operandSegmentSizes array<i32: 1, 1, 0>}))))))))))
                          (operation
                            (name pdl.pattern)
                            (attributes {:benefit (: 1 i16) :sym_name @mlsp_identifier})
                            (regions
                              (region
                                (block
                                  (arguments [])
                                  (operation
                                    (name pdl.type)
                                    (result-bindings [%27])
                                    (result-types !pdl.type)
                                    (attributes {:constantType !llvm.ptr}))
                                  (operation
                                    (name pdl.type)
                                    (result-bindings [%28])
                                    (result-types !pdl.type)
                                    (attributes {:constantType i64}))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%29])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operation)
                                    (result-bindings [%30])
                                    (result-types !pdl.operation)
                                    (operand-uses %29 %27)
                                    (attributes {:attributeValueNames [] :opName "mlsp.identifier" :operandSegmentSizes array<i32: 1, 0, 1>}))
                                  (operation
                                    (name pdl.rewrite)
                                    (operand-uses %30)
                                    (attributes {:operandSegmentSizes array<i32: 1, 0>})
                                    (regions
                                      (region
                                        (block
                                          (arguments [])
                                          (operation
                                            (name pdl.attribute)
                                            (result-bindings [%31])
                                            (result-types !pdl.attribute)
                                            (attributes {:value (: 0 i64)}))
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%32])
                                            (result-types !pdl.operation)
                                            (operand-uses %31 %28)
                                            (attributes {:attributeValueNames ["value"] :opName "arith.constant" :operandSegmentSizes array<i32: 0, 1, 1>}))
                                          (operation
                                            (name pdl.result)
                                            (result-bindings [%33])
                                            (result-types !pdl.value)
                                            (operand-uses %32)
                                            (attributes {:index (: 0 i32)}))
                                          (operation
                                            (name pdl.attribute)
                                            (result-bindings [%34])
                                            (result-types !pdl.attribute)
                                            (attributes {:value @create_identifier}))
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%35])
                                            (result-types !pdl.operation)
                                            (operand-uses %29 %33 %34 %27)
                                            (attributes {:attributeValueNames ["callee"] :opName "func.call" :operandSegmentSizes array<i32: 2, 1, 1>}))
                                          (operation
                                            (name pdl.replace)
                                            (operand-uses %30 %35)
                                            (attributes {:operandSegmentSizes array<i32: 1, 1, 0>}))))))))))
                          ;; mlsp.identifier_const -> llvm.addressof + create_identifier
                          (operation
                            (name pdl.pattern)
                            (attributes {:benefit (: 1 i16) :sym_name @mlsp_identifier_const})
                            (regions
                              (region
                                (block
                                  (arguments [])
                                  (operation
                                    (name pdl.type)
                                    (result-bindings [%50])
                                    (result-types !pdl.type)
                                    (attributes {:constantType !llvm.ptr}))
                                  (operation
                                    (name pdl.type)
                                    (result-bindings [%51])
                                    (result-types !pdl.type)
                                    (attributes {:constantType i64}))
                                  (operation
                                    (name pdl.attribute)
                                    (result-bindings [%52])
                                    (result-types !pdl.attribute))
                                  (operation
                                    (name pdl.operation)
                                    (result-bindings [%53])
                                    (result-types !pdl.operation)
                                    (operand-uses %52 %50)
                                    (attributes {:attributeValueNames ["global"] :opName "mlsp.identifier_const" :operandSegmentSizes array<i32: 0, 1, 1>}))
                                  (operation
                                    (name pdl.rewrite)
                                    (operand-uses %53)
                                    (attributes {:operandSegmentSizes array<i32: 1, 0>})
                                    (regions
                                      (region
                                        (block
                                          (arguments [])
                                          ;; Get string address
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%54])
                                            (result-types !pdl.operation)
                                            (operand-uses %52 %50)
                                            (attributes {:attributeValueNames ["global_name"] :opName "llvm.mlir.addressof" :operandSegmentSizes array<i32: 0, 1, 1>}))
                                          (operation
                                            (name pdl.result)
                                            (result-bindings [%55])
                                            (result-types !pdl.value)
                                            (operand-uses %54)
                                            (attributes {:index (: 0 i32)}))
                                          ;; Create constant 0 for type
                                          (operation
                                            (name pdl.attribute)
                                            (result-bindings [%56])
                                            (result-types !pdl.attribute)
                                            (attributes {:value (: 0 i64)}))
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%57])
                                            (result-types !pdl.operation)
                                            (operand-uses %56 %51)
                                            (attributes {:attributeValueNames ["value"] :opName "arith.constant" :operandSegmentSizes array<i32: 0, 1, 1>}))
                                          (operation
                                            (name pdl.result)
                                            (result-bindings [%58])
                                            (result-types !pdl.value)
                                            (operand-uses %57)
                                            (attributes {:index (: 0 i32)}))
                                          ;; Call create_identifier
                                          (operation
                                            (name pdl.attribute)
                                            (result-bindings [%59])
                                            (result-types !pdl.attribute)
                                            (attributes {:value @create_identifier}))
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%60])
                                            (result-types !pdl.operation)
                                            (operand-uses %55 %58 %59 %50)
                                            (attributes {:attributeValueNames ["callee"] :opName "func.call" :operandSegmentSizes array<i32: 2, 1, 1>}))
                                          (operation
                                            (name pdl.replace)
                                            (operand-uses %53 %60)
                                            (attributes {:operandSegmentSizes array<i32: 1, 1, 0>}))))))))))
                          (operation
                            (name pdl.pattern)
                            (attributes {:benefit (: 1 i16) :sym_name @mlsp_list_2})
                            (regions
                              (region
                                (block
                                  (arguments [])
                                  (operation
                                    (name pdl.type)
                                    (result-bindings [%21])
                                    (result-types !pdl.type)
                                    (attributes {:constantType !llvm.ptr}))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%22])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%23])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operation)
                                    (result-bindings [%24])
                                    (result-types !pdl.operation)
                                    (operand-uses %22 %23 %21)
                                    (attributes {:attributeValueNames [] :opName "mlsp.list" :operandSegmentSizes array<i32: 2, 0, 1>}))
                                  (operation
                                    (name pdl.rewrite)
                                    (operand-uses %24)
                                    (attributes {:operandSegmentSizes array<i32: 1, 0>})
                                    (regions
                                      (region
                                        (block
                                          (arguments [])
                                          (operation
                                            (name pdl.attribute)
                                            (result-bindings [%25])
                                            (result-types !pdl.attribute)
                                            (attributes {:value @create_list_2}))
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%26])
                                            (result-types !pdl.operation)
                                            (operand-uses %22 %23 %25 %21)
                                            (attributes {:attributeValueNames ["callee"] :opName "func.call" :operandSegmentSizes array<i32: 2, 1, 1>}))
                                          (operation
                                            (name pdl.replace)
                                            (operand-uses %24 %26)
                                            (attributes {:operandSegmentSizes array<i32: 1, 1, 0>}))))))))))
                          (operation
                            (name pdl.pattern)
                            (attributes {:benefit (: 1 i16) :sym_name @mlsp_list_3})
                            (regions
                              (region
                                (block
                                  (arguments [])
                                  (operation
                                    (name pdl.type)
                                    (result-bindings [%14])
                                    (result-types !pdl.type)
                                    (attributes {:constantType !llvm.ptr}))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%15])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%16])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%17])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operation)
                                    (result-bindings [%18])
                                    (result-types !pdl.operation)
                                    (operand-uses %15 %16 %17 %14)
                                    (attributes {:attributeValueNames [] :opName "mlsp.list" :operandSegmentSizes array<i32: 3, 0, 1>}))
                                  (operation
                                    (name pdl.rewrite)
                                    (operand-uses %18)
                                    (attributes {:operandSegmentSizes array<i32: 1, 0>})
                                    (regions
                                      (region
                                        (block
                                          (arguments [])
                                          (operation
                                            (name pdl.attribute)
                                            (result-bindings [%19])
                                            (result-types !pdl.attribute)
                                            (attributes {:value @create_list_3}))
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%20])
                                            (result-types !pdl.operation)
                                            (operand-uses %15 %16 %17 %19 %14)
                                            (attributes {:attributeValueNames ["callee"] :opName "func.call" :operandSegmentSizes array<i32: 3, 1, 1>}))
                                          (operation
                                            (name pdl.replace)
                                            (operand-uses %18 %20)
                                            (attributes {:operandSegmentSizes array<i32: 1, 1, 0>}))))))))))
                          (operation
                            (name pdl.pattern)
                            (attributes {:benefit (: 1 i16) :sym_name @mlsp_list_4})
                            (regions
                              (region
                                (block
                                  (arguments [])
                                  (operation
                                    (name pdl.type)
                                    (result-bindings [%6])
                                    (result-types !pdl.type)
                                    (attributes {:constantType !llvm.ptr}))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%7])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%8])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%9])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operand)
                                    (result-bindings [%10])
                                    (result-types !pdl.value))
                                  (operation
                                    (name pdl.operation)
                                    (result-bindings [%11])
                                    (result-types !pdl.operation)
                                    (operand-uses %7 %8 %9 %10 %6)
                                    (attributes {:attributeValueNames [] :opName "mlsp.list" :operandSegmentSizes array<i32: 4, 0, 1>}))
                                  (operation
                                    (name pdl.rewrite)
                                    (operand-uses %11)
                                    (attributes {:operandSegmentSizes array<i32: 1, 0>})
                                    (regions
                                      (region
                                        (block
                                          (arguments [])
                                          (operation
                                            (name pdl.attribute)
                                            (result-bindings [%12])
                                            (result-types !pdl.attribute)
                                            (attributes {:value @create_list_4}))
                                          (operation
                                            (name pdl.operation)
                                            (result-bindings [%13])
                                            (result-types !pdl.operation)
                                            (operand-uses %7 %8 %9 %10 %12 %6)
                                            (attributes {:attributeValueNames ["callee"] :opName "func.call" :operandSegmentSizes array<i32: 4, 1, 1>}))
                                          (operation
                                            (name pdl.replace)
                                            (operand-uses %11 %13)
                                            (attributes {:operandSegmentSizes array<i32: 1, 1, 0>}))))))))))
                          (operation
                            (name transform.sequence)
                            (operand-uses %arg1)
                            (attributes {:failure_propagation_mode (: 1 i32) :operandSegmentSizes array<i32: 1, 0>})
                            (regions
                              (region
                                (block [^bb0]
                                  (arguments [(: %arg2 !transform.any_op)])
                                  (operation
                                    (name transform.pdl_match)
                                    (result-bindings [%0])
                                    (result-types !transform.any_op)
                                    (operand-uses %arg2)
                                    (attributes {:pattern_name @mlsp_string_const}))
                                  (operation
                                    (name transform.pdl_match)
                                    (result-bindings [%1])
                                    (result-types !transform.any_op)
                                    (operand-uses %arg2)
                                    (attributes {:pattern_name @mlsp_get_element}))
                                  (operation
                                    (name transform.pdl_match)
                                    (result-bindings [%2])
                                    (result-types !transform.any_op)
                                    (operand-uses %arg2)
                                    (attributes {:pattern_name @mlsp_identifier}))
                                  (operation
                                    (name transform.pdl_match)
                                    (result-bindings [%6])
                                    (result-types !transform.any_op)
                                    (operand-uses %arg2)
                                    (attributes {:pattern_name @mlsp_identifier_const}))
                                  (operation
                                    (name transform.pdl_match)
                                    (result-bindings [%3])
                                    (result-types !transform.any_op)
                                    (operand-uses %arg2)
                                    (attributes {:pattern_name @mlsp_list_2}))
                                  (operation
                                    (name transform.pdl_match)
                                    (result-bindings [%4])
                                    (result-types !transform.any_op)
                                    (operand-uses %arg2)
                                    (attributes {:pattern_name @mlsp_list_3}))
                                  (operation
                                    (name transform.pdl_match)
                                    (result-bindings [%5])
                                    (result-types !transform.any_op)
                                    (operand-uses %arg2)
                                    (attributes {:pattern_name @mlsp_list_4}))
                                  (operation
                                    (name transform.yield))))))))))
                  (operation
                    (name transform.yield))))))))))

;; ========== INFRASTRUCTURE ==========

;; Declare external malloc function
(operation
  (name llvm.func)
  (attributes {:sym_name @malloc
               :function_type (!function !llvm.func<ptr (i64)>)
               :linkage #llvm.linkage<external>})
  (regions
    (region)))

;; ========== HELPER FUNCTIONS ==========

;; Helper: get_list_element
(defn get_list_element [
  (: %list_value !llvm.ptr)
  (: %index i64)] !llvm.ptr

  (constant %ptr_size (: 8 i64))
  (op %data_ptr_field (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%list_value]))
  (op %data_ptr (: !llvm.ptr) (llvm.load [%data_ptr_field]))
  (op %offset (: i64) (llvm.mul [%index %ptr_size]))
  (op %elem_pp (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: -2147483648>} [%data_ptr %offset]))
  (op %elem_ptr (: !llvm.ptr) (llvm.load [%elem_pp]))
  (return %elem_ptr))

;; Helper: create_identifier
(defn create_identifier [
  (: %str_ptr !llvm.ptr)
  (: %str_len i64)] !llvm.ptr
  (constant %value_size (: 56 i64))
  (constant %identifier_tag (: 0 i8))
  (constant %c0 (: 0 i64))
  (op %null_ptr (: !llvm.ptr) (llvm.mlir.zero))
  (operation (name llvm.call) (result-bindings [%value]) (result-types !llvm.ptr) (operand-uses %value_size) (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))
  (op %type_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%value]))
  (op (llvm.store [%identifier_tag %type_ptr]))
  (op %data_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%value]))
  (op (llvm.store [%str_ptr %data_ptr]))
  (op %len_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 16>} [%value]))
  (op (llvm.store [%str_len %len_ptr]))
  (op %cap_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 24>} [%value]))
  (op (llvm.store [%c0 %cap_ptr]))
  (op %elem_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 32>} [%value]))
  (op (llvm.store [%c0 %elem_ptr]))
  (op %extra1_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 40>} [%value]))
  (op (llvm.store [%null_ptr %extra1_ptr]))
  (op %extra2_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 48>} [%value]))
  (op (llvm.store [%null_ptr %extra2_ptr]))
  (return %value))

;; Helper: create_list_2 (for 2-element lists)
(defn create_list_2 [
  (: %elem0 !llvm.ptr)
  (: %elem1 !llvm.ptr)] !llvm.ptr
  (constant %c2 (: 2 i64))
  (constant %c8 (: 8 i64))
  (constant %c16 (: 16 i64))
  (op %array_size (: i64) (llvm.mul [%c2 %c8]))
  (operation (name llvm.call) (result-bindings [%array]) (result-types !llvm.ptr) (operand-uses %array_size) (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))
  (op %ptr0 (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%array]))
  (op (llvm.store [%elem0 %ptr0]))
  (op %ptr1 (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%array]))
  (op (llvm.store [%elem1 %ptr1]))

  (constant %value_size (: 56 i64))
  (constant %list_tag (: 9 i8))
  (constant %c0 (: 0 i64))
  (op %null_ptr (: !llvm.ptr) (llvm.mlir.zero))
  (operation (name llvm.call) (result-bindings [%value]) (result-types !llvm.ptr) (operand-uses %value_size) (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))
  (op %type_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%value]))
  (op (llvm.store [%list_tag %type_ptr]))
  (op %data_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%value]))
  (op (llvm.store [%array %data_ptr]))
  (op %len_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 16>} [%value]))
  (op (llvm.store [%c2 %len_ptr]))
  (op %cap_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 24>} [%value]))
  (op (llvm.store [%c0 %cap_ptr]))
  (op %elem_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 32>} [%value]))
  (op (llvm.store [%c0 %elem_ptr]))
  (op %extra1_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 40>} [%value]))
  (op (llvm.store [%null_ptr %extra1_ptr]))
  (op %extra2_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 48>} [%value]))
  (op (llvm.store [%null_ptr %extra2_ptr]))
  (return %value))

;; Helper: create_list_3 (for 3-element lists)
(defn create_list_3 [
  (: %elem0 !llvm.ptr)
  (: %elem1 !llvm.ptr)
  (: %elem2 !llvm.ptr)] !llvm.ptr
  (constant %c3 (: 3 i64))
  (constant %c8 (: 8 i64))
  (op %array_size (: i64) (llvm.mul [%c3 %c8]))
  (operation (name llvm.call) (result-bindings [%array]) (result-types !llvm.ptr) (operand-uses %array_size) (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))
  (op %ptr0 (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%array]))
  (op (llvm.store [%elem0 %ptr0]))
  (op %ptr1 (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%array]))
  (op (llvm.store [%elem1 %ptr1]))
  (op %ptr2 (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 16>} [%array]))
  (op (llvm.store [%elem2 %ptr2]))

  (constant %value_size (: 56 i64))
  (constant %list_tag (: 9 i8))
  (constant %c0 (: 0 i64))
  (op %null_ptr (: !llvm.ptr) (llvm.mlir.zero))
  (operation (name llvm.call) (result-bindings [%value]) (result-types !llvm.ptr) (operand-uses %value_size) (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))
  (op %type_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%value]))
  (op (llvm.store [%list_tag %type_ptr]))
  (op %data_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%value]))
  (op (llvm.store [%array %data_ptr]))
  (op %len_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 16>} [%value]))
  (op (llvm.store [%c3 %len_ptr]))
  (op %cap_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 24>} [%value]))
  (op (llvm.store [%c0 %cap_ptr]))
  (op %elem_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 32>} [%value]))
  (op (llvm.store [%c0 %elem_ptr]))
  (op %extra1_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 40>} [%value]))
  (op (llvm.store [%null_ptr %extra1_ptr]))
  (op %extra2_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 48>} [%value]))
  (op (llvm.store [%null_ptr %extra2_ptr]))
  (return %value))

;; Helper: create_list_4 (for 4-element lists)
(defn create_list_4 [
  (: %elem0 !llvm.ptr)
  (: %elem1 !llvm.ptr)
  (: %elem2 !llvm.ptr)
  (: %elem3 !llvm.ptr)] !llvm.ptr
  (constant %c4 (: 4 i64))
  (constant %c8 (: 8 i64))
  (op %array_size (: i64) (llvm.mul [%c4 %c8]))
  (operation (name llvm.call) (result-bindings [%array]) (result-types !llvm.ptr) (operand-uses %array_size) (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))
  (op %ptr0 (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%array]))
  (op (llvm.store [%elem0 %ptr0]))
  (op %ptr1 (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%array]))
  (op (llvm.store [%elem1 %ptr1]))
  (op %ptr2 (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 16>} [%array]))
  (op (llvm.store [%elem2 %ptr2]))
  (op %ptr3 (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 24>} [%array]))
  (op (llvm.store [%elem3 %ptr3]))

  (constant %value_size (: 56 i64))
  (constant %list_tag (: 9 i8))
  (constant %c0 (: 0 i64))
  (op %null_ptr (: !llvm.ptr) (llvm.mlir.zero))
  (operation (name llvm.call) (result-bindings [%value]) (result-types !llvm.ptr) (operand-uses %value_size) (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))
  (op %type_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%value]))
  (op (llvm.store [%list_tag %type_ptr]))
  (op %data_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%value]))
  (op (llvm.store [%array %data_ptr]))
  (op %len_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 16>} [%value]))
  (op (llvm.store [%c4 %len_ptr]))
  (op %cap_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 24>} [%value]))
  (op (llvm.store [%c0 %cap_ptr]))
  (op %elem_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 32>} [%value]))
  (op (llvm.store [%c0 %elem_ptr]))
  (op %extra1_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 40>} [%value]))
  (op (llvm.store [%null_ptr %extra1_ptr]))
  (op %extra2_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 48>} [%value]))
  (op (llvm.store [%null_ptr %extra2_ptr]))
  (return %value))

;; ========== STRING CONSTANTS ==========

(operation
  (name llvm.mlir.global)
  (attributes {:sym_name @str_operation
               :value "operation\00"
               :global_type !llvm.array<10 x i8>
               :linkage #llvm.linkage<internal>
               :constant true
               :unnamed_addr (: 2 i64)
               :addr_space (: 0 i32)
               :alignment (: 1 i64)
               :dso_local true
               :visibility_ (: 0 i64)})
  (regions (region)))

(operation
  (name llvm.mlir.global)
  (attributes {:sym_name @str_name
               :value "name\00"
               :global_type !llvm.array<5 x i8>
               :linkage #llvm.linkage<internal>
               :constant true
               :unnamed_addr (: 2 i64)
               :addr_space (: 0 i32)
               :alignment (: 1 i64)
               :dso_local true
               :visibility_ (: 0 i64)})
  (regions (region)))

(operation
  (name llvm.mlir.global)
  (attributes {:sym_name @str_arith_addi
               :value "arith.addi\00"
               :global_type !llvm.array<11 x i8>
               :linkage #llvm.linkage<internal>
               :constant true
               :unnamed_addr (: 2 i64)
               :addr_space (: 0 i32)
               :alignment (: 1 i64)
               :dso_local true
               :visibility_ (: 0 i64)})
  (regions (region)))

(operation
  (name llvm.mlir.global)
  (attributes {:sym_name @str_result_types
               :value "result-types\00"
               :global_type !llvm.array<13 x i8>
               :linkage #llvm.linkage<internal>
               :constant true
               :unnamed_addr (: 2 i64)
               :addr_space (: 0 i32)
               :alignment (: 1 i64)
               :dso_local true
               :visibility_ (: 0 i64)})
  (regions (region)))

(operation
  (name llvm.mlir.global)
  (attributes {:sym_name @str_operands
               :value "operands\00"
               :global_type !llvm.array<9 x i8>
               :linkage #llvm.linkage<internal>
               :constant true
               :unnamed_addr (: 2 i64)
               :addr_space (: 0 i32)
               :alignment (: 1 i64)
               :dso_local true
               :visibility_ (: 0 i64)})
  (regions (region)))

;; ========== MAIN MACRO: addMacro using MLSP ==========

(defn addMacro [(: %args_ptr !llvm.ptr)] !llvm.ptr

  ;; Constants
  (constant %c0 (: 0 i64))
  (constant %c1 (: 1 i64))
  (constant %c2 (: 2 i64))

  ;; ========== EXTRACT ARGUMENTS ==========
  (op %type_expr_ptr (: !llvm.ptr) (mlsp.get_element [%args_ptr %c0]))
  (op %operand1_ptr (: !llvm.ptr) (mlsp.get_element [%args_ptr %c1]))
  (op %operand2_ptr (: !llvm.ptr) (mlsp.get_element [%args_ptr %c2]))

  ;; Extract type from (: type)
  (op %result_type_ptr (: !llvm.ptr) (mlsp.get_element [%type_expr_ptr %c1]))

  ;; ========== CREATE IDENTIFIERS FROM CONSTANTS ==========
  (op %operation_id (: !llvm.ptr) (mlsp.identifier_const {:global @str_operation}))
  (op %name_id (: !llvm.ptr) (mlsp.identifier_const {:global @str_name}))
  (op %addi_id (: !llvm.ptr) (mlsp.identifier_const {:global @str_arith_addi}))
  (op %result_types_id (: !llvm.ptr) (mlsp.identifier_const {:global @str_result_types}))
  (op %operands_id (: !llvm.ptr) (mlsp.identifier_const {:global @str_operands}))

  ;; ========== BUILD LISTS ==========
  ;; (name arith.addi)
  (op %name_list (: !llvm.ptr) (mlsp.list [%name_id %addi_id]))

  ;; (result-types type)
  (op %types_list (: !llvm.ptr) (mlsp.list [%result_types_id %result_type_ptr]))

  ;; (operands op1 op2)
  (op %operands_list (: !llvm.ptr) (mlsp.list [%operands_id %operand1_ptr %operand2_ptr]))

  ;; (operation ...)
  (op %result (: !llvm.ptr) (mlsp.list [%operation_id %name_list %types_list %operands_list]))

  (return %result))

;; Test main function
(defn main [] i64
  (constant %result (: 123 i64))
  (return %result))
