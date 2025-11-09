;; ========================================
;; MLSP Dialect Transforms
;; ========================================
;; This file contains helper functions and PDL patterns to lower mlsp operations to LLVM

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

;; Helper: create-identifier
;; Creates a CValueLayout* with identifier tag and string data
(defn create-identifier [
  (: %str_ptr !llvm.ptr)
  (: %str_len i64)
] !llvm.ptr

  ;; Constants
  (constant %value_size (: 56 i64))
  (constant %identifier_tag (: 0 i8))
  (constant %c0 (: 0 i64))
  (op %null_ptr (: !llvm.ptr) (llvm.mlir.zero))

  ;; Allocate 56-byte Value struct
  (operation
    (name llvm.call)
    (result-bindings [%value])
    (result-types !llvm.ptr)
    (operands %value_size)
    (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))

  ;; Store fields: type, data_ptr, data_len
  (op %type_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%value]))
  (op (llvm.store [%identifier_tag %type_ptr]))
  (op %data_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%value]))
  (op (llvm.store [%str_ptr %data_ptr]))
  (op %len_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 16>} [%value]))
  (op (llvm.store [%str_len %len_ptr]))

  ;; Zero out remaining fields
  (op %cap_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 24>} [%value]))
  (op (llvm.store [%c0 %cap_ptr]))
  (op %elem_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 32>} [%value]))
  (op (llvm.store [%c0 %elem_ptr]))
  (op %extra1_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 40>} [%value]))
  (op (llvm.store [%null_ptr %extra1_ptr]))
  (op %extra2_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 48>} [%value]))
  (op (llvm.store [%null_ptr %extra2_ptr]))

  (return %value))

;; Helper: create-list
;; Creates a CValueLayout* with list tag and array data
(defn create-list [
  (: %array_ptr !llvm.ptr)
  (: %array_len i64)
] !llvm.ptr

  ;; Constants
  (constant %value_size (: 56 i64))
  (constant %list_tag (: 9 i8))
  (constant %c0 (: 0 i64))
  (op %null_ptr (: !llvm.ptr) (llvm.mlir.zero))

  ;; Allocate 56-byte Value struct
  (operation
    (name llvm.call)
    (result-bindings [%value])
    (result-types !llvm.ptr)
    (operands %value_size)
    (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))

  ;; Store fields: type, data_ptr (array), data_len
  (op %type_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%value]))
  (op (llvm.store [%list_tag %type_ptr]))
  (op %data_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%value]))
  (op (llvm.store [%array_ptr %data_ptr]))
  (op %len_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 16>} [%value]))
  (op (llvm.store [%array_len %len_ptr]))

  ;; Zero out remaining fields
  (op %cap_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 24>} [%value]))
  (op (llvm.store [%c0 %cap_ptr]))
  (op %elem_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 32>} [%value]))
  (op (llvm.store [%c0 %elem_ptr]))
  (op %extra1_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 40>} [%value]))
  (op (llvm.store [%null_ptr %extra1_ptr]))
  (op %extra2_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 48>} [%value]))
  (op (llvm.store [%null_ptr %extra2_ptr]))

  (return %value))

;; Helper: allocate-value-array
;; Allocates an array sized to hold N Value* pointers
(defn allocate-value-array [
  (: %count i64)
] !llvm.ptr

  (constant %ptr_size (: 8 i64))
  (op %alloc_size (: i64) (llvm.mul [%count %ptr_size]))
  (operation
    (name llvm.call)
    (result-bindings [%array])
    (result-types !llvm.ptr)
    (operands %alloc_size)
    (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))
  (return %array))

;; Helper: store-value-at-index
;; Stores a Value* into an array at the specified index
(defn store-value-at-index [
  (: %array !llvm.ptr)
  (: %index i64)
  (: %value !llvm.ptr)
] !llvm.ptr

  (constant %ptr_size (: 8 i64))
  (op %offset (: i64) (llvm.mul [%index %ptr_size]))
  (op %elem_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: -2147483648>} [%array %offset]))
  (op (llvm.store [%value %elem_ptr]))
  (return %array))

;; Helper: get-list-element
;; Extracts a Value* from a CValueLayout list at the specified index
(defn get-list-element [
  (: %list_value !llvm.ptr)
  (: %index i64)
] !llvm.ptr

  (constant %ptr_size (: 8 i64))

  ;; Get data_ptr field (offset 8) - points to array
  (op %data_ptr_field (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%list_value]))
  (op %data_ptr (: !llvm.ptr) (llvm.load [%data_ptr_field]))

  ;; Calculate byte offset and load element
  (op %offset (: i64) (llvm.mul [%index %ptr_size]))
  (op %elem_pp (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: -2147483648>} [%data_ptr %offset]))
  (op %elem_ptr (: !llvm.ptr) (llvm.load [%elem_pp]))

  (return %elem_ptr))

;; ========================================
;; PDL Transform Patterns
;; ========================================

(operation
  (name builtin.module)
  (attributes {:transform.with_named_sequence unit})
  (regions
    (region
      (block
        (operation
          (name transform.named_sequence)
          (attributes {:sym_name @__transform_main :function_type (!function (inputs !transform.any_op) (results))})
          (regions
            (region
              (block [^entry]
                (arguments [(: %arg0 !transform.any_op)])
                (operation
                  (name transform.with_pdl_patterns)
                  (operands %arg0)
                  (regions
                    (region
                      (block
                        (arguments [(: %payload !transform.any_op)])

                        ;; Pattern: mlsp.string_const → llvm.mlir.addressof
                        (operation
                          (name pdl.pattern)
                          (attributes {:benefit (: 1 i16) :sym_name @mlsp_string_const})
                          (regions
                            (region
                              (block
                                (operation
                                  (name pdl.type)
                                  (result-bindings [%ptr_type])
                                  (result-types !pdl.type)
                                  (attributes {:constantType !llvm.ptr}))
                                (operation
                                  (name pdl.operation)
                                  (result-bindings [%op])
                                  (result-types !pdl.operation)
                                  (operands %ptr_type)
                                  (attributes {:opName "mlsp.string_const" :operandSegmentSizes array<i32: 0, 0, 1>}))
                                (operation
                                  (name pdl.rewrite)
                                  (operands %op)
                                  (attributes {:operandSegmentSizes array<i32: 1, 0>})
                                  (regions
                                    (region
                                      (block
                                        (operation
                                          (name pdl.operation)
                                          (result-bindings [%new_op])
                                          (result-types !pdl.operation)
                                          (operands %ptr_type)
                                          (attributes {:opName "llvm.mlir.addressof" :operandSegmentSizes array<i32: 0, 0, 1>}))
                                        (operation
                                          (name pdl.replace)
                                          (operands %op %new_op)
                                          (attributes {:operandSegmentSizes array<i32: 1, 1, 0>}))))))))))

                        (operation
                          (name transform.sequence)
                          (operands %payload)
                          (attributes {:failure_propagation_mode (: 1 i32) :operandSegmentSizes array<i32: 1, 0>})
                          (regions
                            (region
                              (block [^bb1]
                                (arguments [(: %arg1 !transform.any_op)])
                                (operation
                                  (name transform.pdl_match)
                                  (operands %arg1)
                                  (result-bindings [%matched])
                                  (result-types !transform.any_op)
                                  (attributes {:pattern_name @mlsp_string_const}))
                                (operation
                                  (name transform.yield))))))))
                (operation
                  (name transform.yield)))))))))))))
