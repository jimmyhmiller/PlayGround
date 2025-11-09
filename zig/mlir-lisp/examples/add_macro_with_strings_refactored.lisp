;; + macro with string constants - REFACTORED VERSION
;;
;; This version uses helper functions to eliminate boilerplate and improve readability.
;; All helpers are defined in the same file for simplicity.
;;
;; Benefits:
;; - 5 identifier creations: 5 function calls vs 100 lines of boilerplate
;; - 4 list creations: 4 function calls vs 88 lines of boilerplate
;; - Array operations: cleaner and more maintainable
;; - Total: ~280 lines vs original ~430 lines (35% reduction)



;; ========== INFRASTRUCTURE ==========

;; Declare external malloc function
(operation
  (name llvm.func)
  (attributes {:sym_name @malloc
               :function_type (!function !llvm.func<ptr (i64)>)
               :linkage #llvm.linkage<external>})
  (regions
    (region)))

;; Declare string constants as module-level globals
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

;; ========== HELPER FUNCTIONS ==========

;; Helper: create-identifier
;; Creates a CValueLayout* with identifier tag and string data
;; Replaces ~20 lines of malloc + GEP + store boilerplate
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
;; Replaces ~22 lines of malloc + GEP + store boilerplate
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
;; Each Value* is 8 bytes
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
;; Returns the array pointer for convenience (allows chaining if needed)
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

;; ========== MAIN MACRO: addMacro ==========

(defn addMacro [(: %args_ptr !llvm.ptr)] !llvm.ptr

  ;; Constants
  (constant %c0_i64 (: 0 i64))
  (constant %c1 (: 1 i64))
  (constant %c2 (: 2 i64))
  (constant %c3 (: 3 i64))
  (constant %c4 (: 4 i64))
  (constant %c9 (: 9 i64))    ;; length of "operation"
  (constant %c4_len (: 4 i64))  ;; length of "name"
  (constant %c10 (: 10 i64))  ;; length of "arith.addi"
  (constant %c12 (: 12 i64))  ;; length of "result-types"
  (constant %c8_len (: 8 i64))  ;; length of "operands"

  ;; ========== EXTRACT ARGUMENTS ==========
  ;; Input: (+ (: i32) arg1 arg2)
  ;; Extract: type, arg1, arg2

  (call %type_expr_ptr @get-list-element %args_ptr %c0_i64 !llvm.ptr)
  (call %operand1_ptr @get-list-element %args_ptr %c1 !llvm.ptr)
  (call %operand2_ptr @get-list-element %args_ptr %c2 !llvm.ptr)

  ;; Extract type from (: type) - get element at index 1
  (call %result_type_ptr @get-list-element %type_expr_ptr %c1 !llvm.ptr)

  ;; ========== GET STRING ADDRESSES ==========

  (op %str_operation_ptr (: !llvm.ptr) (llvm.mlir.addressof {:global_name @str_operation} []))
  (op %str_name_ptr (: !llvm.ptr) (llvm.mlir.addressof {:global_name @str_name} []))
  (op %str_arith_addi_ptr (: !llvm.ptr) (llvm.mlir.addressof {:global_name @str_arith_addi} []))
  (op %str_result_types_ptr (: !llvm.ptr) (llvm.mlir.addressof {:global_name @str_result_types} []))
  (op %str_operands_ptr (: !llvm.ptr) (llvm.mlir.addressof {:global_name @str_operands} []))

  ;; ========== CREATE IDENTIFIERS ==========
  ;; Before: 5 × 20 lines = 100 lines of boilerplate
  ;; After: 5 simple function calls!

  (call %operation_id @create-identifier %str_operation_ptr %c9 !llvm.ptr)
  (call %name_id @create-identifier %str_name_ptr %c4_len !llvm.ptr)
  (call %addi_id @create-identifier %str_arith_addi_ptr %c10 !llvm.ptr)
  (call %result_types_id @create-identifier %str_result_types_ptr %c12 !llvm.ptr)
  (call %operands_id @create-identifier %str_operands_ptr %c8_len !llvm.ptr)

  ;; ========== BUILD (name arith.addi) LIST ==========

  (call %name_arr @allocate-value-array %c2 !llvm.ptr)
  (call @store-value-at-index %name_arr %c0_i64 %name_id !llvm.ptr)
  (call @store-value-at-index %name_arr %c1 %addi_id !llvm.ptr)
  (call %name_list @create-list %name_arr %c2 !llvm.ptr)

  ;; ========== BUILD (result-types type) LIST ==========

  (call %types_arr @allocate-value-array %c2 !llvm.ptr)
  (call @store-value-at-index %types_arr %c0_i64 %result_types_id !llvm.ptr)
  (call @store-value-at-index %types_arr %c1 %result_type_ptr !llvm.ptr)
  (call %types_list @create-list %types_arr %c2 !llvm.ptr)

  ;; ========== BUILD (operands op1 op2) LIST ==========

  (call %operands_arr @allocate-value-array %c3 !llvm.ptr)
  (call @store-value-at-index %operands_arr %c0_i64 %operands_id !llvm.ptr)
  (call @store-value-at-index %operands_arr %c1 %operand1_ptr !llvm.ptr)
  (call @store-value-at-index %operands_arr %c2 %operand2_ptr !llvm.ptr)
  (call %operands_list @create-list %operands_arr %c3 !llvm.ptr)

  ;; ========== BUILD FINAL (operation ...) LIST ==========

  (call %result_arr @allocate-value-array %c4 !llvm.ptr)
  (call @store-value-at-index %result_arr %c0_i64 %operation_id !llvm.ptr)
  (call @store-value-at-index %result_arr %c1 %name_list !llvm.ptr)
  (call @store-value-at-index %result_arr %c2 %types_list !llvm.ptr)
  (call @store-value-at-index %result_arr %c3 %operands_list !llvm.ptr)
  (call %result @create-list %result_arr %c4 !llvm.ptr)

  (return %result))


