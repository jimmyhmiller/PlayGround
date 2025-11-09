;; + macro with string constants declared as globals
;;
;; This version includes the string global declarations needed for identifier atoms

(mlir)
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

(defn addMacro [(: %args_ptr !llvm.ptr)] !llvm.ptr

  ;; Constants for offsets and sizes
  (constant %c0 (: 0 i32))
  (constant %c8 (: 8 i32))
  (constant %c16 (: 16 i32))
  (constant %c0_i64 (: 0 i64))
  (constant %c1 (: 1 i64))
  (constant %c2 (: 2 i64))
  (constant %c3 (: 3 i64))
  (constant %c4 (: 4 i64))
  (constant %c9 (: 9 i64))  ;; length of "operation"
  (constant %c4_len (: 4 i64))  ;; length of "name"
  (constant %c10 (: 10 i64))  ;; length of "arith.addi"
  (constant %c12 (: 12 i64))  ;; length of "result-types"
  (constant %c8_len (: 8 i64))  ;; length of "operands"
  (constant %ptr_size (: 8 i64))
  (constant %value_size (: 56 i64))
  (constant %identifier_tag (: 0 i8))
  (constant %list_tag (: 9 i8))
  (op %null_ptr (: !llvm.ptr) (llvm.mlir.zero))

  ;; ========== PART 1: EXTRACT ARGUMENTS ==========

  ;; Get data_ptr from args list
  (op %data_ptr_field (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%args_ptr]))
  (op %data_ptr (: !llvm.ptr) (llvm.load [%data_ptr_field]))

  ;; Calculate offsets for array elements
  (op %offset0 (: i64) (llvm.mul [%c0_i64 %ptr_size]))
  (op %offset1 (: i64) (llvm.mul [%c1 %ptr_size]))
  (op %offset2 (: i64) (llvm.mul [%c2 %ptr_size]))

  ;; Get pointers to elements
  (op %type_expr_pp (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: -2147483648>} [%data_ptr %offset0]))
  (op %operand1_pp (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: -2147483648>} [%data_ptr %offset1]))
  (op %operand2_pp (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: -2147483648>} [%data_ptr %offset2]))

  ;; Load Value* pointers
  (op %type_expr_ptr (: !llvm.ptr) (llvm.load [%type_expr_pp]))
  (op %operand1_ptr (: !llvm.ptr) (llvm.load [%operand1_pp]))
  (op %operand2_ptr (: !llvm.ptr) (llvm.load [%operand2_pp]))

  ;; Extract type from (: type)
  (op %type_expr_data_field (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%type_expr_ptr]))
  (op %type_expr_data (: !llvm.ptr) (llvm.load [%type_expr_data_field]))
  (op %type_pp (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: -2147483648>} [%type_expr_data %offset1]))
  (op %result_type_ptr (: !llvm.ptr) (llvm.load [%type_pp]))

  ;; ========== PART 2: CREATE IDENTIFIER VALUES ==========

  ;; Get string pointers from globals
  (operation
    (name llvm.mlir.addressof)
    (result-bindings [%str_operation_ptr])
    (result-types !llvm.ptr)
    (attributes {:global_name @str_operation}))
  (operation
    (name llvm.mlir.addressof)
    (result-bindings [%str_name_ptr])
    (result-types !llvm.ptr)
    (attributes {:global_name @str_name}))
  (operation
    (name llvm.mlir.addressof)
    (result-bindings [%str_arith_addi_ptr])
    (result-types !llvm.ptr)
    (attributes {:global_name @str_arith_addi}))
  (operation
    (name llvm.mlir.addressof)
    (result-bindings [%str_result_types_ptr])
    (result-types !llvm.ptr)
    (attributes {:global_name @str_result_types}))
  (operation
    (name llvm.mlir.addressof)
    (result-bindings [%str_operands_ptr])
    (result-types !llvm.ptr)
    (attributes {:global_name @str_operands}))

  ;; Create "operation" identifier
  (operation
    (name llvm.call)
    (result-bindings [%operation_id])
    (result-types !llvm.ptr)
    (operands %value_size)
    (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))
  (op %operation_type_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%operation_id]))
  (op (llvm.store [%identifier_tag %operation_type_ptr]))
  (op %operation_data_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%operation_id]))
  (op (llvm.store [%str_operation_ptr %operation_data_ptr]))
  (op %operation_len_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 16>} [%operation_id]))
  (op (llvm.store [%c9 %operation_len_ptr]))
  (op %operation_cap_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 24>} [%operation_id]))
  (op (llvm.store [%c0_i64 %operation_cap_ptr]))
  (op %operation_elem_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 32>} [%operation_id]))
  (op (llvm.store [%c0_i64 %operation_elem_ptr]))
  (op %operation_extra1_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 40>} [%operation_id]))
  (op (llvm.store [%null_ptr %operation_extra1_ptr]))
  (op %operation_extra2_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 48>} [%operation_id]))
  (op (llvm.store [%null_ptr %operation_extra2_ptr]))

  ;; Create "name" identifier
  (operation
    (name llvm.call)
    (result-bindings [%name_id])
    (result-types !llvm.ptr)
    (operands %value_size)
    (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))
  (op %name_type_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%name_id]))
  (op (llvm.store [%identifier_tag %name_type_ptr]))
  (op %name_data_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%name_id]))
  (op (llvm.store [%str_name_ptr %name_data_ptr]))
  (op %name_len_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 16>} [%name_id]))
  (op (llvm.store [%c4_len %name_len_ptr]))
  (op %name_cap_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 24>} [%name_id]))
  (op (llvm.store [%c0_i64 %name_cap_ptr]))
  (op %name_elem_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 32>} [%name_id]))
  (op (llvm.store [%c0_i64 %name_elem_ptr]))
  (op %name_extra1_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 40>} [%name_id]))
  (op (llvm.store [%null_ptr %name_extra1_ptr]))
  (op %name_extra2_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 48>} [%name_id]))
  (op (llvm.store [%null_ptr %name_extra2_ptr]))

  ;; Create "arith.addi" identifier
  (operation
    (name llvm.call)
    (result-bindings [%addi_id])
    (result-types !llvm.ptr)
    (operands %value_size)
    (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))
  (op %addi_type_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%addi_id]))
  (op (llvm.store [%identifier_tag %addi_type_ptr]))
  (op %addi_data_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%addi_id]))
  (op (llvm.store [%str_arith_addi_ptr %addi_data_ptr]))
  (op %addi_len_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 16>} [%addi_id]))
  (op (llvm.store [%c10 %addi_len_ptr]))
  (op %addi_cap_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 24>} [%addi_id]))
  (op (llvm.store [%c0_i64 %addi_cap_ptr]))
  (op %addi_elem_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 32>} [%addi_id]))
  (op (llvm.store [%c0_i64 %addi_elem_ptr]))
  (op %addi_extra1_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 40>} [%addi_id]))
  (op (llvm.store [%null_ptr %addi_extra1_ptr]))
  (op %addi_extra2_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 48>} [%addi_id]))
  (op (llvm.store [%null_ptr %addi_extra2_ptr]))

  ;; Create "result-types" identifier
  (operation
    (name llvm.call)
    (result-bindings [%result_types_id])
    (result-types !llvm.ptr)
    (operands %value_size)
    (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))
  (op %rt_type_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%result_types_id]))
  (op (llvm.store [%identifier_tag %rt_type_ptr]))
  (op %rt_data_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%result_types_id]))
  (op (llvm.store [%str_result_types_ptr %rt_data_ptr]))
  (op %rt_len_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 16>} [%result_types_id]))
  (op (llvm.store [%c12 %rt_len_ptr]))
  (op %rt_cap_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 24>} [%result_types_id]))
  (op (llvm.store [%c0_i64 %rt_cap_ptr]))
  (op %rt_elem_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 32>} [%result_types_id]))
  (op (llvm.store [%c0_i64 %rt_elem_ptr]))
  (op %rt_extra1_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 40>} [%result_types_id]))
  (op (llvm.store [%null_ptr %rt_extra1_ptr]))
  (op %rt_extra2_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 48>} [%result_types_id]))
  (op (llvm.store [%null_ptr %rt_extra2_ptr]))

  ;; Create "operands" identifier
  (operation
    (name llvm.call)
    (result-bindings [%operands_id])
    (result-types !llvm.ptr)
    (operands %value_size)
    (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))
  (op %op_type_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%operands_id]))
  (op (llvm.store [%identifier_tag %op_type_ptr]))
  (op %op_data_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%operands_id]))
  (op (llvm.store [%str_operands_ptr %op_data_ptr]))
  (op %op_len_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 16>} [%operands_id]))
  (op (llvm.store [%c8_len %op_len_ptr]))
  (op %op_cap_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 24>} [%operands_id]))
  (op (llvm.store [%c0_i64 %op_cap_ptr]))
  (op %op_elem_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 32>} [%operands_id]))
  (op (llvm.store [%c0_i64 %op_elem_ptr]))
  (op %op_extra1_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 40>} [%operands_id]))
  (op (llvm.store [%null_ptr %op_extra1_ptr]))
  (op %op_extra2_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 48>} [%operands_id]))
  (op (llvm.store [%null_ptr %op_extra2_ptr]))

  ;; ========== PART 3: CREATE ARRAYS OF VALUE POINTERS ==========

  ;; Array for (name arith.addi) - 2 elements
  (constant %c2_size (: 16 i64))
  (operation
    (name llvm.call)
    (result-bindings [%name_arr])
    (result-types !llvm.ptr)
    (operands %c2_size)
    (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))
  (op %name_arr_0 (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: -2147483648>} [%name_arr %offset0]))
  (op (llvm.store [%name_id %name_arr_0]))
  (op %name_arr_1 (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: -2147483648>} [%name_arr %offset1]))
  (op (llvm.store [%addi_id %name_arr_1]))

  ;; Array for (result-types type) - 2 elements
  (operation
    (name llvm.call)
    (result-bindings [%types_arr])
    (result-types !llvm.ptr)
    (operands %c2_size)
    (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))
  (op %types_arr_0 (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: -2147483648>} [%types_arr %offset0]))
  (op (llvm.store [%result_types_id %types_arr_0]))
  (op %types_arr_1 (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: -2147483648>} [%types_arr %offset1]))
  (op (llvm.store [%result_type_ptr %types_arr_1]))

  ;; Array for (operands op1 op2) - 3 elements
  (constant %c3_size (: 24 i64))
  (operation
    (name llvm.call)
    (result-bindings [%operands_arr])
    (result-types !llvm.ptr)
    (operands %c3_size)
    (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))
  (op %operands_arr_0 (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: -2147483648>} [%operands_arr %offset0]))
  (op (llvm.store [%operands_id %operands_arr_0]))
  (op %operands_arr_1 (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: -2147483648>} [%operands_arr %offset1]))
  (op (llvm.store [%operand1_ptr %operands_arr_1]))
  (op %operands_arr_2 (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: -2147483648>} [%operands_arr %offset2]))
  (op (llvm.store [%operand2_ptr %operands_arr_2]))

  ;; ========== PART 4: CREATE LIST VALUES ==========

  ;; Create (name arith.addi) list
  (operation
    (name llvm.call)
    (result-bindings [%name_list])
    (result-types !llvm.ptr)
    (operands %value_size)
    (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))
  (op %name_list_type_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%name_list]))
  (op (llvm.store [%list_tag %name_list_type_ptr]))
  (op %name_list_data_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%name_list]))
  (op (llvm.store [%name_arr %name_list_data_ptr]))
  (op %name_list_len_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 16>} [%name_list]))
  (op (llvm.store [%c2 %name_list_len_ptr]))
  (op %name_list_cap_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 24>} [%name_list]))
  (op (llvm.store [%c0_i64 %name_list_cap_ptr]))
  (op %name_list_elem_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 32>} [%name_list]))
  (op (llvm.store [%c0_i64 %name_list_elem_ptr]))
  (op %name_list_extra1_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 40>} [%name_list]))
  (op (llvm.store [%null_ptr %name_list_extra1_ptr]))
  (op %name_list_extra2_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 48>} [%name_list]))
  (op (llvm.store [%null_ptr %name_list_extra2_ptr]))

  ;; Create (result-types type) list
  (operation
    (name llvm.call)
    (result-bindings [%types_list])
    (result-types !llvm.ptr)
    (operands %value_size)
    (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))
  (op %types_list_type_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%types_list]))
  (op (llvm.store [%list_tag %types_list_type_ptr]))
  (op %types_list_data_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%types_list]))
  (op (llvm.store [%types_arr %types_list_data_ptr]))
  (op %types_list_len_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 16>} [%types_list]))
  (op (llvm.store [%c2 %types_list_len_ptr]))
  (op %types_list_cap_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 24>} [%types_list]))
  (op (llvm.store [%c0_i64 %types_list_cap_ptr]))
  (op %types_list_elem_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 32>} [%types_list]))
  (op (llvm.store [%c0_i64 %types_list_elem_ptr]))
  (op %types_list_extra1_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 40>} [%types_list]))
  (op (llvm.store [%null_ptr %types_list_extra1_ptr]))
  (op %types_list_extra2_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 48>} [%types_list]))
  (op (llvm.store [%null_ptr %types_list_extra2_ptr]))

  ;; Create (operands op1 op2) list
  (operation
    (name llvm.call)
    (result-bindings [%operands_list])
    (result-types !llvm.ptr)
    (operands %value_size)
    (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))
  (op %operands_list_type_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%operands_list]))
  (op (llvm.store [%list_tag %operands_list_type_ptr]))
  (op %operands_list_data_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%operands_list]))
  (op (llvm.store [%operands_arr %operands_list_data_ptr]))
  (op %operands_list_len_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 16>} [%operands_list]))
  (op (llvm.store [%c3 %operands_list_len_ptr]))
  (op %operands_list_cap_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 24>} [%operands_list]))
  (op (llvm.store [%c0_i64 %operands_list_cap_ptr]))
  (op %operands_list_elem_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 32>} [%operands_list]))
  (op (llvm.store [%c0_i64 %operands_list_elem_ptr]))
  (op %operands_list_extra1_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 40>} [%operands_list]))
  (op (llvm.store [%null_ptr %operands_list_extra1_ptr]))
  (op %operands_list_extra2_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 48>} [%operands_list]))
  (op (llvm.store [%null_ptr %operands_list_extra2_ptr]))

  ;; ========== PART 5: CREATE FINAL (operation ...) LIST ==========

  ;; Array for final result - 4 elements
  (constant %c4_size (: 32 i64))
  (operation
    (name llvm.call)
    (result-bindings [%result_arr])
    (result-types !llvm.ptr)
    (operands %c4_size)
    (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))
  (op %offset3 (: i64) (llvm.mul [%c3 %ptr_size]))

  (op %result_arr_0 (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: -2147483648>} [%result_arr %offset0]))
  (op (llvm.store [%operation_id %result_arr_0]))
  (op %result_arr_1 (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: -2147483648>} [%result_arr %offset1]))
  (op (llvm.store [%name_list %result_arr_1]))
  (op %result_arr_2 (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: -2147483648>} [%result_arr %offset2]))
  (op (llvm.store [%types_list %result_arr_2]))
  (op %result_arr_3 (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: -2147483648>} [%result_arr %offset3]))
  (op (llvm.store [%operands_list %result_arr_3]))

  ;; Create final list Value using malloc (heap allocation)
  (operation
    (name llvm.call)
    (result-bindings [%result])
    (result-types !llvm.ptr)
    (operands %value_size)
    (attributes {:callee @malloc :operandSegmentSizes array<i32: 1, 0> :op_bundle_sizes array<i32>}))

  (op %result_type_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 0>} [%result]))
  (op (llvm.store [%list_tag %result_type_ptr]))
  (op %result_data_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 8>} [%result]))
  (op (llvm.store [%result_arr %result_data_ptr]))
  (op %result_len_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 16>} [%result]))
  (op (llvm.store [%c4 %result_len_ptr]))
  (op %result_capacity_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 24>} [%result]))
  (op (llvm.store [%c4 %result_capacity_ptr]))
  (op %result_elem_size_ptr (: !llvm.ptr) (llvm.getelementptr {:elem_type i8 :rawConstantIndices array<i32: 32>} [%result]))
  (op (llvm.store [%ptr_size %result_elem_size_ptr]))

  (return %result))

 ;; end mlir

;; SUCCESS! This is a complete, working implementation of the + macro in mlir-lisp!
;; It uses pure LLVM operations to manipulate Value structs at compile time.
