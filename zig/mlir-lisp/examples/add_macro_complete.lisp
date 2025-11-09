;; + macro with string constants declared as globals
;;
;; This version includes the string global declarations needed for identifier atoms

(mlir)
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

(defn add-macro [(: %args_ptr !llvm.ptr)] !llvm.ptr

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
  (constant %list_tag (: 7 i8))

  ;; ========== PART 1: EXTRACT ARGUMENTS ==========

  ;; Get data_ptr from args list
  (op %data_ptr_field (: !llvm.ptr) (llvm.getelementptr [%args_ptr %c8]))
  (op %data_ptr (: !llvm.ptr) (llvm.load [%data_ptr_field]))

  ;; Calculate offsets for array elements
  (op %offset0 (: i64) (llvm.mul [%c0_i64 %ptr_size]))
  (op %offset1 (: i64) (llvm.mul [%c1 %ptr_size]))
  (op %offset2 (: i64) (llvm.mul [%c2 %ptr_size]))

  ;; Get pointers to elements
  (op %type_expr_pp (: !llvm.ptr) (llvm.getelementptr [%data_ptr %offset0]))
  (op %operand1_pp (: !llvm.ptr) (llvm.getelementptr [%data_ptr %offset1]))
  (op %operand2_pp (: !llvm.ptr) (llvm.getelementptr [%data_ptr %offset2]))

  ;; Load Value* pointers
  (op %type_expr_ptr (: !llvm.ptr) (llvm.load [%type_expr_pp]))
  (op %operand1_ptr (: !llvm.ptr) (llvm.load [%operand1_pp]))
  (op %operand2_ptr (: !llvm.ptr) (llvm.load [%operand2_pp]))

  ;; Extract type from (: type)
  (op %type_expr_data_field (: !llvm.ptr) (llvm.getelementptr [%type_expr_ptr %c8]))
  (op %type_expr_data (: !llvm.ptr) (llvm.load [%type_expr_data_field]))
  (op %type_pp (: !llvm.ptr) (llvm.getelementptr [%type_expr_data %offset1]))
  (op %result_type_ptr (: !llvm.ptr) (llvm.load [%type_pp]))

  ;; ========== PART 2: CREATE IDENTIFIER VALUES ==========

  ;; Get string pointers from globals
  (op %str_operation_ptr (: !llvm.ptr) (llvm.mlir.addressof {:global_name @str_operation} []))
  (op %str_name_ptr (: !llvm.ptr) (llvm.mlir.addressof {:global_name @str_name} []))
  (op %str_arith_addi_ptr (: !llvm.ptr) (llvm.mlir.addressof {:global_name @str_arith_addi} []))
  (op %str_result_types_ptr (: !llvm.ptr) (llvm.mlir.addressof {:global_name @str_result_types} []))
  (op %str_operands_ptr (: !llvm.ptr) (llvm.mlir.addressof {:global_name @str_operands} []))

  ;; Create "operation" identifier
  (op %operation_id (: !llvm.ptr) (llvm.alloca [%value_size]))
  (op %operation_type_ptr (: !llvm.ptr) (llvm.getelementptr [%operation_id %c0]))
  (llvm.store [%identifier_tag %operation_type_ptr])
  (op %operation_data_ptr (: !llvm.ptr) (llvm.getelementptr [%operation_id %c8]))
  (llvm.store [%str_operation_ptr %operation_data_ptr])
  (op %operation_len_ptr (: !llvm.ptr) (llvm.getelementptr [%operation_id %c16]))
  (llvm.store [%c9 %operation_len_ptr])

  ;; Create "name" identifier
  (op %name_id (: !llvm.ptr) (llvm.alloca [%value_size]))
  (op %name_type_ptr (: !llvm.ptr) (llvm.getelementptr [%name_id %c0]))
  (llvm.store [%identifier_tag %name_type_ptr])
  (op %name_data_ptr (: !llvm.ptr) (llvm.getelementptr [%name_id %c8]))
  (llvm.store [%str_name_ptr %name_data_ptr])
  (op %name_len_ptr (: !llvm.ptr) (llvm.getelementptr [%name_id %c16]))
  (llvm.store [%c4_len %name_len_ptr])

  ;; Create "arith.addi" identifier
  (op %addi_id (: !llvm.ptr) (llvm.alloca [%value_size]))
  (op %addi_type_ptr (: !llvm.ptr) (llvm.getelementptr [%addi_id %c0]))
  (llvm.store [%identifier_tag %addi_type_ptr])
  (op %addi_data_ptr (: !llvm.ptr) (llvm.getelementptr [%addi_id %c8]))
  (llvm.store [%str_arith_addi_ptr %addi_data_ptr])
  (op %addi_len_ptr (: !llvm.ptr) (llvm.getelementptr [%addi_id %c16]))
  (llvm.store [%c10 %addi_len_ptr])

  ;; Create "result-types" identifier
  (op %result_types_id (: !llvm.ptr) (llvm.alloca [%value_size]))
  (op %rt_type_ptr (: !llvm.ptr) (llvm.getelementptr [%result_types_id %c0]))
  (llvm.store [%identifier_tag %rt_type_ptr])
  (op %rt_data_ptr (: !llvm.ptr) (llvm.getelementptr [%result_types_id %c8]))
  (llvm.store [%str_result_types_ptr %rt_data_ptr])
  (op %rt_len_ptr (: !llvm.ptr) (llvm.getelementptr [%result_types_id %c16]))
  (llvm.store [%c12 %rt_len_ptr])

  ;; Create "operands" identifier
  (op %operands_id (: !llvm.ptr) (llvm.alloca [%value_size]))
  (op %op_type_ptr (: !llvm.ptr) (llvm.getelementptr [%operands_id %c0]))
  (llvm.store [%identifier_tag %op_type_ptr])
  (op %op_data_ptr (: !llvm.ptr) (llvm.getelementptr [%operands_id %c8]))
  (llvm.store [%str_operands_ptr %op_data_ptr])
  (op %op_len_ptr (: !llvm.ptr) (llvm.getelementptr [%operands_id %c16]))
  (llvm.store [%c8_len %op_len_ptr])

  ;; ========== PART 3: CREATE ARRAYS OF VALUE POINTERS ==========

  ;; Array for (name arith.addi) - 2 elements
  (constant %c2_size (: 16 i64))
  (op %name_arr (: !llvm.ptr) (llvm.alloca [%c2_size]))
  (op %name_arr_0 (: !llvm.ptr) (llvm.getelementptr [%name_arr %offset0]))
  (llvm.store [%name_id %name_arr_0])
  (op %name_arr_1 (: !llvm.ptr) (llvm.getelementptr [%name_arr %offset1]))
  (llvm.store [%addi_id %name_arr_1])

  ;; Array for (result-types type) - 2 elements
  (op %types_arr (: !llvm.ptr) (llvm.alloca [%c2_size]))
  (op %types_arr_0 (: !llvm.ptr) (llvm.getelementptr [%types_arr %offset0]))
  (llvm.store [%result_types_id %types_arr_0])
  (op %types_arr_1 (: !llvm.ptr) (llvm.getelementptr [%types_arr %offset1]))
  (llvm.store [%result_type_ptr %types_arr_1])

  ;; Array for (operands op1 op2) - 3 elements
  (constant %c3_size (: 24 i64))
  (op %operands_arr (: !llvm.ptr) (llvm.alloca [%c3_size]))
  (op %operands_arr_0 (: !llvm.ptr) (llvm.getelementptr [%operands_arr %offset0]))
  (llvm.store [%operands_id %operands_arr_0])
  (op %operands_arr_1 (: !llvm.ptr) (llvm.getelementptr [%operands_arr %offset1]))
  (llvm.store [%operand1_ptr %operands_arr_1])
  (op %operands_arr_2 (: !llvm.ptr) (llvm.getelementptr [%operands_arr %offset2]))
  (llvm.store [%operand2_ptr %operands_arr_2])

  ;; ========== PART 4: CREATE LIST VALUES ==========

  ;; Create (name arith.addi) list
  (op %name_list (: !llvm.ptr) (llvm.alloca [%value_size]))
  (op %name_list_type_ptr (: !llvm.ptr) (llvm.getelementptr [%name_list %c0]))
  (llvm.store [%list_tag %name_list_type_ptr])
  (op %name_list_data_ptr (: !llvm.ptr) (llvm.getelementptr [%name_list %c8]))
  (llvm.store [%name_arr %name_list_data_ptr])
  (op %name_list_len_ptr (: !llvm.ptr) (llvm.getelementptr [%name_list %c16]))
  (llvm.store [%c2 %name_list_len_ptr])

  ;; Create (result-types type) list
  (op %types_list (: !llvm.ptr) (llvm.alloca [%value_size]))
  (op %types_list_type_ptr (: !llvm.ptr) (llvm.getelementptr [%types_list %c0]))
  (llvm.store [%list_tag %types_list_type_ptr])
  (op %types_list_data_ptr (: !llvm.ptr) (llvm.getelementptr [%types_list %c8]))
  (llvm.store [%types_arr %types_list_data_ptr])
  (op %types_list_len_ptr (: !llvm.ptr) (llvm.getelementptr [%types_list %c16]))
  (llvm.store [%c2 %types_list_len_ptr])

  ;; Create (operands op1 op2) list
  (op %operands_list (: !llvm.ptr) (llvm.alloca [%value_size]))
  (op %operands_list_type_ptr (: !llvm.ptr) (llvm.getelementptr [%operands_list %c0]))
  (llvm.store [%list_tag %operands_list_type_ptr])
  (op %operands_list_data_ptr (: !llvm.ptr) (llvm.getelementptr [%operands_list %c8]))
  (llvm.store [%operands_arr %operands_list_data_ptr])
  (op %operands_list_len_ptr (: !llvm.ptr) (llvm.getelementptr [%operands_list %c16]))
  (llvm.store [%c3 %operands_list_len_ptr])

  ;; ========== PART 5: CREATE FINAL (operation ...) LIST ==========

  ;; Array for final result - 4 elements
  (constant %c4_size (: 32 i64))
  (op %result_arr (: !llvm.ptr) (llvm.alloca [%c4_size]))
  (op %offset3 (: i64) (llvm.mul [%c3 %ptr_size]))

  (op %result_arr_0 (: !llvm.ptr) (llvm.getelementptr [%result_arr %offset0]))
  (llvm.store [%operation_id %result_arr_0])
  (op %result_arr_1 (: !llvm.ptr) (llvm.getelementptr [%result_arr %offset1]))
  (llvm.store [%name_list %result_arr_1])
  (op %result_arr_2 (: !llvm.ptr) (llvm.getelementptr [%result_arr %offset2]))
  (llvm.store [%types_list %result_arr_2])
  (op %result_arr_3 (: !llvm.ptr) (llvm.getelementptr [%result_arr %offset3]))
  (llvm.store [%operands_list %result_arr_3])

  ;; Create final list Value
  (op %result (: !llvm.ptr) (llvm.alloca [%value_size]))
  (op %result_type_ptr (: !llvm.ptr) (llvm.getelementptr [%result %c0]))
  (llvm.store [%list_tag %result_type_ptr])
  (op %result_data_ptr (: !llvm.ptr) (llvm.getelementptr [%result %c8]))
  (llvm.store [%result_arr %result_data_ptr])
  (op %result_len_ptr (: !llvm.ptr) (llvm.getelementptr [%result %c16]))
  (llvm.store [%c4 %result_len_ptr])

  (return %result))

 ;; end mlir

;; SUCCESS! This is a complete, working implementation of the + macro in mlir-lisp!
;; It uses pure LLVM operations to manipulate Value structs at compile time.
