;; + macro using MLSP dialect operations
;;
;; This version demonstrates the mlsp dialect by reimplementing the addMacro
;; using high-level operations instead of raw LLVM malloc/GEP/store.
;;
;; Comparison:
;; - Original: ~430 lines with helper functions
;; - Refactored: ~280 lines with helper functions
;; - MLSP version: ~60 lines (80% reduction!)

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
  ;; Input: (+ (: i32) arg1 arg2)
  ;; Extract: type, arg1, arg2

  (op %type_expr_ptr (: !llvm.ptr) (mlsp.get_element [%args_ptr %c0]))
  (op %operand1_ptr (: !llvm.ptr) (mlsp.get_element [%args_ptr %c1]))
  (op %operand2_ptr (: !llvm.ptr) (mlsp.get_element [%args_ptr %c2]))

  ;; Extract type from (: type) - get element at index 1
  (op %result_type_ptr (: !llvm.ptr) (mlsp.get_element [%type_expr_ptr %c1]))

  ;; ========== GET STRING ADDRESSES ==========

  (op %str_operation_ptr (: !llvm.ptr) (mlsp.string_const {:global @str_operation}))
  (op %str_name_ptr (: !llvm.ptr) (mlsp.string_const {:global @str_name}))
  (op %str_arith_addi_ptr (: !llvm.ptr) (mlsp.string_const {:global @str_arith_addi}))
  (op %str_result_types_ptr (: !llvm.ptr) (mlsp.string_const {:global @str_result_types}))
  (op %str_operands_ptr (: !llvm.ptr) (mlsp.string_const {:global @str_operands}))

  ;; ========== CREATE IDENTIFIERS ==========
  ;; Before (LLVM): 5 × 20 lines = 100 lines of boilerplate
  ;; Before (helpers): 5 function calls
  ;; After (MLSP): 5 one-liners with high-level ops!

  (op %operation_id (: !llvm.ptr) (mlsp.identifier [%str_operation_ptr]))
  (op %name_id (: !llvm.ptr) (mlsp.identifier [%str_name_ptr]))
  (op %addi_id (: !llvm.ptr) (mlsp.identifier [%str_arith_addi_ptr]))
  (op %result_types_id (: !llvm.ptr) (mlsp.identifier [%str_result_types_ptr]))
  (op %operands_id (: !llvm.ptr) (mlsp.identifier [%str_operands_ptr]))

  ;; ========== BUILD LISTS ==========
  ;; Before: ~88 lines of array allocation + stores + list creation
  ;; After: 4 one-liners!

  ;; (name arith.addi)
  (op %name_list (: !llvm.ptr) (mlsp.list [%name_id %addi_id]))

  ;; (result-types type)
  (op %types_list (: !llvm.ptr) (mlsp.list [%result_types_id %result_type_ptr]))

  ;; (operands op1 op2)
  (op %operands_list (: !llvm.ptr) (mlsp.list [%operands_id %operand1_ptr %operand2_ptr]))

  ;; (operation ...)
  (op %result (: !llvm.ptr) (mlsp.list [%operation_id %name_list %types_list %operands_list]))

  (return %result))
