;; ========================================
;; MLSP DIALECT: High-Level Macro Building
;; ========================================
;;
;; Application file with IRDL dialect definition and macro implementation

;; ========================================
;; PART 1: IRDL Dialect Definition
;; ========================================

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
                  (operand-uses %ptr_type))))))))))

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
  (: %index i64)
] !llvm.ptr

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
  (: %str_len i64)
] !llvm.ptr
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
  (: %elem1 !llvm.ptr)
] !llvm.ptr
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
  (: %elem2 !llvm.ptr)
] !llvm.ptr
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
  (: %elem3 !llvm.ptr)
] !llvm.ptr
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

  ;; ========== GET STRING ADDRESSES ==========
  (op %str_operation_ptr (: !llvm.ptr) (mlsp.string_const {:global @str_operation}))
  (op %str_name_ptr (: !llvm.ptr) (mlsp.string_const {:global @str_name}))
  (op %str_arith_addi_ptr (: !llvm.ptr) (mlsp.string_const {:global @str_arith_addi}))
  (op %str_result_types_ptr (: !llvm.ptr) (mlsp.string_const {:global @str_result_types}))
  (op %str_operands_ptr (: !llvm.ptr) (mlsp.string_const {:global @str_operands}))

  ;; ========== CREATE IDENTIFIERS ==========
  (op %operation_id (: !llvm.ptr) (mlsp.identifier [%str_operation_ptr]))
  (op %name_id (: !llvm.ptr) (mlsp.identifier [%str_name_ptr]))
  (op %addi_id (: !llvm.ptr) (mlsp.identifier [%str_arith_addi_ptr]))
  (op %result_types_id (: !llvm.ptr) (mlsp.identifier [%str_result_types_ptr]))
  (op %operands_id (: !llvm.ptr) (mlsp.identifier [%str_operands_ptr]))

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
