;; ========================================
;; MLSP DIALECT: High-Level Macro Building
;; ========================================
;;
;; This file defines the mlsp (MLIR-Lisp) dialect for building CValueLayout
;; structures using high-level operations that lower to LLVM.
;;
;; Replaces ~310 lines of malloc/GEP/store boilerplate with ~15 lines of
;; clean, declarative operations.

;; ========================================
;; PART 1: IRDL Dialect Definition
;; ========================================

(operation
  (name builtin.module)
  (attributes {:metadata unit})
  (regions
    (region
      (block))))
(operation
  (name irdl.dialect)
  (attributes {:sym_name @mlsp})
  (regions
    (region
      (block
        ;; =====================================
        ;; mlsp.identifier - Create identifier atom
        ;; =====================================
        ;; Syntax: %val = mlsp.identifier(%str_ptr)
        ;; Creates CValueLayout with identifier type tag and string data
        (operation
          (name irdl.operation)
          (attributes {:sym_name @identifier})
          (regions
            (region
              (block
                ;; Operand: string pointer (runtime value)
                (op %ptr_type (: !irdl.attribute) (irdl.is {:expected !llvm.ptr} []))
                (operation
                  (name irdl.operands)
                  (attributes {:names ["value"] :variadicity #irdl<variadicity_array[ single]>})
                  (operand-uses %ptr_type))

                ;; Result: !llvm.ptr (CValueLayout*)
                (operation
                  (name irdl.results)
                  (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>})
                  (operand-uses %ptr_type))))))

        ;; =====================================
        ;; mlsp.number - Create number atom
        ;; =====================================
        ;; Syntax: %val = mlsp.number(%str_ptr)
        ;; Creates CValueLayout with number type tag and string data
        (operation
          (name irdl.operation)
          (attributes {:sym_name @number})
          (regions
            (region
              (block
                ;; Operand: string pointer (runtime value)
                (op %ptr_type (: !irdl.attribute) (irdl.is {:expected !llvm.ptr} []))
                (operation
                  (name irdl.operands)
                  (attributes {:names ["value"] :variadicity #irdl<variadicity_array[ single]>})
                  (operand-uses %ptr_type))

                ;; Result: !llvm.ptr (CValueLayout*)
                (operation
                  (name irdl.results)
                  (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>})
                  (operand-uses %ptr_type))))))

        ;; =====================================
        ;; mlsp.string - Create string atom
        ;; =====================================
        ;; Syntax: %val = mlsp.string(%str_ptr)
        ;; Creates CValueLayout with string type tag and string data
        (operation
          (name irdl.operation)
          (attributes {:sym_name @string})
          (regions
            (region
              (block
                ;; Operand: string pointer (runtime value)
                (op %ptr_type (: !irdl.attribute) (irdl.is {:expected !llvm.ptr} []))
                (operation
                  (name irdl.operands)
                  (attributes {:names ["value"] :variadicity #irdl<variadicity_array[ single]>})
                  (operand-uses %ptr_type))

                ;; Result: !llvm.ptr (CValueLayout*)
                (operation
                  (name irdl.results)
                  (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>})
                  (operand-uses %ptr_type))))))

        ;; =====================================
        ;; mlsp.list - Create list from elements
        ;; =====================================
        ;; Syntax: %list = mlsp.list(%val1, %val2, %val3)
        ;; Creates CValueLayout list containing variadic children
        (operation
          (name irdl.operation)
          (attributes {:sym_name @list})
          (regions
            (region
              (block
                ;; Operands: variadic !llvm.ptr (CValueLayout* elements)
                (op %ptr_type (: !irdl.attribute) (irdl.is {:expected !llvm.ptr} []))
                (operation
                  (name irdl.operands)
                  (attributes {:names ["elements"] :variadicity #irdl<variadicity_array[ variadic]>})
                  (operand-uses %ptr_type))

                ;; Result: !llvm.ptr (CValueLayout*)
                (operation
                  (name irdl.results)
                  (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>})
                  (operand-uses %ptr_type))))))

        ;; =====================================
        ;; mlsp.vector - Create vector from elements
        ;; =====================================
        ;; Syntax: %vec = mlsp.vector(%val1, %val2)
        ;; Creates CValueLayout vector containing variadic children
        (operation
          (name irdl.operation)
          (attributes {:sym_name @vector})
          (regions
            (region
              (block
                ;; Operands: variadic !llvm.ptr (CValueLayout* elements)
                (op %ptr_type (: !irdl.attribute) (irdl.is {:expected !llvm.ptr} []))
                (operation
                  (name irdl.operands)
                  (attributes {:names ["elements"] :variadicity #irdl<variadicity_array[ variadic]>})
                  (operand-uses %ptr_type))

                ;; Result: !llvm.ptr (CValueLayout*)
                (operation
                  (name irdl.results)
                  (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>})
                  (operand-uses %ptr_type))))))

        ;; =====================================
        ;; mlsp.get_element - Extract list element by index
        ;; =====================================
        ;; Syntax: %elem = mlsp.get_element %list, %index
        ;; Loads element from CValueLayout list at given index
        (operation
          (name irdl.operation)
          (attributes {:sym_name @get_element})
          (regions
            (region
              (block
                ;; Operand 1: list pointer
                (op %ptr_type (: !irdl.attribute) (irdl.is {:expected !llvm.ptr} []))
                (operation
                  (name irdl.operands)
                  (attributes {:names ["list"] :variadicity #irdl<variadicity_array[ single]>})
                  (operand-uses %ptr_type))

                ;; Operand 2: index value
                (op %i64_type (: !irdl.attribute) (irdl.is {:expected i64} []))
                (operation
                  (name irdl.operands)
                  (attributes {:names ["index"] :variadicity #irdl<variadicity_array[ single]>})
                  (operand-uses %i64_type))

                ;; Result: !llvm.ptr (CValueLayout*)
                (operation
                  (name irdl.results)
                  (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>})
                  (operand-uses %ptr_type))))))

        ;; =====================================
        ;; mlsp.string_const - Reference global string
        ;; =====================================
        ;; Syntax: %str = mlsp.string_const @my_string
        ;; Gets pointer to global string constant
        (operation
          (name irdl.operation)
          (attributes {:sym_name @string_const})
          (regions
            (region
              (block
                ;; Attribute: global symbol reference
                (op %sym_attr (: !irdl.attribute) (irdl.is {:expected !llvm.ptr} []))
                (operation
                  (name irdl.attributes)
                  (attributes {:attributeValueNames ["global"]})
                  (operand-uses %sym_attr))

                ;; Result: !llvm.ptr
                (op %ptr_type (: !irdl.attribute) (irdl.is {:expected !llvm.ptr} []))
                (operation
                  (name irdl.results)
                  (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>})
                  (operand-uses %ptr_type))))))

        ;; =====================================
        ;; mlsp.build_operation - High-level operation builder
        ;; =====================================
        ;; Syntax: %op = mlsp.build_operation %name, %result_types, %operands
        ;; Creates full operation CValueLayout structure with named sections
        (operation
          (name irdl.operation)
          (attributes {:sym_name @build_operation})
          (regions
            (region
              (block
                ;; Operand 1: name
                (op %ptr_type1 (: !irdl.attribute) (irdl.is {:expected !llvm.ptr} []))
                (operation
                  (name irdl.operands)
                  (attributes {:names ["name"] :variadicity #irdl<variadicity_array[ single]>})
                  (operand-uses %ptr_type1))

                ;; Operand 2: result_types
                (op %ptr_type2 (: !irdl.attribute) (irdl.is {:expected !llvm.ptr} []))
                (operation
                  (name irdl.operands)
                  (attributes {:names ["result_types"] :variadicity #irdl<variadicity_array[ single]>})
                  (operand-uses %ptr_type2))

                ;; Operand 3: operands
                (op %ptr_type3 (: !irdl.attribute) (irdl.is {:expected !llvm.ptr} []))
                (operation
                  (name irdl.operands)
                  (attributes {:names ["operands"] :variadicity #irdl<variadicity_array[ single]>})
                  (operand-uses %ptr_type3))

                ;; Result: !llvm.ptr (CValueLayout*)
                (op %ptr_result (: !irdl.attribute) (irdl.is {:expected !llvm.ptr} []))
                (operation
                  (name irdl.results)
                  (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>})
                  (operand-uses %ptr_result))))))))))

;; ========================================
;; PART 2: PDL Transform Patterns
;; ========================================
;; These patterns lower high-level mlsp operations to LLVM IR

(operation
  (name builtin.module)
  (attributes {:metadata unit})
  (regions
    (region
      (block
        (operation
          (name transform.with_pdl_patterns)
          (regions
            (region
              (block [^bb0]
                (arguments [(: %root !transform.any_op)])

                ;; =====================================
                ;; Pattern: mlsp.identifier → LLVM
                ;; =====================================
                ;; Lowers to: malloc(56) + initialize CValueLayout fields
                (operation
                  (name pdl.pattern)
                  (attributes {:benefit (: 1 i16) :sym_name @mlsp_identifier_to_llvm})
                  (regions
                    (region
                      (block
                        ;; Match mlsp.identifier with value operand
                        (operation
                          (name pdl.type)
                          (result-bindings [%ptr_type])
                          (result-types !pdl.type))
                        (operation
                          (name pdl.operand)
                          (operand-uses %ptr_type)
                          (result-bindings [%str_val])
                          (result-types !pdl.value))
                        (operation
                          (name pdl.operation)
                          (attributes {:opName "mlsp.identifier"}
                                     :operandSegmentSizes array<i32: 1, 0, 1>)
                          (operand-uses %str_val %ptr_type)
                          (result-bindings [%mlsp_op])
                          (result-types !pdl.operation))

                        ;; Rewrite to LLVM sequence
                        (operation
                          (name pdl.rewrite)
                          (attributes {:operandSegmentSizes array<i32: 1, 0>})
                          (operand-uses %mlsp_op)
                          (regions
                            (region
                              (block
                                ;; TODO: Generate full LLVM lowering:
                                ;; 1. malloc(56) for CValueLayout
                                ;; 2. GEP to type field (offset 0), store identifier_tag (0)
                                ;; 3. GEP to data_ptr (offset 8), store string pointer
                                ;; 4. GEP to data_len (offset 16), store string length
                                ;; 5. Zero remaining fields
                                ;;
                                ;; For now: placeholder - replace with call to helper function
                                (operation
                                  (name pdl.operation)
                                  (attributes {:opName "llvm.call"}
                                             :attributeValueNames ["callee"]
                                             :operandSegmentSizes array<i32: 1, 0, 1>)
                                  (operand-uses %str_val %ptr_type)
                                  (result-bindings [%result_op])
                                  (result-types !pdl.operation))

                                (operation
                                  (name pdl.replace)
                                  (attributes {:operandSegmentSizes array<i32: 1, 1, 0>})
                                  (operand-uses %mlsp_op %result_op))))))))))

                ;; =====================================
                ;; Pattern: mlsp.list → LLVM
                ;; =====================================
                ;; Lowers to: allocate array + malloc(56) + initialize list fields
                (operation
                  (name pdl.pattern)
                  (attributes {:benefit (: 1 i16) :sym_name @mlsp_list_to_llvm})
                  (regions
                    (region
                      (block
                        ;; Match mlsp.list with variadic operands
                        (operation
                          (name pdl.types)
                          (result-bindings [%elem_types])
                          (result-types !pdl.range<type>))
                        (operation
                          (name pdl.operands)
                          (operand-uses %elem_types)
                          (result-bindings [%elements])
                          (result-types !pdl.range<value>))
                        (operation
                          (name pdl.type)
                          (result-bindings [%ptr_type])
                          (result-types !pdl.type))
                        (operation
                          (name pdl.operation)
                          (attributes {:opName "mlsp.list"}
                                     :operandSegmentSizes array<i32: 1, 0, 1>)
                          (operand-uses %elements %ptr_type)
                          (result-bindings [%mlsp_op])
                          (result-types !pdl.operation))

                        ;; Rewrite to LLVM sequence
                        (operation
                          (name pdl.rewrite)
                          (attributes {:operandSegmentSizes array<i32: 1, 0>})
                          (operand-uses %mlsp_op)
                          (regions
                            (region
                              (block
                                ;; TODO: Generate full LLVM lowering:
                                ;; 1. Calculate array size (num_elements * 8)
                                ;; 2. malloc array for element pointers
                                ;; 3. Store each element pointer into array
                                ;; 4. malloc(56) for CValueLayout
                                ;; 5. Store list type tag (1)
                                ;; 6. Store array pointer
                                ;; 7. Store length
                                ;;
                                ;; For now: placeholder
                                (operation
                                  (name pdl.operation)
                                  (attributes {:opName "llvm.call"}
                                             :attributeValueNames ["callee"]
                                             :operandSegmentSizes array<i32: 1, 0, 1>)
                                  (operand-uses %elements %ptr_type)
                                  (result-bindings [%result_op])
                                  (result-types !pdl.operation))

                                (operation
                                  (name pdl.replace)
                                  (attributes {:operandSegmentSizes array<i32: 1, 1, 0>})
                                  (operand-uses %mlsp_op %result_op))))))))))

                ;; =====================================
                ;; Pattern: mlsp.get_element → LLVM
                ;; =====================================
                ;; Lowers to: GEP into data_ptr array + load
                (operation
                  (name pdl.pattern)
                  (attributes {:benefit (: 1 i16) :sym_name @mlsp_get_element_to_llvm})
                  (regions
                    (region
                      (block
                        ;; Match mlsp.get_element with list and index operands
                        (operation
                          (name pdl.type)
                          (result-bindings [%ptr_type])
                          (result-types !pdl.type))
                        (operation
                          (name pdl.type)
                          (result-bindings [%i64_type])
                          (result-types !pdl.type))
                        (operation
                          (name pdl.operand)
                          (operand-uses %ptr_type)
                          (result-bindings [%list_val])
                          (result-types !pdl.value))
                        (operation
                          (name pdl.operand)
                          (operand-uses %i64_type)
                          (result-bindings [%index_val])
                          (result-types !pdl.value))
                        (operation
                          (name pdl.operation)
                          (attributes {:opName "mlsp.get_element"}
                                     :operandSegmentSizes array<i32: 2, 0, 1>)
                          (operand-uses %list_val %index_val %ptr_type)
                          (result-bindings [%mlsp_op])
                          (result-types !pdl.operation))

                        ;; Rewrite to LLVM sequence
                        (operation
                          (name pdl.rewrite)
                          (attributes {:operandSegmentSizes array<i32: 1, 0>})
                          (operand-uses %mlsp_op)
                          (regions
                            (region
                              (block
                                ;; TODO: Generate full LLVM lowering:
                                ;; 1. GEP to data_ptr field (offset 8)
                                ;; 2. Load array pointer
                                ;; 3. GEP into array at index
                                ;; 4. Load element pointer
                                ;;
                                ;; For now: placeholder
                                (operation
                                  (name pdl.operation)
                                  (attributes {:opName "llvm.call"}
                                             :attributeValueNames ["callee"]
                                             :operandSegmentSizes array<i32: 2, 0, 1>)
                                  (operand-uses %list_val %index_val %ptr_type)
                                  (result-bindings [%result_op])
                                  (result-types !pdl.operation))

                                (operation
                                  (name pdl.replace)
                                  (attributes {:operandSegmentSizes array<i32: 1, 1, 0>})
                                  (operand-uses %mlsp_op %result_op))))))))))

                ;; Transform sequence to apply all patterns
                (operation
                  (name transform.sequence)
                  (attributes {:failure_propagation_mode (: 1 i32)}
                             :operandSegmentSizes array<i32: 1, 0>)
                  (operand-uses %root)
                  (result-types !transform.any_op)
                  (regions
                    (region
                      (block [^bb1]
                        (arguments [(: %arg1 !transform.any_op)])

                        ;; Apply mlsp.identifier lowering
                        (operation
                          (name transform.pdl_match)
                          (attributes {:pattern_name @mlsp_identifier_to_llvm})
                          (operand-uses %arg1)
                          (result-bindings [%matched1])
                          (result-types !transform.any_op))

                        ;; Apply mlsp.list lowering
                        (operation
                          (name transform.pdl_match)
                          (attributes {:pattern_name @mlsp_list_to_llvm})
                          (operand-uses %arg1)
                          (result-bindings [%matched2])
                          (result-types !transform.any_op))

                        ;; Apply mlsp.get_element lowering
                        (operation
                          (name transform.pdl_match)
                          (attributes {:pattern_name @mlsp_get_element_to_llvm})
                          (operand-uses %arg1)
                          (result-bindings [%matched3])
                          (result-types !transform.any_op))

                        (operation
                          (name transform.yield))))))))))))))
