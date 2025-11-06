;; ========================================
;; MLSP DIALECT DEFINITION ONLY
;; ========================================
;; Test file with just the IRDL dialect definition, no transforms

(operation
  (name irdl.dialect)
  (attributes {:sym_name @mlsp})
  (regions
    (region
      (block
        ;; mlsp.identifier
        (operation
          (name irdl.operation)
          (attributes {:sym_name @identifier})
          (regions
            (region
              (block
                ;; Accept any attribute for value (symbol ref or string)
                (op %any_attr (: !irdl.attribute) (irdl.any []))
                (operation
                  (name irdl.attributes)
                  (attributes {:attributeValueNames ["value"]})
                  (operand-uses %any_attr))
                (op %ptr_type (: !irdl.attribute) (irdl.is {:expected !llvm.ptr} []))
                (operation
                  (name irdl.results)
                  (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>})
                  (operand-uses %ptr_type))))))

        ;; mlsp.list
        (operation
          (name irdl.operation)
          (attributes {:sym_name @list})
          (regions
            (region
              (block
                (op %ptr_type (: !irdl.attribute) (irdl.is {:expected !llvm.ptr} []))
                (operation
                  (name irdl.operands)
                  (attributes {:names ["elements"] :variadicity #irdl<variadicity_array[ variadic]>})
                  (operand-uses %ptr_type))
                (operation
                  (name irdl.results)
                  (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>})
                  (operand-uses %ptr_type))))))

        ;; mlsp.get_element
        (operation
          (name irdl.operation)
          (attributes {:sym_name @get_element})
          (regions
            (region
              (block
                ;; Define a single variadic operand that accepts both ptr and i64
                (op %ptr_type (: !irdl.attribute) (irdl.is {:expected !llvm.ptr} []))
                (op %i64_type (: !irdl.attribute) (irdl.is {:expected i64} []))
                (op %any_type (: !irdl.attribute) (irdl.any_of [%ptr_type %i64_type]))
                (operation
                  (name irdl.operands)
                  (attributes {:names ["inputs"] :variadicity #irdl<variadicity_array[ variadic]>})
                  (operand-uses %any_type))
                (operation
                  (name irdl.results)
                  (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>})
                  (operand-uses %ptr_type))))))))))

;; Test application
(defn main [] i64
  ;; Use mlsp operations (will not lower yet, just verify dialect works)
  (op %name (: !llvm.ptr) (mlsp.identifier {:value @str_test}))
  (op %elem1 (: !llvm.ptr) (mlsp.identifier {:value @str_a}))
  (op %elem2 (: !llvm.ptr) (mlsp.identifier {:value @str_b}))
  (op %list (: !llvm.ptr) (mlsp.list [%elem1 %elem2]))
  ;; NOTE: mlsp.get_element currently not working due to IRDL multi-operand limitations
  ;; (constant %idx (: 0 i64))
  ;; (op %extracted (: !llvm.ptr) (mlsp.get_element [%list %idx]))
  (constant %result (: 42 i64))
  (return %result))

;; Strings
(operation
  (name llvm.mlir.global)
  (attributes {:sym_name @str_test :value "test\00"
               :global_type !llvm.array<5 x i8>
               :linkage #llvm.linkage<internal> :constant true
               :unnamed_addr (: 2 i64) :addr_space (: 0 i32) :alignment (: 1 i64)})
  (regions (region)))

(operation
  (name llvm.mlir.global)
  (attributes {:sym_name @str_a :value "a\00"
               :global_type !llvm.array<2 x i8>
               :linkage #llvm.linkage<internal> :constant true
               :unnamed_addr (: 2 i64) :addr_space (: 0 i32) :alignment (: 1 i64)})
  (regions (region)))

(operation
  (name llvm.mlir.global)
  (attributes {:sym_name @str_b :value "b\00"
               :global_type !llvm.array<2 x i8>
               :linkage #llvm.linkage<internal> :constant true
               :unnamed_addr (: 2 i64) :addr_space (: 0 i32) :alignment (: 1 i64)})
  (regions (region)))
