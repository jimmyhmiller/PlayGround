;; ========================================
;; MLSP Dialect Basic Test
;; ========================================
;;
;; Tests that mlsp dialect loads and basic operations work.
;; This uses placeholder lowerings until full LLVM transforms are complete.

;; The dialect definition is auto-loaded from mlsp_dialect.lisp

;; ========================================
;; Test Application
;; ========================================

(defn main [] i64
  ;; Test mlsp.identifier - creates identifier atom
  (op %name (: !llvm.ptr) (mlsp.identifier {:value @str_test}))

  ;; Test mlsp.list - creates list from elements
  (op %elem1 (: !llvm.ptr) (mlsp.identifier {:value @str_a}))
  (op %elem2 (: !llvm.ptr) (mlsp.identifier {:value @str_b}))
  (op %list (: !llvm.ptr) (mlsp.list [%elem1 %elem2]))

  ;; Test mlsp.get_element - extract from list
  (constant %idx (: 0 i64))
  (op %extracted (: !llvm.ptr) (mlsp.get_element [%list %idx]))

  ;; Return success
  (constant %result (: 42 i64))
  (return %result))

;; String constants for testing
(operation
  (name llvm.mlir.global)
  (attributes {:sym_name @str_test
               :value "test\00"
               :global_type !llvm.array<5 x i8>
               :linkage #llvm.linkage<internal>
               :constant true
               :unnamed_addr (: 2 i64)
               :addr_space (: 0 i32)
               :alignment (: 1 i64)})
  (regions (region)))

(operation
  (name llvm.mlir.global)
  (attributes {:sym_name @str_a
               :value "a\00"
               :global_type !llvm.array<2 x i8>
               :linkage #llvm.linkage<internal>
               :constant true
               :unnamed_addr (: 2 i64)
               :addr_space (: 0 i32)
               :alignment (: 1 i64)})
  (regions (region)))

(operation
  (name llvm.mlir.global)
  (attributes {:sym_name @str_b
               :value "b\00"
               :global_type !llvm.array<2 x i8>
               :linkage #llvm.linkage<internal>
               :constant true
               :unnamed_addr (: 2 i64)
               :addr_space (: 0 i32)
               :alignment (: 1 i64)})
  (regions (region)))
