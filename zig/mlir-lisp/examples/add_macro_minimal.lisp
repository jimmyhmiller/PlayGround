;; MINIMAL WORKING VERSION: What if we implement just the core primitives?
;;
;; This shows what the + macro would look like if we had JUST these operations:
;; - ast.list-nth : (!llvm.ptr, i64) -> !llvm.ptr
;; - ast.make-identifier : !llvm.ptr -> !llvm.ptr  (takes string ptr)
;; - ast.make-list : !llvm.ptr -> !llvm.ptr  (takes vector ptr)
;; - ast.vector-new : () -> !llvm.ptr
;; - ast.vector-push : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
;;
;; For simplicity, we'll skip validation (length checks, type checks)
;; and assume the macro is called correctly

(defn add-macro-minimal [(: %args !llvm.ptr)] !llvm.ptr

  ;; Extract the 3 arguments
  (constant %c0 (: 0 i64))
  (constant %c1 (: 1 i64))
  (constant %c2 (: 2 i64))

  ;; Get: (: type), operand1, operand2
  (op %type_expr (: !llvm.ptr) (ast.list-nth [%args %c0]))
  (op %operand1 (: !llvm.ptr) (ast.list-nth [%args %c1]))
  (op %operand2 (: !llvm.ptr) (ast.list-nth [%args %c2]))

  ;; Extract the type from (: type) - get element at index 1
  (op %result_type (: !llvm.ptr) (ast.list-nth [%type_expr %c1]))

  ;; Build: (operation (name arith.addi) (result-types type) (operands op1 op2))

  ;; Create identifiers we need
  ;; NOTE: ast.make-identifier doesn't exist yet, so we stub this out
  ;; These WOULD be: (op %operation_id (: !llvm.ptr) (ast.make-identifier ["operation"]))
  ;; For now, just create dummy constants to show the structure
  (constant %operation_id (: 0 i64))
  (constant %name_id (: 1 i64))
  (constant %arith_addi_id (: 2 i64))
  (constant %result_types_id (: 3 i64))
  (constant %operands_id (: 4 i64))

  ;; Build (name arith.addi)
  ;; NOTE: These ast.* operations don't exist yet, showing hypothetical code
  ;; (op %name_vec (: !llvm.ptr) (ast.vector-new))
  ;; (op %name_vec1 (: !llvm.ptr) (ast.vector-push [%name_vec %name_id]))
  ;; (op %name_vec2 (: !llvm.ptr) (ast.vector-push [%name_vec1 %arith_addi_id]))
  ;; (op %name_list (: !llvm.ptr) (ast.make-list [%name_vec2]))

  ;; Build (result-types type)
  ;; (op %types_vec (: !llvm.ptr) (ast.vector-new))
  ;; (op %types_vec1 (: !llvm.ptr) (ast.vector-push [%types_vec %result_types_id]))
  ;; (op %types_vec2 (: !llvm.ptr) (ast.vector-push [%types_vec1 %result_type]))
  ;; (op %types_list (: !llvm.ptr) (ast.make-list [%types_vec2]))

  ;; Build (operands operand1 operand2)
  ;; (op %operands_vec (: !llvm.ptr) (ast.vector-new))
  ;; (op %operands_vec1 (: !llvm.ptr) (ast.vector-push [%operands_vec %operands_id]))
  ;; (op %operands_vec2 (: !llvm.ptr) (ast.vector-push [%operands_vec1 %operand1]))
  ;; (op %operands_vec3 (: !llvm.ptr) (ast.vector-push [%operands_vec2 %operand2]))
  ;; (op %operands_list (: !llvm.ptr) (ast.make-list [%operands_vec3]))

  ;; Build final (operation ...)
  ;; (op %op_vec (: !llvm.ptr) (ast.vector-new))
  ;; (op %op_vec1 (: !llvm.ptr) (ast.vector-push [%op_vec %operation_id]))
  ;; (op %op_vec2 (: !llvm.ptr) (ast.vector-push [%op_vec1 %name_list]))
  ;; (op %op_vec3 (: !llvm.ptr) (ast.vector-push [%op_vec2 %types_list]))
  ;; (op %op_vec4 (: !llvm.ptr) (ast.vector-push [%op_vec3 %operands_list]))
  ;; (op %result (: !llvm.ptr) (ast.make-list [%op_vec4]))

  ;; Return a dummy value since we can't actually build the result yet
  (constant %dummy (: 0 i64))
  (return %dummy))

;; ANALYSIS:
;;
;; Lines in Zig version: ~70
;; Lines in this version: ~58
;;
;; So even with just 5 primitive operations, we're already competitive!
;;
;; The verbosity comes from:
;; 1. Immutable vectors requiring threading (vec -> vec1 -> vec2 -> ...)
;; 2. No string literals for identifiers (need ast.make-identifier)
;; 3. Manual list construction instead of quasiquote
;;
;; With quasiquote, this becomes:
;; (defmacro + [(: type) op1 op2]
;;   `(operation (name arith.addi) (result-types ,type) (operands ,op1 ,op2)))
;;
;; That's ~3 lines vs 58!
;;
;; CONCLUSION:
;; - The 5 primitive operations ARE sufficient to write macros
;; - It's just very verbose without quasiquote
;; - Quasiquote should be the priority after basic primitives work
