;; ============================================================================
;; FULL WORKFLOW: Dialect → Transform → Program → Execute
;; ============================================================================
;; Run: mlir-lisp examples/full_workflow.lisp

;; ============================================================================
;; STEP 1: Define Source Dialect
;; ============================================================================

(defirdl-dialect calc
  :namespace "calc"
  :description "Simple calculator dialect"

  (defirdl-op constant
    :summary "Integer constant"
    :attributes [(value IntegerAttr)]
    :results [(result I32)]
    :traits [Pure NoMemoryEffect])

  (defirdl-op add
    :summary "Addition"
    :operands [(lhs I32) (rhs I32)]
    :results [(result I32)]
    :traits [Pure Commutative NoMemoryEffect])

  (defirdl-op mul
    :summary "Multiplication"
    :operands [(lhs I32) (rhs I32)]
    :results [(result I32)]
    :traits [Pure Commutative NoMemoryEffect]))

;; ============================================================================
;; STEP 2: Define Transform Patterns (using PDL and Transform dialects)
;; ============================================================================

;; These patterns will be emitted as MLIR operations and executed by the
;; transform interpreter. No special Rust code needed!

(defpdl-pattern lower-calc-constant
  :benefit 1
  :description "Lower calc.constant to arith.constant"
  :match
  (let [val (pdl.attribute)
        type (pdl.type)
        op (pdl.operation "calc.constant" :attributes {:value val} :results [type])]
    op)
  :rewrite
  (pdl.operation "arith.constant" :attributes {:value val} :results [type]))

(defpdl-pattern lower-calc-add
  :benefit 1
  :description "Lower calc.add to arith.addi"
  :match
  (let [lhs (pdl.operand)
        rhs (pdl.operand)
        type (pdl.type)
        op (pdl.operation "calc.add" :operands [lhs rhs] :results [type])]
    op)
  :rewrite
  (pdl.operation "arith.addi" :operands [lhs rhs] :results [type]))

(defpdl-pattern lower-calc-mul
  :benefit 1
  :description "Lower calc.mul to arith.muli"
  :match
  (let [lhs (pdl.operand)
        rhs (pdl.operand)
        type (pdl.type)
        op (pdl.operation "calc.mul" :operands [lhs rhs] :results [type])]
    op)
  :rewrite
  (pdl.operation "arith.muli" :operands [lhs rhs] :results [type]))

;; ============================================================================
;; STEP 3: Write Program Using Custom Dialect
;; ============================================================================

;; This program uses calc.* operations
;; It will be lowered to arith.* by our transforms

(defn compute [] i32
  (calc.add
    (calc.mul
      (calc.constant 10)
      (calc.constant 20))
    (calc.constant 30)))

;; Expected result: (10 * 20) + 30 = 230

;; ============================================================================
;; STEP 4: Show What We Have
;; ============================================================================

(println "Registered dialects:" (list-dialects))
(println "Registered patterns:" (list-patterns))

;; ============================================================================
;; STEP 5: Apply Transform (when interpreter is ready)
;; ============================================================================

;; Once MLIR transform interpreter is available, this will:
;; 1. Generate transform dialect IR from our patterns
;; 2. Execute it on the program module
;; 3. Get back lowered arith.* operations
;; 4. Continue to LLVM lowering
;; 5. JIT execute
;;
;; (apply-transform "calc-lowering" "compute")
;; => 230
