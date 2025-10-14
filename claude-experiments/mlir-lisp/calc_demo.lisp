;; ============================================================================
;; Complete IRDL + Transform Dialect Demo
;; ============================================================================
;; This demonstrates the full workflow:
;; 1. Define a custom dialect using IRDL
;; 2. Define lowering patterns using PDL
;; 3. Write code using the custom dialect
;; 4. JIT execute (applies transforms automatically)

;; Step 1: Define the calc dialect using IRDL
(defirdl-dialect calc
  :namespace "calc"
  :description "Calculator dialect"

  (defirdl-op add
    :summary "Addition operation"
    :operands [(lhs i32) (rhs i32)]
    :results [(result i32)])

  (defirdl-op mul
    :summary "Multiplication operation"
    :operands [(lhs i32) (rhs i32)]
    :results [(result i32)]))

;; Step 2: Define PDL patterns to lower calc -> arith
(defpdl-pattern calc_add_to_arith
  :benefit 1
  :match (calc.add $x $y)
  :rewrite (arith.addi $x $y))

(defpdl-pattern calc_mul_to_arith
  :benefit 1
  :match (calc.mul $x $y)
  :rewrite (arith.muli $x $y))

;; Step 3: Write a function using the custom calc dialect
(defn compute [] i32
  (calc.add (calc.mul 10 20) 30))

;; Step 4: Execute! The transform will be applied automatically
(jit-execute compute)
