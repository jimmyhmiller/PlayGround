;; ============================================================================
;; REAL END-TO-END DEMO
;; ============================================================================
;; This file demonstrates the complete workflow:
;; 1. Define a dialect
;; 2. Define transforms
;; 3. Write a program
;; 4. Compile it
;; 5. Run it!

;; ============================================================================
;; STEP 1: Define the Dialect
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
;; STEP 2: Define Optimization Transforms
;; ============================================================================

(defpdl-pattern constant-fold-add
  :benefit 10
  :description "Fold (+ const const) at compile time"
  :match
  (let [val1 (pdl.attribute)
        val2 (pdl.attribute)
        c1 (pdl.operation "calc.constant" :attributes {:value val1})
        c2 (pdl.operation "calc.constant" :attributes {:value val2})
        add (pdl.operation "calc.add" :operands [c1 c2])]
    add)
  :rewrite
  (pdl.operation "calc.constant" :value (+ val1 val2)))

(defpdl-pattern constant-fold-mul
  :benefit 10
  :description "Fold (* const const) at compile time"
  :match
  (let [val1 (pdl.attribute)
        val2 (pdl.attribute)
        c1 (pdl.operation "calc.constant" :attributes {:value val1})
        c2 (pdl.operation "calc.constant" :attributes {:value val2})
        mul (pdl.operation "calc.mul" :operands [c1 c2])]
    mul)
  :rewrite
  (pdl.operation "calc.constant" :value (* val1 val2)))

;; ============================================================================
;; STEP 3: Define Lowering to Standard Dialects
;; ============================================================================

(deftransform lower-calc-to-arith
  :description "Lower calc.* to arith.*"
  (transform.sequence
    (let [constants (transform.match :ops ["calc.constant"])]
      (transform.apply-patterns :to constants
        (use-pattern lower-constant)))

    (let [adds (transform.match :ops ["calc.add"])]
      (transform.apply-patterns :to adds
        (use-pattern lower-add)))

    (let [muls (transform.match :ops ["calc.mul"])]
      (transform.apply-patterns :to muls
        (use-pattern lower-mul)))))

(defpdl-pattern lower-constant
  :match (pdl.operation "calc.constant" :attributes {:value val})
  :rewrite (pdl.operation "arith.constant" :attributes {:value val}))

(defpdl-pattern lower-add
  :match (pdl.operation "calc.add" :operands [lhs rhs])
  :rewrite (pdl.operation "arith.addi" :operands [lhs rhs]))

(defpdl-pattern lower-mul
  :match (pdl.operation "calc.mul" :operands [lhs rhs])
  :rewrite (pdl.operation "arith.muli" :operands [lhs rhs]))

;; ============================================================================
;; STEP 4: Write a Program Using Our Dialect
;; ============================================================================

;; This program computes: (10 * 20) + 30
;; Expected result: 230

(defn compute [] i32
  (calc.add
    (calc.mul
      (calc.constant 10)
      (calc.constant 20))
    (calc.constant 30)))

;; ============================================================================
;; STEP 5: Compilation Pipeline
;; ============================================================================

;; Compile with optimizations:
;; 1. Constant fold: (10 * 20) -> 200
;; 2. Constant fold: (200 + 30) -> 230
;; 3. Lower to arith dialect
;; 4. Lower to LLVM
;; 5. JIT and execute

(compile compute
  :optimize [constant-fold-mul constant-fold-add]
  :lower [lower-calc-to-arith]
  :jit true)

;; Expected output: 230
