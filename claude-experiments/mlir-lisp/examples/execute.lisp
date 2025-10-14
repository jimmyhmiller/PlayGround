;; ============================================================================
;; END-TO-END EXECUTION EXAMPLE
;; ============================================================================
;; Run: mlir-lisp examples/execute.lisp
;;
;; This demonstrates the COMPLETE pipeline:
;; 1. Define dialect
;; 2. Write program using dialect
;; 3. Compile to MLIR
;; 4. Lower to LLVM
;; 5. JIT execute and get result!

;; ============================================================================
;; STEP 1: Define calc dialect (or use arith directly)
;; ============================================================================

;; For now, let's use arith directly since we need lowering to work
;; In the future, we'll apply transforms to lower calc.* → arith.*

;; ============================================================================
;; STEP 2: Write a simple program
;; ============================================================================

;; Compute: (10 * 20) + 30 = 230
(defn compute [] i32
  (+ (* 10 20) 30))

;; ============================================================================
;; STEP 3: Execute it!
;; ============================================================================

(println "\n🚀 Executing compute function...")
(jit-execute "compute" "compute")

(println "\n✅ Complete! The program was compiled, lowered to LLVM, and executed!")
