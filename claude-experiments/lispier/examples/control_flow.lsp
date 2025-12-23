; Control flow example with branches and loops
; Note: In MLIR, values must be passed between blocks via block arguments

(require-dialect [cf :as c] [arith :as a] [func :as f])

(module
  (do
    ; Max function using conditional branch
    ; Each successor block receives its value as a block argument
    (f/func {:sym_name "max"
             :function_type (-> [i64 i64] [i64])}
      (region
        (block [(: x i64) (: y i64)]
          (def cond (a/cmpi {:predicate "sgt"} x y))
          ; Pass x to return_x block, y to return_y block
          ; operandSegmentSizes: [condition, trueDestOperands, falseDestOperands]
          (c/cond_br {:operandSegmentSizes array<i32: 1, 1, 1>} cond ^return_x x ^return_y y))

        (block ^return_x [(: val i64)]
          (f/return val))

        (block ^return_y [(: val i64)]
          (f/return val))))

    ; Simple function that just returns a constant
    (f/func {:sym_name "get_zero"
             :function_type (-> [] [i64])}
      (region
        (block []
          (f/return (: 0 i64)))))))