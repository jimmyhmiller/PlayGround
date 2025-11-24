; Arithmetic operations example

(require-dialect [arith :as a] [func :as f])

(module
  (do
    ; Add two numbers
    (f/func {:sym_name "add"
             :function_type (-> [i64 i64] [i64])}
      (do
        (block [(: x i64) (: y i64)]
          (def result (a/addi x y))
          (f/return result))))

    ; Multiply two numbers
    (f/func {:sym_name "mul"
             :function_type (-> [i64 i64] [i64])}
      (do
        (block [(: x i64) (: y i64)]
          (def result (a/muli x y))
          (f/return result))))

    ; Compute (a + b) * c
    (f/func {:sym_name "add_then_mul"
             :function_type (-> [i64 i64 i64] [i64])}
      (do
        (block [(: a i64) (: b i64) (: c i64)]
          (def sum (a/addi a b))
          (def result (a/muli sum c))
          (f/return result))))

    ; Main function
    (f/func {:sym_name "main"
             :function_type (-> [] [i64])}
      (do
        (block []
          (def x (f/call "add" 10 20))
          (def y (f/call "mul" x 2))
          (def z (f/call "add_then_mul" 5 7 3))
          (f/return z))))))
