;; Test 2: Simple arithmetic (40 + 2)
;; Expected: Returns 42


(op "func.func" [] [] {sym_name "main" function_type (-> [] [i32])} [
  [(block [] [
    (op "arith.constant" [i32] [] {value (40 i32)} [])
    (op "arith.constant" [i32] [] {value (2 i32)} [])
    (op "arith.addi" [i32] [0 1] {} [])
    (op "func.return" [] [2] {} [])
  ])]
])
