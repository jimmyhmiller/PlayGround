;; Test 1: Simple constant return
;; Expected: Returns 42


(op "func.func" [] [] {"sym_name" "\"main\"" "function_type" "\"() -> i32\""} [
  [(block [] [
    (op "arith.constant" ["i32"] [] {"value" "\"42 : i32\""} [])
    (op "func.return" [] ["0"] {} [])
  ])]
])
