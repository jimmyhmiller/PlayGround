;; Test 3: Fibonacci function
;; Expected: fib(10) = 55

;; fibonacci function
(op "func.func" [] []
  {sym_name "fib" function_type (-> [i32] [i32])}
  [[(block [[arg0 i32]] [
    (op "arith.constant" [i32] [] {value (1 i32)} [])
    (op "arith.cmpi" [i1] [0 1] {predicate "sle"} [])
    (op "scf.if" [i32] [2] {} [
      [(block [] [
        (op "scf.yield" [] [0] {} [])
      ])]
      [(block [] [
        (op "arith.constant" [i32] [] {value (1 i32)} [])
        (op "arith.subi" [i32] [0 3] {} [])
        (op "arith.constant" [i32] [] {value (2 i32)} [])
        (op "arith.subi" [i32] [0 5] {} [])
        (op "func.call" [i32] [4] {callee "@fib"} [])
        (op "func.call" [i32] [6] {callee "@fib"} [])
        (op "arith.addi" [i32] [7 8] {} [])
        (op "scf.yield" [] [9] {} [])
      ])]
    ])
    (op "func.return" [] [10] {} [])
  ])])])
