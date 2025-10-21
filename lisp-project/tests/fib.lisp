;; Test 3: Fibonacci function
;; Expected: fib(10) = 55

;; fibonacci function
(op "func.func" [] []
  {sym_name "fib" function_type (-> [i32] [i32])}
  [[(block [[%arg0 i32]] [
    (op "arith.constant" [%0 i32] [] {value (1 i32)} [])
    (op "arith.cmpi" [%1 i1] [%arg0 %0] {predicate "sle"} [])
    (op "scf.if" [%2 i32] [%1] {} [
      [(block [] [
        (op "scf.yield" [] [%0] {} [])
      ])]
      [(block [] [
        (op "arith.constant" [%3 i32] [] {value (1 i32)} [])
        (op "arith.subi" [%4 i32] [%arg0 %3] {} [])
        (op "arith.constant" [%5 i32] [] {value (2 i32)} [])
        (op "arith.subi" [%6 i32] [%arg0 %5] {} [])
        (op "func.call" [%7 i32] [%4] {callee "@fib"} [])
        (op "func.call" [%8 i32] [%6] {callee "@fib"} [])
        (op "arith.addi" [%9 i32] [%7 %8] {} [])
        (op "scf.yield" [] [%9] {} [])
      ])]
    ])
    (op "func.return" [] [%2] {} [])
  ])])])
