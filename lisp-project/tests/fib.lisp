;; Test 3: Fibonacci function
;; Expected: fib(10) = 55

;; fibonacci function
(op "func.func" [] []
  {"sym_name" "\"fib\"" "function_type" "(i32) -> i32"}
  [[(block [["arg0" "i32"]] [
    (op "arith.constant" ["i32"] [] {"value" "1 : i32"} [])
    (op "arith.cmpi" ["i1"] ["arg0" "0"] {"predicate" "sle"} [])
    (op "scf.if" ["i32"] ["1"] {} [
      [(block [] [
        (op "scf.yield" [] ["arg0"] {} [])
      ])]
      [(block [] [
        (op "arith.constant" ["i32"] [] {"value" "1 : i32"} [])
        (op "arith.subi" ["i32"] ["arg0" "2"] {} [])
        (op "arith.constant" ["i32"] [] {"value" "2 : i32"} [])
        (op "arith.subi" ["i32"] ["arg0" "4"] {} [])
        (op "func.call" ["i32"] ["3"] {"callee" "@fib"} [])
        (op "func.call" ["i32"] ["5"] {"callee" "@fib"} [])
        (op "arith.addi" ["i32"] ["6" "7"] {} [])
        (op "scf.yield" [] ["8"] {} [])
      ])]
    ])
    (op "func.return" [] ["9"] {} [])
  ])]])
