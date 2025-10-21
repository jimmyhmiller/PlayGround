;; Test: Simple function with one argument
;; Expected: identity function

(op "func.func" [] []
  {sym_name "test" function_type (-> [i32] [i32])}
  [[(block [[%arg0 i32]] [
    (op "func.return" [] [%arg0] {} [])
  ])]])
