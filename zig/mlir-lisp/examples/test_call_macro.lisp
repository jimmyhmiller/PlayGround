(operation
  (name func.func)
  (attributes {}
    :sym_name @add
    :function_type (!function (inputs i64 i64) (results i64)))
  
  (regions
    (region
      (block
        (arguments [(: %arg0 i64) (: %arg1 i64)])
        (operation
          (name arith.addi)
          (result-bindings [%sum])
          (result-types i64)
          (operands %arg0 %arg1))
        (operation
          (name func.return)
          (operands %sum))))))

(operation
  (name func.func)
  (attributes {}
    :sym_name @main
    :function_type (!function (inputs) (results i64)))
  
  (regions
    (region
      (block
        (arguments [])
        (operation
          (name arith.constant)
          (result-bindings [%a])
          (result-types i64)
          (attributes { :value (: 10 i64)}))
        (operation
          (name arith.constant)
          (result-bindings [%b])
          (result-types i64)
          (attributes { :value (: 32 i64)}))
        (call @add %a %b i64)
        (operation
          (name func.return)
          (operands %result0))))))
