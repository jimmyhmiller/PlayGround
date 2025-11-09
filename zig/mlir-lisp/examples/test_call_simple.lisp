(operation
  (name func.func)
  (attributes {}
    :sym_name @get_number
    :function_type (!function (inputs) (results i64)))
  
  (regions
    (region
      (block
        (arguments [])
        (operation
          (name arith.constant)
          (result-bindings [%num])
          (result-types i64)
          (attributes { :value (: 99 i64)}))
        (operation
          (name func.return)
          (operands %num))))))

(operation
  (name func.func)
  (attributes {}
    :sym_name @main
    :function_type (!function (inputs) (results i64)))
  
  (regions
    (region
      (block
        (arguments [])
        (call @get_number i64)
        (operation
          (name func.return)
          (operands %result0))))))
