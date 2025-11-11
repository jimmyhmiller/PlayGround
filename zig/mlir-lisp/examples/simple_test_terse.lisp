(operation
  (name func.func)
  (attributes {}
    :sym_name @main
    :function_type (!function (inputs) (results i64)))

  (regions
    (region
      (block
        (arguments [])
        (declare result (arith.constant {:value (: 42 i64)}))
        (func.return %result)))))
