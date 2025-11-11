(mlir
  (operation
    (name func.func)
    (attributes {
      :sym_name @main
      :function_type (!function (inputs) (results i32))
    })
    (regions
      (region
        (block
          (arguments [])

          (declare x (arith.constant {:value (: 42 i32)}))

          (func.return %x))))))
