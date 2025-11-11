;; Test if arith.subi works with terse syntax

(mlir
  (operation
    (name func.func)
    (attributes {
      :sym_name @main
      :function_type (!function (inputs) (results i64))
    })
    (regions
      (region
        (block
          (arguments [])

          ;; Test subi with terse syntax
          (declare c10 (arith.constant {:value (: 10 i32)}))
          (declare c3 (arith.constant {:value (: 3 i32)}))
          (declare result (arith.subi %c10 %c3))

          ;; Convert to i64 and return
          (operation
            (name arith.extsi)
            (result-bindings [%result_i64])
            (result-types i64)
            (operands %result))

          (operation
            (name func.return)
            (operands %result_i64)))))))
