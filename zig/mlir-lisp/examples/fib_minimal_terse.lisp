;; Minimal fibonacci test with terse operations

(mlir
  ;; Simple non-recursive version: just returns n - 1
  (operation
    (name func.func)
    (attributes {
      :sym_name @simple_fib
      :function_type (!function (inputs i32) (results i32))
    })
    (regions
      (region
        (block
          (arguments [ (: %n i32) ])

          ;; Use terse arithmetic with function argument
          (declare c1 (arith.constant {:value (: 1 i32)}))
          (declare result (arith.subi %n %c1))

          (operation
            (name func.return)
            (operands %result))))))

  ;; Main function
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

          (declare n (arith.constant {:value (: 5 i32)}))

          ;; Call function
          (operation
            (name func.call)
            (result-bindings [%fib_result])
            (result-types i32)
            (operands %n)
            (attributes { :callee @simple_fib }))

          ;; Convert and return
          (operation
            (name arith.extsi)
            (result-bindings [%result_i64])
            (result-types i64)
            (operands %fib_result))

          (operation
            (name func.return)
            (operands %result_i64)))))))
