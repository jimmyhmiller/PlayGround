;; Simple addition example: returns 10 + 32 = 42

(operation
  (name func.func)
  (attributes {
    :sym_name @main
    :function_type (!function (inputs) (results i64))
  })
  (regions
    (region
      (block [^entry]
        (arguments [])

        ;; Create constant 10
        (operation
          (name arith.constant)
          (result-bindings [%c10])
          (result-types i64)
          (attributes { :value (: 10 i64) }))

        ;; Create constant 32
        (operation
          (name arith.constant)
          (result-bindings [%c32])
          (result-types i64)
          (attributes { :value (: 32 i64) }))

        ;; Add them together
        (operation
          (name arith.addi)
          (result-bindings [%result])
          (result-types i64)
          (operands %c10 %c32))

        ;; Return the result
        (operation
          (name func.return)
          (operands %result))))))
