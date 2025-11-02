;; Example demonstrating WAST-style nested operations
;; This computes: (5 + 3) * 2 = 16
;; Operations are nested directly without intermediate bindings

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

        ;; Return (5 + 3) * 2 using nested operations
        ;; The flattener will automatically generate SSA bindings
        (operation
          (name func.return)
          (operands
            ;; Multiply: (5 + 3) * 2
            (operation
              (name arith.muli)
              (result-types i64)
              (operands
                ;; Add: 5 + 3
                (operation
                  (name arith.addi)
                  (result-types i64)
                  (operands
                    (operation
                      (name arith.constant)
                      (result-types i64)
                      (attributes { :value (: 5 i64) }))
                    (operation
                      (name arith.constant)
                      (result-types i64)
                      (attributes { :value (: 3 i64) }))))
                ;; Constant: 2
                (operation
                  (name arith.constant)
                  (result-types i64)
                  (attributes { :value (: 2 i64) }))))))))))
