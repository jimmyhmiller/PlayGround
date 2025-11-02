;; Example demonstrating + and * arithmetic macros
;; This computes: (5 + 3) * 2 = 16
;; Using the new + and * macros with type annotations

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

        ;; Using +, *, and return macros with nested expressions
        ;; Syntax: (+ (: type) operand1 operand2)
        ;; Syntax: (* (: type) operand1 operand2)
        ;; Syntax: (return operand)
        (return
          ;; (5 + 3) * 2
          (* (: i64)
            ;; 5 + 3
            (+ (: i64)
              (operation
                (name arith.constant)
                (result-types i64)
                (attributes { :value (: 5 i64) }))
              (operation
                (name arith.constant)
                (result-types i64)
                (attributes { :value (: 3 i64) })))
            ;; * 2
            (operation
              (name arith.constant)
              (result-types i64)
              (attributes { :value (: 2 i64) }))))))))
