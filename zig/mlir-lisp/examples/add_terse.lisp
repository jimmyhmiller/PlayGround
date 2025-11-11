;; Simple addition example: returns 10 + 32 = 42
;; Using terse operation syntax

(operation
  (name func.func)
  (attributes {}
    :sym_name @main
    :function_type (!function (inputs) (results i64)))

  (regions
    (region
      (block [^entry]
        (arguments [])

        ;; Create constants using terse syntax - type inferred from :value attribute
        (declare c10 (arith.constant {:value (: 10 i64)}))
        (declare c32 (arith.constant {:value (: 32 i64)}))

        ;; Add them together - result type inferred from operands
        (declare result (arith.addi %c10 %c32))

        ;; Return the result
        (func.return %result)))))
