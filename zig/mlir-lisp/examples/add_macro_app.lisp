;; Application using the + macro
;; This should expand to arith.addi, which we'll transform

(operation
  (name func.func)
  (attributes {:sym_name @main :function_type (!function (inputs) (results i64))})
  (regions
    (region
      (block [^entry]
        (arguments [])
        ;; Use the + macro - this will expand to arith.addi
        ;; Result: 10 + 32 = 42
        (return
          (* (: i64)
            (+ (: i64)
              (operation
                (name arith.constant)
                (result-types i64)
                (attributes {:value (: 10 i64)}))
              (operation
                (name arith.constant)
                (result-types i64)
                (attributes {:value (: 32 i64)})))
            (operation
              (name arith.constant)
              (result-types i64)
              (attributes {:value (: 1 i64)}))))))))
