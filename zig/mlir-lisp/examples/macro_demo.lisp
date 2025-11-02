;; Macro demo: Shows +, *, and return macros working together
;; Computes: (10 + 32) * 2 = 84

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

        ;; Beautiful nested expression using all our macros!
        ;; (return (* (: i64) (+ (: i64) 10 32) 2))
        (return
          (* (: i64)
            (+ (: i64)
              (operation
                (name arith.constant)
                (result-types i64)
                (attributes { :value (: 10 i64) }))
              (operation
                (name arith.constant)
                (result-types i64)
                (attributes { :value (: 32 i64) })))
            (operation
              (name arith.constant)
              (result-types i64)
              (attributes { :value (: 2 i64) }))))))))
