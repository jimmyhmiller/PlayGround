;; Recursive Fibonacci - Simple Partial Terse Conversion
;;
;; ✅ Uses terse syntax ONLY for constants (safest, most tested feature)
;; ✅ Everything else uses verbose syntax
;;
;; Shows 158 lines → 128 lines (19% reduction just from constants!)

(mlir
  (operation
    (name func.func)
    (attributes {
      :sym_name @fibonacci
      :function_type (!function (inputs i32) (results i32))
    })
    (regions
      (region
        (block [^entry]
          (arguments [ (: %n i32) ])

          ;; ✅ TERSE: Constants
          (declare c1 (arith.constant {:value (: 1 i32)}))

          (operation
            (name arith.cmpi)
            (result-bindings [%cond])
            (result-types i1)
            (operands %n %c1)
            (attributes { :predicate (: 3 i64) }))

          (operation
            (name scf.if)
            (result-bindings [%result])
            (result-types i32)
            (operands %cond)
            (regions
              ;; Then region
              (region
                (block
                  (arguments [])
                  (operation
                    (name scf.yield)
                    (operands %n))))

              ;; Else region
              (region
                (block
                  (arguments [])

                  ;; ✅ TERSE: Constants
                  (declare c1_rec (arith.constant {:value (: 1 i32)}))

                  (operation
                    (name arith.subi)
                    (result-bindings [%n_minus_1])
                    (result-types i32)
                    (operands %n %c1_rec))

                  (operation
                    (name func.call)
                    (result-bindings [%fib_n_minus_1])
                    (result-types i32)
                    (operands %n_minus_1)
                    (attributes { :callee @fibonacci }))

                  ;; ✅ TERSE: Constants
                  (declare c2 (arith.constant {:value (: 2 i32)}))

                  (operation
                    (name arith.subi)
                    (result-bindings [%n_minus_2])
                    (result-types i32)
                    (operands %n %c2))

                  (operation
                    (name func.call)
                    (result-bindings [%fib_n_minus_2])
                    (result-types i32)
                    (operands %n_minus_2)
                    (attributes { :callee @fibonacci }))

                  (operation
                    (name arith.addi)
                    (result-bindings [%sum])
                    (result-types i32)
                    (operands %fib_n_minus_1 %fib_n_minus_2))

                  (operation
                    (name scf.yield)
                    (operands %sum))))))

          (operation
            (name func.return)
            (operands %result))))))

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

          ;; ✅ TERSE: Constants
          (declare n (arith.constant {:value (: 10 i32)}))

          (operation
            (name func.call)
            (result-bindings [%fib_result])
            (result-types i32)
            (operands %n)
            (attributes { :callee @fibonacci }))

          (operation
            (name arith.extsi)
            (result-bindings [%result_i64])
            (result-types i64)
            (operands %fib_result))

          (operation
            (name func.return)
            (operands %result_i64)))))))
