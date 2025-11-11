;; Fibonacci with MORE terse operations
;; Let's see how much we can actually convert!

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

          ;; ✅ TERSE: Check if n <= 1
          (declare c1 (arith.constant {:value (: 1 i32)}))

          ;; arith.cmpi - needs explicit type (returns i1)
          (operation
            (name arith.cmpi)
            (result-bindings [%cond])
            (result-types i1)
            (operands %n %c1)
            (attributes { :predicate (: 3 i64) }))

          ;; scf.if - no terse syntax yet
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

                  ;; ✅ TERSE: All constants and arithmetic!
                  (declare c1_rec (arith.constant {:value (: 1 i32)}))
                  (declare n_minus_1 (arith.subi %n %c1_rec))

                  ;; func.call - needs explicit type
                  (operation
                    (name func.call)
                    (result-bindings [%fib_n_minus_1])
                    (result-types i32)
                    (operands %n_minus_1)
                    (attributes { :callee @fibonacci }))

                  ;; ✅ TERSE: More constants and arithmetic!
                  (declare c2 (arith.constant {:value (: 2 i32)}))
                  (declare n_minus_2 (arith.subi %n %c2))

                  ;; func.call - needs explicit type
                  (operation
                    (name func.call)
                    (result-bindings [%fib_n_minus_2])
                    (result-types i32)
                    (operands %n_minus_2)
                    (attributes { :callee @fibonacci }))

                  ;; ✅ TERSE: Addition!
                  (declare sum (arith.addi %fib_n_minus_1 %fib_n_minus_2))

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

          ;; ✅ TERSE
          (declare n (arith.constant {:value (: 10 i32)}))

          ;; func.call - needs explicit type
          (operation
            (name func.call)
            (result-bindings [%fib_result])
            (result-types i32)
            (operands %n)
            (attributes { :callee @fibonacci }))

          ;; arith.extsi - needs explicit target type
          (operation
            (name arith.extsi)
            (result-bindings [%result_i64])
            (result-types i64)
            (operands %fib_result))

          (operation
            (name func.return)
            (operands %result_i64)))))))
