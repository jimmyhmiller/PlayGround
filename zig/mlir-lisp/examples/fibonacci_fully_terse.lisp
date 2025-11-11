;; Fibonacci - FULLY TERSE with explicit type annotations!
;;
;; ✅ Uses terse syntax for EVERYTHING
;; ✅ Type annotations (: expr type) for ops without inference
;; ✅ All constants and arithmetic use type inference

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
          (arguments [ (: %n i32)])

          ;; ✅ TERSE: Check if n <= 1 (with type annotation for cmpi)
          (declare c1 (arith.constant {:value (: 1 i32)}))
          (declare cond (: (arith.cmpi {:predicate (: 3 i64)} %n %c1) i1))

          ;; ✅ TERSE scf.if with regions and implicit yields!
          (declare result (:
            (scf.if %cond
              (region %n)              ;; Then: implicit (scf.yield %n)
              (region                   ;; Else: compute fib(n-1) + fib(n-2)
                ;; ✅ TERSE: Compute fib(n-1)
                (declare c1_rec (arith.constant {:value (: 1 i32)}))
                (declare n_minus_1 (arith.subi %n %c1_rec))
                (declare fib_n_minus_1 (: (func.call {:callee @fibonacci} %n_minus_1) i32))

                ;; ✅ TERSE: Compute fib(n-2)
                (declare c2 (arith.constant {:value (: 2 i32)}))
                (declare n_minus_2 (arith.subi %n %c2))
                (declare fib_n_minus_2 (: (func.call {:callee @fibonacci} %n_minus_2) i32))

                ;; ✅ TERSE: Add results
                (declare sum (arith.addi %fib_n_minus_1 %fib_n_minus_2))
                %sum))                ;; Implicit (scf.yield %sum)
            i32))

          ;; Return result (TERSE!)
          (func.return %result)))))

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

          ;; ✅ FULLY TERSE main function!
          (declare n (arith.constant {:value (: 10 i32)}))
          (declare fib_result (: (func.call {:callee @fibonacci} %n) i32))
          (declare result_i64 (: (arith.extsi %fib_result) i64))

          (func.return %result_i64))))))