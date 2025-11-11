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

          ;; scf.if still verbose (no terse syntax yet)
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

              ;; Else region - ALL TERSE!
              (region
                (block
                  (arguments [])
                  
                  (scf.yield
                   (arith.addi
                    (: (func.call {:callee @fibonacci}
                                  (arith.subi %n
                                              (arith.constant {:value (: 1 i32)}))) i32)
                    (: (func.call {:callee @fibonacci}
                                  (arith.subi %n (arith.constant {:value (: 2 i32)}))) i32)))))))

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
