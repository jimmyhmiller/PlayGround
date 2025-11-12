(mlir
 ;; Helper: factorial(n) = if n <= 1 then 1 else n * factorial(n-1)
 (operation (name func.func)
            (attributes {:sym_name @factorial
                        :function_type (!function (inputs i32) (results i32))})
            (regions (region (block [^entry]
                               (arguments [(: %n i32)])
                               (func.return
                                (scf.if
                                 (arith.cmpi {:predicate (: 3 i64)} %n (arith.constant {:value (: 1 i32)}))
                                 (region (arith.constant {:value (: 1 i32)}))
                                 (region
                                  (arith.muli %n
                                              (: (func.call {:callee @factorial}
                                                            (arith.subi %n (arith.constant {:value (: 1 i32)}))) i32)))))))))

 ;; Helper: sum_to_n(n) = if n <= 0 then 0 else n + sum_to_n(n-1)
 (operation (name func.func)
            (attributes {:sym_name @sum_to_n
                        :function_type (!function (inputs i32) (results i32))})
            (regions (region (block [^entry]
                               (arguments [(: %n i32)])
                               (func.return
                                (scf.if
                                 (arith.cmpi {:predicate (: 3 i64)} %n (arith.constant {:value (: 0 i32)}))
                                 (region (arith.constant {:value (: 0 i32)}))
                                 (region
                                  (arith.addi %n
                                              (: (func.call {:callee @sum_to_n}
                                                            (arith.subi %n (arith.constant {:value (: 1 i32)}))) i32)))))))))

 ;; Helper: square(n) = n * n
 (operation (name func.func)
            (attributes {:sym_name @square
                        :function_type (!function (inputs i32) (results i32))})
            (regions (region (block [^entry]
                               (arguments [(: %n i32)])
                               (func.return (arith.muli %n %n))))))

 ;; Compute function that adds two function calls
 (operation (name func.func)
            (attributes {:sym_name @compute
                        :function_type (!function (inputs i32) (results i32))})
            (regions (region (block [^entry]
                               (arguments [(: %n i32)])
                               (func.return
                                (arith.addi
                                 (: (func.call {:callee @square} %n) i32)
                                 (: (func.call {:callee @sum_to_n} %n) i32)))))))

 ;; Main: compute(5) = square(5) + sum_to_n(5) = 25 + 15 = 40
 (operation (name func.func)
            (attributes {:sym_name @main
                        :function_type (!function (inputs) (results i64))})
            (regions
             (region
              (block [^entry]
                (arguments [])
                (func.return
                 (: (arith.extsi
                     (: (func.call {:callee @compute}
                                   (arith.constant {:value (: 5 i32)})) i32)) i64)))))))