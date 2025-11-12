(mlir
 ;; Helper function: compute n! recursively
 ;; factorial(n) = if n <= 1 then 1 else n * factorial(n-1)
 (operation (name func.func)
            (attributes {:sym_name @factorial
                        :function_type (!function (inputs i32) (results i32))})
            (regions (region (block [^entry]
                               (arguments [(: %n i32)])
                               (func.return
                                (scf.if
                                 (arith.cmpi {:predicate (: 3 i64)}
                                             %n
                                             (arith.constant {:value (: 1 i32)}))
                                 (region (block (scf.yield (arith.constant {:value (: 1 i32)}))))
                                 (region (block (scf.yield
                                  (arith.muli
                                   %n
                                   (: (func.call {:callee @factorial}
                                                 (arith.subi %n (arith.constant {:value (: 1 i32)}))) i32)))))))))))

 ;; Helper function: sum from 1 to n recursively
 ;; sum_to_n(n) = if n <= 0 then 0 else n + sum_to_n(n-1)
 (operation (name func.func)
            (attributes {:sym_name @sum_to_n
                        :function_type (!function (inputs i32) (results i32))})
            (regions (region (block [^entry]
                               (arguments [(: %n i32)])
                               (func.return
                                (scf.if
                                 (arith.cmpi {:predicate (: 3 i64)}
                                             %n
                                             (arith.constant {:value (: 0 i32)}))
                                 (region (block (scf.yield (arith.constant {:value (: 0 i32)}))))
                                 (region (block (scf.yield
                                  (arith.addi
                                   %n
                                   (: (func.call {:callee @sum_to_n}
                                                 (arith.subi %n (arith.constant {:value (: 1 i32)}))) i32)))))))))))

 ;; Helper function: compute n^2
 ;; square(n) = n * n
 (operation (name func.func)
            (attributes {:sym_name @square
                        :function_type (!function (inputs i32) (results i32))})
            (regions (region (block [^entry]
                               (arguments [(: %n i32)])
                               (func.return (arith.muli %n %n))))))

 ;; Helper function: compute n^3
 ;; cube(n) = n * n * n = n * square(n)
 (operation (name func.func)
            (attributes {:sym_name @cube
                        :function_type (!function (inputs i32) (results i32))})
            (regions (region (block [^entry]
                               (arguments [(: %n i32)])
                               (func.return
                                (arith.muli %n (: (func.call {:callee @square} %n) i32)))))))

 ;; Helper function: check if n is even
 ;; is_even(n) = if n <= 0 then true else if n == 1 then false else is_even(n-2)
 (operation (name func.func)
            (attributes {:sym_name @is_even
                        :function_type (!function (inputs i32) (results i1))})
            (regions (region (block [^entry]
                               (arguments [(: %n i32)])
                               (func.return
                                (scf.if
                                 (arith.cmpi {:predicate (: 3 i64)}
                                             %n
                                             (arith.constant {:value (: 0 i32)}))
                                 (region (block (scf.yield (arith.constant {:value (: 1 i1)}))))
                                 (region (block (scf.yield
                                  (scf.if
                                   (arith.cmpi {:predicate (: 0 i64)}
                                               %n
                                               (arith.constant {:value (: 1 i32)}))
                                   (region (block (scf.yield (arith.constant {:value (: 0 i1)}))))
                                   (region (block (scf.yield
                                    (: (func.call {:callee @is_even}
                                                  (arith.subi %n (arith.constant {:value (: 2 i32)}))) i1))))))))))))))

 ;; Helper function: check if n is divisible by 3
 (operation (name func.func)
            (attributes {:sym_name @div_by_3
                        :function_type (!function (inputs i32) (results i1))})
            (regions (region (block [^entry]
                               (arguments [(: %n i32)])
                               (func.return
                                (scf.if
                                 (arith.cmpi {:predicate (: 3 i64)}
                                             %n
                                             (arith.constant {:value (: 0 i32)}))
                                 (region (block (scf.yield (arith.constant {:value (: 1 i1)}))))
                                 (region (block (scf.yield
                                  (scf.if
                                   (arith.cmpi {:predicate (: 2 i64)}
                                               %n
                                               (arith.constant {:value (: 3 i32)}))
                                   (region (block (scf.yield (arith.constant {:value (: 0 i1)}))))
                                   (region (block (scf.yield
                                    (: (func.call {:callee @div_by_3}
                                                  (arith.subi %n (arith.constant {:value (: 3 i32)}))) i1))))))))))))))

 ;; Complex computation: compute(n) = if is_even(n) then factorial(n) else square(n) + sum_to_n(n)
 (operation (name func.func)
            (attributes {:sym_name @compute
                        :function_type (!function (inputs i32) (results i32))})
            (regions (region (block [^entry]
                               (arguments [(: %n i32)])
                               (func.return
                                (scf.if
                                 (: (func.call {:callee @is_even} %n) i1)
                                 (region (block (scf.yield (: (func.call {:callee @factorial} %n) i32))))
                                 (region (block (scf.yield
                                  (arith.addi
                                   (: (func.call {:callee @square} %n) i32)
                                   (: (func.call {:callee @sum_to_n} %n) i32)))))))))))

 ;; More complex: complex_compute(n) = if div_by_3(n) then cube(n) else compute(n)
 (operation (name func.func)
            (attributes {:sym_name @complex_compute
                        :function_type (!function (inputs i32) (results i32))})
            (regions (region (block [^entry]
                               (arguments [(: %n i32)])
                               (func.return
                                (scf.if
                                 (: (func.call {:callee @div_by_3} %n) i1)
                                 (region (block (scf.yield (: (func.call {:callee @cube} %n) i32))))
                                 (region (block (scf.yield (: (func.call {:callee @compute} %n) i32))))))))))

 ;; Aggregate: sum_complex_compute(n) = if n <= 0 then 0 else complex_compute(n) + sum_complex_compute(n-1)
 (operation (name func.func)
            (attributes {:sym_name @sum_complex_compute
                        :function_type (!function (inputs i32) (results i32))})
            (regions (region (block [^entry]
                               (arguments [(: %n i32)])
                               (func.return
                                (scf.if
                                 (arith.cmpi {:predicate (: 3 i64)}
                                             %n
                                             (arith.constant {:value (: 0 i32)}))
                                 (region (block (scf.yield (arith.constant {:value (: 0 i32)}))))
                                 (region (block (scf.yield
                                  (arith.addi
                                   (: (func.call {:callee @complex_compute} %n) i32)
                                   (: (func.call {:callee @sum_complex_compute}
                                                 (arith.subi %n (arith.constant {:value (: 1 i32)}))) i32)))))))))))

 ;; Main function: computes sum_complex_compute(6) = 311
 ;; Trace:
 ;; - complex_compute(1) = square(1) + sum_to_n(1) = 1 + 1 = 2 (odd, not div 3)
 ;; - complex_compute(2) = factorial(2) = 2 (even)
 ;; - complex_compute(3) = cube(3) = 27 (div 3)
 ;; - complex_compute(4) = factorial(4) = 24 (even)
 ;; - complex_compute(5) = square(5) + sum_to_n(5) = 25 + 15 = 40 (odd, not div 3)
 ;; - complex_compute(6) = cube(6) = 216 (div 3)
 ;; Total: 2 + 2 + 27 + 24 + 40 + 216 = 311
 (operation (name func.func)
            (attributes {:sym_name @main
                        :function_type (!function (inputs) (results i64))})
            (regions
             (region
              (block [^entry]
                (arguments [])
                (func.return
                 (: (arith.extsi
                     (: (func.call {:callee @sum_complex_compute}
                                   (arith.constant {:value (: 6 i32)})) i32)) i64)))))))