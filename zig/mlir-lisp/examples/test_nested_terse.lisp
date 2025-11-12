(mlir
 ;; Test nested terse scf.if regions (like is_even function)
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
                                 (region (arith.constant {:value (: 1 i1)}))
                                 (region
                                  (scf.if
                                   (arith.cmpi {:predicate (: 0 i64)}
                                               %n
                                               (arith.constant {:value (: 1 i32)}))
                                   (region (arith.constant {:value (: 0 i1)}))
                                   (region
                                    (: (func.call {:callee @is_even}
                                                  (arith.subi %n (arith.constant {:value (: 2 i32)}))) i1))))))))))

 (operation (name func.func)
            (attributes {:sym_name @main
                        :function_type (!function (inputs) (results i64))})
            (regions
             (region
              (block [^entry]
                (arguments [])
                (func.return
                 (: (arith.extsi
                     (: (func.call {:callee @is_even}
                                   (arith.constant {:value (: 6 i32)})) i1)) i64)))))))
