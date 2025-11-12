(mlir
 ;; Test nested scf.if to isolate the crash
 (operation (name func.func)
            (attributes {:sym_name @nested_if_test
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
                                   (region (arith.constant {:value (: 1 i1)}))))))))))

 (operation (name func.func)
            (attributes {:sym_name @main
                        :function_type (!function (inputs) (results i64))})
            (regions
             (region
              (block [^entry]
                (arguments [])
                (func.return
                 (: (arith.extsi
                     (: (func.call {:callee @nested_if_test}
                                   (arith.constant {:value (: 5 i32)})) i1)) i64)))))))
