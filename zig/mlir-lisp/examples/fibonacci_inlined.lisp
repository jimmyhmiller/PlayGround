(mlir
 (operation (name func.func)
            (attributes {:sym_name @fibonacci
                        :function_type (!function (inputs i32) (results i32))})
            (regions (region (block [^entry]
                               (arguments [(: %n i32)])
                               (func.return
                                (scf.if
                                 (arith.cmpi {:predicate (: 3 i64)}
                                             %n
                                             (arith.constant {:value (: 1 i32)}))
                                 (region %n)
                                 (region
                                  (arith.addi (: (func.call {:callee @fibonacci}
                                                            (arith.subi %n (arith.constant {:value (: 1 i32)})))
                                                 i32)
                                              (: (func.call {:callee @fibonacci}
                                                            (arith.subi %n (arith.constant {:value (: 2 i32)})))
                                                 i32)))))))))
 (operation (name func.func)
            (attributes {:sym_name @main
                        :function_type (!function (inputs) (results i64))})
            (regions
             (region
              (block [^entry]
                (arguments [])
                (func.return
                 (: (arith.extsi
                     (: (func.call {:callee @fibonacci}
                                   (arith.constant {:value (: 10 i32)}))
                        i32))
                    i64)))))))
