(mlir
 ;; Test terse scf.if regions - should auto-add scf.yield
 (operation (name func.func)
            (attributes {:sym_name @test_terse
                        :function_type (!function (inputs i32) (results i32))})
            (regions (region (block [^entry]
                               (arguments [(: %n i32)])
                               (func.return
                                (scf.if
                                 (arith.cmpi {:predicate (: 3 i64)}
                                             %n
                                             (arith.constant {:value (: 0 i32)}))
                                 (region (arith.constant {:value (: 1 i32)}))
                                 (region (arith.muli %n %n))))))))

 (operation (name func.func)
            (attributes {:sym_name @main
                        :function_type (!function (inputs) (results i64))})
            (regions
             (region
              (block [^entry]
                (arguments [])
                (func.return
                 (: (arith.extsi
                     (: (func.call {:callee @test_terse}
                                   (arith.constant {:value (: 5 i32)})) i32)) i64)))))))
