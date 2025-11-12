(mlir
 ;; Simple power function: n^2
 (operation (name func.func)
            (attributes {:sym_name @square
                        :function_type (!function (inputs i32) (results i32))})
            (regions (region (block [^entry]
                               (arguments [(: %n i32)])
                               (func.return
                                (arith.muli %n %n))))))

 ;; Main: compute square(7) = 49
 (operation (name func.func)
            (attributes {:sym_name @main
                        :function_type (!function (inputs) (results i64))})
            (regions
             (region
              (block [^entry]
                (arguments [])
                (func.return
                 (: (arith.extsi
                     (: (func.call {:callee @square}
                                   (arith.constant {:value (: 7 i32)})
                        i32))
                    i64))))))))