(mlir
  (operation
    (name func.func)
    (attributes {
      :sym_name @main
      :function_type (!function (inputs) (results i32))
    })
    
    (regions
      (region
        (block
          (arguments [])

          (declare x (arith.constant {:value (: 1 i32)}))
          (declare y (arith.constant {:value (: 2 i32)}))
          (declare cond (: (arith.cmpi {:predicate (: 4 i64)} %x %y) i1))

          (operation
            (name scf.if)
            (result-bindings [%result])
            (result-types i32)
            (operands %cond)
            (regions
              (region
                (block
                  (arguments [])
                  (operation
                    (name scf.yield)
                    (operands %x))))
              (region
                (block
                  (arguments [])
                  (operation
                    (name scf.yield)
                    (operands %y))))))

          (operation
            (name func.return)
            (operands %result)))))))
