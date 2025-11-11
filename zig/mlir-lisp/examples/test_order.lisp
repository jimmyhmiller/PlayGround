(mlir
  (operation
    (name func.func)
    (attributes {
      :sym_name @test
      :function_type (!function (inputs) (results i32))
    })
    (regions
      (region
        (block
          (arguments [])

          (declare x (arith.constant {:value (: 1 i32)}))

          (operation
            (name scf.if)
            (result-bindings [%result])
            (result-types i32)
            (operands %x)
            (regions
              (region
                (block
                  (arguments [])
                  (operation
                    (name scf.yield)
                    (operands %x))))))

          (func.return %result)))))
