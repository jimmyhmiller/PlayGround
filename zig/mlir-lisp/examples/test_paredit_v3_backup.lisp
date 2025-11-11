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
          (arguments [ (: %n i32) ])

          (declare c1 (arith.constant {:value (: 1 i32)}))
          (declare cond (: (arith.cmpi {:predicate (: 3 i64)} %n %c1) i1))

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
                    (operands %n))))
              (region
                (block
                  (arguments [])
                  (declare c1_rec (arith.constant {:value (: 1 i32)}))
                  (declare n_minus_1 (arith.subi %n %c1_rec))
                  (operation
                    (name scf.yield)
                    (operands %n_minus_1)))))))

          (func.return %result))))))