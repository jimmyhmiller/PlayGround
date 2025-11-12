(module
  (defn add_two [(: %a i32) (: %b i32)] i32
    (declare result (: (arith.addi %a %b) i32))
    (func.return %result))

  (defn main [] i32
    (declare result (: (func.call {:callee @add_two}
                                  (arith.constant {:value (: 5 i32)})
                                  (arith.constant {:value (: 7 i32)}))
                       i32))
    (func.return %result)))
