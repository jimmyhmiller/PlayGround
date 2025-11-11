(mlir
  (declare c1 (arith.constant {:value (: 42 i64)}))
  (declare c2 (arith.constant {:value (: 10 i64)}))
  (declare result (arith.addi %c1 %c2)))
