;; Demonstration of terse operation syntax

(mlir
  ;; Declare constants with inferred types from attributes
  (declare c1 (arith.constant {:value (: 42 i64)}))
  (declare c2 (arith.constant {:value (: 10 i64)}))

  ;; Arithmetic operations with type inferred from operands
  (declare sum (arith.addi %c1 %c2))
  (declare product (arith.muli %sum %c2))

  ;; Operations with attributes
  (declare c3 (arith.constant {:value (: 100 i64)}))

  ;; More arithmetic
  (declare difference (arith.subi %c3 %product)))
