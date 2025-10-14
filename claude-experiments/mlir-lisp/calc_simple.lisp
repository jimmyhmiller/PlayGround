(defirdl-dialect calc
  :namespace "calc"
  :description "Calculator dialect"
  (defirdl-op add
    :summary "Addition operation"
    :operands [(lhs i32) (rhs i32)]
    :results [(result i32)])
  (defirdl-op mul
    :summary "Multiplication operation"
    :operands [(lhs i32) (rhs i32)]
    :results [(result i32)]))

(defpdl-pattern calc_add_to_arith
  :benefit 1
  :match (calc.add $x $y)
  :rewrite (arith.addi $x $y))

(defpdl-pattern calc_mul_to_arith
  :benefit 1
  :match (calc.mul $x $y)
  :rewrite (arith.muli $x $y))

(defn compute [] i32
  (calc.add (calc.mul 10 20) 30))

(jit-execute compute)
