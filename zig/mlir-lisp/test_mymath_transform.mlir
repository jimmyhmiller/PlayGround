module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.with_pdl_patterns %arg0 : !transform.any_op {
    ^bb0(%payload: !transform.any_op):
      pdl.pattern @mymath_to_arith : benefit(1) {
        %lhs = pdl.operand
        %rhs = pdl.operand
        %result_type = pdl.type
        %mymath_op = pdl.operation "mymath.add"(%lhs, %rhs : !pdl.value, !pdl.value) -> (%result_type : !pdl.type)
        pdl.rewrite %mymath_op {
          %arith_op = pdl.operation "arith.addi"(%lhs, %rhs : !pdl.value, !pdl.value) -> (%result_type : !pdl.type)
          pdl.replace %mymath_op with %arith_op
        }
      }

      transform.sequence %payload : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %matched = pdl_match @mymath_to_arith in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.yield
      }
    }
    transform.yield
  }
}
