module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.with_pdl_patterns %arg0 : !transform.any_op {
    ^bb0(%payload: !transform.any_op):
      pdl.pattern @replace_addi : benefit(1) {
        %lhs = pdl.operand
        %rhs = pdl.operand
        %result_type = pdl.type
        %addi_op = pdl.operation "arith.addi"(%lhs, %rhs : !pdl.value, !pdl.value) -> (%result_type : !pdl.type)
        pdl.rewrite %addi_op {
          %muli_op = pdl.operation "arith.muli"(%lhs, %rhs : !pdl.value, !pdl.value) -> (%result_type : !pdl.type)
          pdl.replace %addi_op with %muli_op
        }
      }

      transform.sequence %payload : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %matched = pdl_match @replace_addi in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.yield
      }
    }
    transform.yield
  }
}
