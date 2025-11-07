module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.with_pdl_patterns %arg0 : !transform.any_op {
    ^bb0(%payload: !transform.any_op):
      pdl.pattern @replace_magic : benefit(1) {
        %type = pdl.type : i32
        %op = pdl.operation "custom.magic" -> (%type : !pdl.type)
        pdl.rewrite %op {
          %attr = pdl.attribute = 42 : i32
          %new = pdl.operation "arith.constant" {"value" = %attr} -> (%type : !pdl.type)
          pdl.replace %op with %new
        }
      }

      transform.sequence %payload : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %matched = pdl_match @replace_magic in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.yield
      }
    }
    transform.yield
  }
}
