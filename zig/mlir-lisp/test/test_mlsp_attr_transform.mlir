module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.with_pdl_patterns %arg0 : !transform.any_op {
    ^bb0(%payload: !transform.any_op):

      // Pattern: mlsp.string_const → llvm.mlir.addressof
      // Need to capture and transfer the 'global' attribute
      pdl.pattern @mlsp_string_const : benefit(1) {
        %ptr_type = pdl.type : !llvm.ptr
        %global_attr = pdl.attribute
        %op = pdl.operation "mlsp.string_const" {"global" = %global_attr} -> (%ptr_type : !pdl.type)

        pdl.rewrite %op {
          %new_op = pdl.operation "llvm.mlir.addressof" {"global_name" = %global_attr} -> (%ptr_type : !pdl.type)
          pdl.replace %op with %new_op
        }
      }

      transform.sequence %payload : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %matched = pdl_match @mlsp_string_const in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.yield
      }
    }
    transform.yield
  }
}
