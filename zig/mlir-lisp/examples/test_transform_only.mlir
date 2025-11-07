module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.apply_patterns.transform.with_pdl_patterns %arg0 {
      pdl.pattern @replace_constant : benefit(1) {
        %cst_val = pdl.attribute = 10 : i32
        %type = pdl.type : i32
        %op = pdl.operation "arith.constant" {"value" = %cst_val} -> (%type : !pdl.type)
        pdl.rewrite %op {
          %new_val = pdl.attribute = 42 : i32
          %new_op = pdl.operation "arith.constant" {"value" = %new_val} -> (%type : !pdl.type)
          pdl.replace %op with %new_op
        }
      }
    } : !transform.any_op
    transform.yield
  }
}

func.func @main() -> i32 {
  %c10 = arith.constant 10 : i32
  return %c10 : i32
}
