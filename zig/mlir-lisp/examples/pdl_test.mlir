module {
  irdl.dialect @custom {
    irdl.operation @magic {
      %i32 = irdl.is i32
      irdl.results("result" = %i32)
    }
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.apply_patterns.transform.with_pdl_patterns %arg0 {
      pdl.pattern @replace_magic : benefit(1) {
        %type = pdl.type : i32
        %op = pdl.operation "custom.magic" -> (%type : !pdl.type)
        pdl.rewrite %op {
          %attr = pdl.attribute = 42 : i32
          %new = pdl.operation "arith.constant" {"value" = %attr} -> (%type : !pdl.type)
          pdl.replace %op with %new
        }
      }
    } : !transform.any_op
    transform.yield
  }
}

func.func @main() -> i64 {
  %0 = "custom.magic"() : () -> i32
  %1 = arith.extsi %0 : i32 to i64
  return %1 : i64
}
