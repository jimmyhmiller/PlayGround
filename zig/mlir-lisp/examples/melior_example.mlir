module {
  irdl.dialect @mymath {
    irdl.operation @add {
      %0 = irdl.is i8
      %1 = irdl.is i16
      %2 = irdl.is i32
      %3 = irdl.is i64
      %4 = irdl.is f32
      %5 = irdl.is f64
      %arith_type = irdl.any_of(%0, %1, %2, %3, %4, %5)
      irdl.operands(lhs: %arith_type, rhs: %arith_type)
      irdl.results(result: %arith_type)
    }
  }
}

module {
  transform.with_pdl_patterns {
  ^bb0(%root: !transform.any_op):
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
    
    transform.sequence %root : !transform.any_op failures(propagate) {
    ^bb1(%arg1: !transform.any_op):
      %matched = pdl_match @mymath_to_arith in %arg1 : (!transform.any_op) -> !transform.any_op
      transform.yield
    }
  }
}

func.func @main() -> i32 {
  %c10 = arith.constant 10 : i32
  %c32 = arith.constant 32 : i32
  %result = "mymath.add"(%c10, %c32) : (i32, i32) -> i32
  return %result : i32
}
