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

func.func @main() -> i32 {
  %c10 = arith.constant 10 : i32
  %c32 = arith.constant 32 : i32
  %result = "mymath.add"(%c10, %c32) : (i32, i32) -> i32
  return %result : i32
}
