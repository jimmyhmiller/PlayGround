// Working PDL pattern example - replaces custom.foo with arith.constant
module {
  // Payload IR with custom operation
  func.func @test() -> i32 {
    %0 = "custom.foo"() : () -> i32
    return %0 : i32
  }
}

// PDL patterns in separate module
module @patterns {
  pdl.pattern @replace_foo : benefit(1) {
    // Match: any operation named "custom.foo"
    %type = pdl.type : i32
    %op = pdl.operation "custom.foo" -> (%type : !pdl.type)

    // Rewrite: replace with arith.constant
    pdl.rewrite %op {
      %attr = pdl.attribute = 42 : i32
      %new_op = pdl.operation "arith.constant" {"value" = %attr} -> (%type : !pdl.type)
      pdl.replace %op with %new_op
    }
  }
}
