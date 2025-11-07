module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.apply_patterns to %arg0 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }

  func.func @main() -> i64 {
    %c42 = arith.constant 42 : i64
    return %c42 : i64
  }
}
