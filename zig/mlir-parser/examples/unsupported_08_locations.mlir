// Example: Trailing locations (source location tracking)
// Grammar: trailing-location ::= `loc` `(` location `)`

// File location
%0 = "arith.constant"() <{value = 42 : i32}> : () -> i32 loc("example.mlir":1:1)

// Named location
%1 = "arith.constant"() <{value = 13 : i32}> : () -> i32 loc("my_constant")

// Call site location (fused location)
%2 = "arith.addi"(%0, %1) : (i32, i32) -> i32 loc(fused["example.mlir":3:10, "another.mlir":5:20])

// Unknown location
%3 = "arith.muli"(%2, %2) : (i32, i32) -> i32 loc(unknown)

// Fused location with metadata
%4 = "test.op"() : () -> i32 loc(fused<"inlined">["caller.mlir":10:5, "callee.mlir":20:10])
