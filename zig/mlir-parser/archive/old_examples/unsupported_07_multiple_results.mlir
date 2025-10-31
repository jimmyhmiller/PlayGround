// Example: Operations with multiple results
// Grammar: op-result ::= value-id (`:` integer-literal)?
// Grammar: value-use ::= value-id (`#` decimal-literal)?

// Operation producing two results (note the :2)
%0:2 = "test.multi_result"() : () -> (i32, i32)

// Using individual results with # notation
%1 = "arith.addi"(%0#0, %0#1) : (i32, i32) -> i32

// Operation producing three results
%2:3 = "test.triple_result"() : () -> (f32, f32, f32)
%3 = "arith.addf"(%2#0, %2#1) : (f32, f32) -> f32
%4 = "arith.addf"(%3, %2#2) : (f32, f32) -> f32

// Mix of single and multiple results
%5 = "arith.constant"() <{value = 10 : i32}> : () -> i32
%6:2 = "test.split"(%5) : (i32) -> (i32, i32)
%7 = "arith.muli"(%6#0, %6#1) : (i32, i32) -> i32
