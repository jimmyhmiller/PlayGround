// Simple arithmetic operations
// Grammar: operation+ (multiple operations)
%0 = arith.constant 42 : i32
%1 = arith.constant 13 : i32
%2 = arith.addi %0, %1 : i32
