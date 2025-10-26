%0:2 = "test.multi_result"() : () -> (i32, i32)
%1 = "arith.addi"(%0#0, %0#1) : (i32, i32) -> i32
