%0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
%1 = "arith.constant"() <{value = 13 : i32}> : () -> i32
%2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
