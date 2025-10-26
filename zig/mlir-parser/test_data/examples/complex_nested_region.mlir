%cond = "test.condition"() : () -> i1
%0 = "scf.if"(%cond) ({
  %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  "scf.yield"(%1) : (i32) -> ()
}, {
  %2 = "arith.constant"() <{value = 2 : i32}> : () -> i32
  "scf.yield"(%2) : (i32) -> ()
}) : (i1) -> i32
