"builtin.module"() ({
  %0 = "arith.constant"() <{value = true}> : () -> i1
  %1 = "scf.if"(%0) ({
    %3 = "arith.constant"() <{value = 42 : i32}> : () -> i32
    "scf.yield"(%3) : (i32) -> ()
  }, {
    %2 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "scf.yield"(%2) : (i32) -> ()
  }) : (i1) -> i32
}) : () -> ()

