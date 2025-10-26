%0 = "arith.constant"() <{value = 1 : i1}> : () -> i1
"scf.if"(%0) ({
  %1 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  "scf.yield"(%1) : (i32) -> ()
}, {
  %2 = "arith.constant"() <{value = 0 : i32}> : () -> i32
  "scf.yield"(%2) : (i32) -> ()
}) : (i1) -> i32
