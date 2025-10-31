"test.op"() ({
^bb0:
  %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %1 = "arith.constant"() <{value = 0 : i32}> : () -> i32
  %2 = "arith.cmpi"(%0, %1) <{predicate = 0 : i64}> : (i32, i32) -> i1
  "test.cond_br"(%2)[^bb1, ^bb2] : (i1) -> ()
^bb1:
  "test.return"() : () -> ()
^bb2:
  "test.return"() : () -> ()
}) : () -> ()
