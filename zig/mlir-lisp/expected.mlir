"builtin.module"() ({
  "func.func"() <{function_type = (i32) -> i32, sym_name = "fibonacci"}> ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %1 = "arith.cmpi"(%arg0, %0) <{predicate = 3 : i64}> : (i32, i32) -> i1
    %2 = "scf.if"(%1) ({
      "scf.yield"(%arg0) : (i32) -> ()
    }, {
      %3 = "arith.constant"() <{value = 1 : i32}> : () -> i32
      %4 = "arith.subi"(%arg0, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %5 = "func.call"(%4) <{callee = @fibonacci}> : (i32) -> i32
      %6 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %7 = "arith.subi"(%arg0, %6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %8 = "func.call"(%7) <{callee = @fibonacci}> : (i32) -> i32
      %9 = "arith.addi"(%5, %8) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "scf.yield"(%9) : (i32) -> ()
    }) : (i1) -> i32
    "func.return"(%2) : (i32) -> ()
  }) : () -> ()
}) : () -> ()
