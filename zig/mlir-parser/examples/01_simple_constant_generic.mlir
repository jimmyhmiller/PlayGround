// Generic format output from: echo '%0 = arith.constant 42 : i32' | mlir-opt -mlir-print-op-generic
"builtin.module"() ({
  %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
}) : () -> ()
