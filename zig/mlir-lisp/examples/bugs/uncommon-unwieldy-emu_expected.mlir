"builtin.module"() ({
  "llvm.mlir.global"() <{addr_space = 0 : i32, constant, dso_local, global_type = !llvm.array<5 x i8>, linkage = #llvm.linkage<internal>, sym_name = "test_global", unnamed_addr = 2 : i64, value = "test\00", visibility_ = 0 : i64}> ({
  }) : () -> ()
  "llvm.func"() <{CConv = #llvm.cconv<ccc>, function_type = !llvm.func<void ()>, linkage = #llvm.linkage<external>, no_unwind, sym_name = "test_func", unnamed_addr = 0 : i64, visibility_ = 0 : i64}> ({
  }) : () -> ()
}) : () -> ()
