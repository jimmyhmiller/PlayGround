module attributes {gpu.container_module} {
  llvm.func @main() {
    %c1 = arith.constant 1 : index
    gpu.launch_func  @test_func::@test_func blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  
    llvm.return
  }
  gpu.module @test_func attributes {llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"} {
    llvm.func @test_func() attributes {gpu.kernel, rocdl.kernel} {
      %0 = llvm.mlir.constant(0 : index) : i64
      %1 = llvm.mlir.constant(0 : i8) : i8
      %2 = llvm.mlir.constant(1 : index) : i64
      %3 = llvm.mlir.constant(1 : index) : i64
      %4 = llvm.alloca %2 x i8 : (i64) -> !llvm.ptr
      %5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %6 = llvm.insertvalue %4, %5[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %7 = llvm.insertvalue %4, %6[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %8 = llvm.mlir.constant(0 : index) : i64
      %9 = llvm.insertvalue %8, %7[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %10 = llvm.insertvalue %2, %9[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %11 = llvm.insertvalue %3, %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %12 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %13 = llvm.getelementptr %12[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      llvm.store %1, %13 : i8, !llvm.ptr
      llvm.return
    }
  }
}

