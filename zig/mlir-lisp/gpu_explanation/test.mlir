
module attributes {gpu.container_module} {
  llvm.func @main() {
    %1 = arith.constant 1 : index
    gpu.launch_func @test_func::@test_func blocks in (%1, %1, %1) threads in (%1, %1, %1)
    llvm.return
  }
  gpu.module @test_func {
    gpu.func @test_func () kernel {
      %0 = memref.alloca() : memref<1xi8>
      %1 = arith.constant 0 :i8
      %2 = arith.constant 0 :index
      memref.store %1, %0[%2] : memref<1xi8>
      gpu.return
    }
  }
}
