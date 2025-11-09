module attributes {gpu.container_module} {
  func.func @main() {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
               threads(%tx, %ty, %tz) in (%block_x = %c2, %block_y = %c1, %block_z = %c1) {
      gpu.printf "Hello from thread %lld\n", %tx : index
      gpu.terminator
    }
    return
  }
}
