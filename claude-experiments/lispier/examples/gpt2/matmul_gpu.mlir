module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @matmul_kernel(%A: memref<64x64xf32>, %B: memref<64x64xf32>, %C: memref<64x64xf32>) kernel {
      %bx = gpu.block_id x
      %by = gpu.block_id y
      %tx = gpu.thread_id x
      %ty = gpu.thread_id y
      %bdimx = gpu.block_dim x
      %bdimy = gpu.block_dim y
      
      %i_part = arith.muli %bx, %bdimx : index
      %i = arith.addi %i_part, %tx : index
      %j_part = arith.muli %by, %bdimy : index
      %j = arith.addi %j_part, %ty : index
      
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %zero = arith.constant 0.0 : f32
      
      %result = scf.for %k = %c0 to %c64 step %c1 iter_args(%acc = %zero) -> f32 {
        %a_val = memref.load %A[%i, %k] : memref<64x64xf32>
        %b_val = memref.load %B[%k, %j] : memref<64x64xf32>
        %prod = arith.mulf %a_val, %b_val : f32
        %new_acc = arith.addf %acc, %prod : f32
        scf.yield %new_acc : f32
      }
      
      memref.store %result, %C[%i, %j] : memref<64x64xf32>
      gpu.return
    }
  }
  
  // CPU reference matmul for validation
  func.func @matmul_cpu(%A: memref<64x64xf32>, %B: memref<64x64xf32>, %C: memref<64x64xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %zero = arith.constant 0.0 : f32
    
    scf.for %i = %c0 to %c64 step %c1 {
      scf.for %j = %c0 to %c64 step %c1 {
        %result = scf.for %k = %c0 to %c64 step %c1 iter_args(%acc = %zero) -> f32 {
          %a_val = memref.load %A[%i, %k] : memref<64x64xf32>
          %b_val = memref.load %B[%k, %j] : memref<64x64xf32>
          %prod = arith.mulf %a_val, %b_val : f32
          %new_acc = arith.addf %acc, %prod : f32
          scf.yield %new_acc : f32
        }
        memref.store %result, %C[%i, %j] : memref<64x64xf32>
      }
    }
    return
  }
  
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c63 = arith.constant 63 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.001 : f32
    
    %hostA = memref.alloc() : memref<64x64xf32>
    %hostB = memref.alloc() : memref<64x64xf32>
    %hostC = memref.alloc() : memref<64x64xf32>
    %cpuC = memref.alloc() : memref<64x64xf32>
    
    // Initialize
    scf.for %i = %c0 to %c64 step %c1 {
      scf.for %j = %c0 to %c64 step %c1 {
        %i_i64 = arith.index_cast %i : index to i64
        %j_i64 = arith.index_cast %j : index to i64
        %c64_i64 = arith.constant 64 : i64
        %idx_part = arith.muli %i_i64, %c64_i64 : i64
        %idx = arith.addi %idx_part, %j_i64 : i64
        %idx_f32 = arith.sitofp %idx : i64 to f32
        %val = arith.mulf %idx_f32, %cst : f32
        memref.store %val, %hostA[%i, %j] : memref<64x64xf32>
        memref.store %val, %hostB[%i, %j] : memref<64x64xf32>
      }
    }
    
    // GPU computation
    %gpuA, %t1 = gpu.alloc async [] () : memref<64x64xf32>
    %gpuB, %t2 = gpu.alloc async [%t1] () : memref<64x64xf32>
    %gpuC, %t3 = gpu.alloc async [%t2] () : memref<64x64xf32>
    %t4 = gpu.memcpy async [%t3] %gpuA, %hostA : memref<64x64xf32>, memref<64x64xf32>
    %t5 = gpu.memcpy async [%t4] %gpuB, %hostB : memref<64x64xf32>, memref<64x64xf32>
    %t6 = gpu.launch_func async [%t5] @kernels::@matmul_kernel blocks in (%c8, %c8, %c1) threads in (%c8, %c8, %c1) args(%gpuA : memref<64x64xf32>, %gpuB : memref<64x64xf32>, %gpuC : memref<64x64xf32>)
    %t7 = gpu.memcpy async [%t6] %hostC, %gpuC : memref<64x64xf32>, memref<64x64xf32>
    gpu.wait [%t7]
    
    // CPU reference
    func.call @matmul_cpu(%hostA, %hostB, %cpuC) : (memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>) -> ()
    
    // Compare results
    %gpu00 = memref.load %hostC[%c0, %c0] : memref<64x64xf32>
    %cpu00 = memref.load %cpuC[%c0, %c0] : memref<64x64xf32>
    %gpu63 = memref.load %hostC[%c63, %c63] : memref<64x64xf32>
    %cpu63 = memref.load %cpuC[%c63, %c63] : memref<64x64xf32>
    
    vector.print %gpu00 : f32
    vector.print %cpu00 : f32
    vector.print %gpu63 : f32
    vector.print %cpu63 : f32
    
    // Cleanup
    gpu.dealloc %gpuA : memref<64x64xf32>
    gpu.dealloc %gpuB : memref<64x64xf32>
    gpu.dealloc %gpuC : memref<64x64xf32>
    memref.dealloc %hostA : memref<64x64xf32>
    memref.dealloc %hostB : memref<64x64xf32>
    memref.dealloc %hostC : memref<64x64xf32>
    memref.dealloc %cpuC : memref<64x64xf32>
    
    return
  }
}
