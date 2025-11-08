// GPU Matrix Multiplication Example
// Computes C = A * B for NxN matrices
module attributes {gpu.container_module} {
  gpu.module @matmul_kernel {
    // Simple matrix multiplication kernel
    // Each thread computes one element of the result matrix
    gpu.func @matmul(%A: memref<16x16xf32>, %B: memref<16x16xf32>, %C: memref<16x16xf32>) kernel {
      // Get 2D thread coordinates
      %row = gpu.block_id x
      %col = gpu.block_id y

      // Initialize accumulator
      %zero = arith.constant 0.0 : f32
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index

      // Compute dot product for this element
      %sum = scf.for %k = %c0 to %c16 step %c1 iter_args(%acc = %zero) -> (f32) {
        // Load A[row, k]
        %a_val = memref.load %A[%row, %k] : memref<16x16xf32>

        // Load B[k, col]
        %b_val = memref.load %B[%k, %col] : memref<16x16xf32>

        // Multiply and accumulate
        %prod = arith.mulf %a_val, %b_val : f32
        %new_acc = arith.addf %acc, %prod : f32

        scf.yield %new_acc : f32
      }

      // Store result C[row, col] = sum
      memref.store %sum, %C[%row, %col] : memref<16x16xf32>
      gpu.return
    }
  }

  func.func @main() -> i32 {
    // Matrix size: 16x16
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index

    // Allocate matrices
    %A = memref.alloc() : memref<16x16xf32>
    %B = memref.alloc() : memref<16x16xf32>
    %C = memref.alloc() : memref<16x16xf32>

    // Launch kernel with 16x16 grid (one block per output element)
    gpu.launch_func @matmul_kernel::@matmul
      blocks in (%c16, %c16, %c1)
      threads in (%c1, %c1, %c1)
      args(%A : memref<16x16xf32>, %B : memref<16x16xf32>, %C : memref<16x16xf32>)

    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
