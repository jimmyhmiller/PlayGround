// Simple GPU kernel that squares each element of a 10x10 matrix
module attributes {gpu.container_module} {
  // Define the GPU kernel module
  gpu.module @kernel_module {
    gpu.func @square_kernel(%input: memref<10x10xf32>, %output: memref<10x10xf32>) kernel {
      %block_x = gpu.block_id x
      %thread_x = gpu.thread_id x

      // Load value from input
      %val = memref.load %input[%block_x, %thread_x] : memref<10x10xf32>

      // Square it
      %result = arith.mulf %val, %val : f32

      // Store to output
      memref.store %result, %output[%block_x, %thread_x] : memref<10x10xf32>
      gpu.return
    }
  }

  // Host function that launches the kernel
  func.func @main() -> i32 {
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index

    // Allocate input and output memrefs
    %input = memref.alloc() : memref<10x10xf32>
    %output = memref.alloc() : memref<10x10xf32>

    // Launch the kernel with 10x10 grid of blocks, each with 10x10 threads
    gpu.launch_func @kernel_module::@square_kernel
      blocks in (%c10, %c1, %c1)
      threads in (%c10, %c1, %c1)
      args(%input : memref<10x10xf32>, %output : memref<10x10xf32>)

    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
